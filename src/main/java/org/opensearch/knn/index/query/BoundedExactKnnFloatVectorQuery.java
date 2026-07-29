/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher.ExactSearcherContext;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Execution-time wrapper that plans a filtered top-k Lucene FLOAT k-NN query at {@code createWeight}
 * time (Option B of the design). It is produced by {@link KNNQueryFactory} only for the supported MVP
 * surface (Lucene engine, FLOAT vectors, non-nested, top-k, filter present, rescore disabled); every
 * other configuration keeps its existing query and never reaches here.
 *
 * <h2>Why execute-time (not query-factory) planning</h2>
 * Bounded filter counting and, crucially, <em>reuse</em> of the collected filter state require the
 * authoritative {@link IndexSearcher} and a single, stable reader generation — both of which are only
 * guaranteed at {@code createWeight}. Planning in the query factory would either force a second filter
 * evaluation or risk mixing reader generations. Deferring to here lets the exact path reuse the exact
 * doc ids gathered while counting.
 *
 * <h2>Flow</h2>
 * <ol>
 *   <li>Build the filter {@link Weight} once and collect matches with {@link BoundedFilterResult},
 *       stopping at {@code threshold + 1}.</li>
 *   <li>{@code 0} matches → {@link MatchNoDocsQuery}.</li>
 *   <li>{@code [1, threshold]} matches → exact search over the <em>already collected</em> live doc ids
 *       (no second filter pass), via the existing {@link ExactSearcher}.</li>
 *   <li>{@code > threshold} matches → delegate to the unchanged approximate query
 *       ({@code annDelegate}); the collected partial state is released and the ANN query re-evaluates
 *       the filter itself.</li>
 * </ol>
 *
 * <p>The wrapper is immutable and thread-safe; all mutable planning state is local to a
 * {@code createWeight} invocation.
 */
@Log4j2
final class BoundedExactKnnFloatVectorQuery extends Query {

    private final Query annDelegate;
    private final Query filterQuery;
    private final String field;
    private final int k;
    private final float[] queryVector;
    private final int shardId;
    private final ExactSearcher exactSearcher;
    // Optional test probe; null in production.
    @Nullable
    private final KNNFilterPlanningStats statsSink;

    BoundedExactKnnFloatVectorQuery(
        final Query annDelegate,
        final Query filterQuery,
        final String field,
        final int k,
        final float[] queryVector,
        final int shardId
    ) {
        this(
            annDelegate,
            filterQuery,
            field,
            k,
            queryVector,
            shardId,
            new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance()),
            null
        );
    }

    @VisibleForTesting
    BoundedExactKnnFloatVectorQuery(
        final Query annDelegate,
        final Query filterQuery,
        final String field,
        final int k,
        final float[] queryVector,
        final int shardId,
        final ExactSearcher exactSearcher,
        @Nullable final KNNFilterPlanningStats statsSink
    ) {
        this.annDelegate = annDelegate;
        this.filterQuery = filterQuery;
        this.field = field;
        this.k = k;
        this.queryVector = queryVector;
        this.shardId = shardId;
        this.exactSearcher = exactSearcher;
        this.statsSink = statsSink;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        final long threshold = BoundedExactSearchDecider.threshold(k);

        // One filter evaluation, bounded to threshold + 1 matches, on this reader generation.
        final Query rewrittenFilter = searcher.rewrite(filterQuery);
        final Weight filterWeight = searcher.createWeight(rewrittenFilter, ScoreMode.COMPLETE_NO_SCORES, 1f);
        final BoundedFilterResult bounded = BoundedFilterResult.collect(searcher, filterWeight, threshold);

        final KNNFilterExecutionStrategy strategy = BoundedExactSearchDecider.strategyForBoundedCount(
            bounded.getObservedMatches(),
            bounded.isThresholdExceeded()
        );

        log.debug(
            "Bounded-exact k-NN decision shard:{}, field:{}, k:{}, observedMatches:{}, limitExceeded:{}, candidateLimit:{}, "
                + "filterAdvances:{}, strategy:{}",
            shardId,
            field,
            k,
            bounded.getObservedMatches(),
            bounded.isThresholdExceeded(),
            threshold,
            bounded.getFilterAdvances(),
            strategy
        );

        recordStats(strategy, bounded, threshold);

        switch (strategy) {
            case MATCH_NONE:
                bounded.release();
                return new MatchNoDocsQuery("k-NN filter matched no documents on shard").createWeight(searcher, scoreMode, boost);
            case EXACT:
                return createExactWeight(searcher, scoreMode, boost, bounded);
            case APPROXIMATE:
            default:
                bounded.release();
                if (statsSink != null) {
                    statsSink.setAnnDelegateExecuted(true);
                }
                // Delegate to the unchanged approximate query; it re-evaluates the filter itself.
                return searcher.rewrite(annDelegate).createWeight(searcher, scoreMode, boost);
        }
    }

    private Weight createExactWeight(
        final IndexSearcher searcher,
        final ScoreMode scoreMode,
        final float boost,
        final BoundedFilterResult bounded
    ) throws IOException {
        final TopDocs[] perLeafResults = exactSearchOverCollected(bounded);
        bounded.release();
        final TopDocs topK = TopDocs.merge(k, perLeafResults);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery("k-NN exact filtered search matched no documents").createWeight(searcher, scoreMode, boost);
        }
        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), topK).createWeight(searcher, scoreMode, boost);
    }

    /**
     * Exact search that reuses the per-segment doc ids gathered during bounded counting — the filter is
     * NOT evaluated again. Each leaf's ascending live matches are wrapped in a lightweight
     * {@link IntArrayDocIdSetIterator} and handed to the existing {@link ExactSearcher}. No
     * {@code FixedBitSet(maxDoc)} is allocated, so the exact path's memory is {@code O(matches)} rather
     * than {@code O(maxDoc)} — a large saving for sparse filters over big segments.
     */
    private TopDocs[] exactSearchOverCollected(final BoundedFilterResult bounded) throws IOException {
        final List<BoundedFilterResult.LeafMatches> leafMatches = bounded.getLeafMatches();
        final List<TopDocs> results = new ArrayList<>(leafMatches.size());
        for (final BoundedFilterResult.LeafMatches leafMatch : leafMatches) {
            final LeafReaderContext leaf = leafMatch.getLeaf();
            final int count = leafMatch.getCount();

            final ExactSearcherContext exactSearcherContext = ExactSearcherContext.builder()
                .matchedDocsIterator(new IntArrayDocIdSetIterator(leafMatch.getDocIds(), count))
                .numberOfMatchedDocs(count)
                .useQuantizedVectorsForSearch(false)
                .k(k)
                .field(field)
                .floatQueryVector(queryVector)
                .build();

            final TopDocs leafResult = exactSearcher.searchLeaf(leaf, exactSearcherContext);
            if (leaf.docBase > 0) {
                for (final ScoreDoc scoreDoc : leafResult.scoreDocs) {
                    scoreDoc.doc += leaf.docBase;
                }
            }
            results.add(leafResult);
        }
        if (results.isEmpty()) {
            return new TopDocs[] { TopDocsCollector.EMPTY_TOPDOCS };
        }
        return results.toArray(TopDocs[]::new);
    }

    private void recordStats(final KNNFilterExecutionStrategy strategy, final BoundedFilterResult bounded, final long threshold) {
        if (statsSink == null) {
            return;
        }
        statsSink.setStrategy(strategy);
        statsSink.setObservedMatches(bounded.getObservedMatches());
        statsSink.setThresholdExceeded(bounded.isThresholdExceeded());
        statsSink.setThreshold(threshold);
        statsSink.setFilterAdvances(bounded.getFilterAdvances());
        statsSink.setFilterScorersCreated(bounded.getFilterScorersCreated());
        // Exact scores every collected live match once.
        statsSink.setExactVectorsScored(strategy == KNNFilterExecutionStrategy.EXACT ? bounded.getObservedMatches() : 0);
    }

    @Override
    public String toString(final String field) {
        return getClass().getSimpleName() + "[field=" + this.field + ", k=" + k + ", filter=" + filterQuery + ", shardId=" + shardId + "]";
    }

    @Override
    public void visit(final QueryVisitor visitor) {
        visitor.visitLeaf(this);
    }

    @Override
    public boolean equals(final Object obj) {
        if (!sameClassAs(obj)) {
            return false;
        }
        final BoundedExactKnnFloatVectorQuery other = (BoundedExactKnnFloatVectorQuery) obj;
        return k == other.k
            && shardId == other.shardId
            && Objects.equals(field, other.field)
            && Objects.equals(annDelegate, other.annDelegate)
            && Objects.equals(filterQuery, other.filterQuery)
            && Arrays.equals(queryVector, other.queryVector);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, k, annDelegate, filterQuery, shardId, Arrays.hashCode(queryVector));
    }
}
