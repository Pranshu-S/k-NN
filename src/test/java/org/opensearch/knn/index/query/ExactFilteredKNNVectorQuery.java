/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import com.google.common.annotations.VisibleForTesting;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.index.query.common.QueryUtils;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher;
import org.opensearch.knn.index.query.exactsearch.ExactSearcher.ExactSearcherContext;
import org.opensearch.knn.indices.ModelDao;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;

/**
 * Test/benchmark-only standalone exact (brute-force) filtered top-k search for Lucene-engine FLOAT
 * vector fields. It re-evaluates the filter itself, so it serves as the "raw exact" baseline in the
 * benchmarks. Production does NOT use this class — {@link BoundedExactKnnFloatVectorQuery} performs
 * exact search inline, reusing the doc ids collected during bounded counting (no second filter pass).
 *
 * <p>Every live document matched by the filter is scored against the query vector and the top
 * {@code k} are returned, via the plugin's existing exact scorer ({@link ExactSearcher}) — the same
 * component the rescore path drives — so scores are identical to current k-NN behavior.
 *
 * <p>Per leaf the flow is: build the filter {@link Weight}, materialize a {@link BitSet} of matching
 * documents intersected with the segment's live docs (deleted documents are excluded, mirroring
 * {@code KNNWeight#getFilteredDocsBitSet}), then hand that iterator to {@link ExactSearcher}. Results
 * are merged to {@code k} across segments. When the filter matches nothing the query degrades to
 * {@link MatchNoDocsQuery}. When fewer than {@code k} documents match, all of them are returned
 * (the exact searcher scores every candidate when the matched set is {@code <= k}).
 */
@Log4j2
final class ExactFilteredKNNVectorQuery extends Query {

    private final Query filterQuery;
    private final String field;
    private final int k;
    private final float[] queryVector;
    // Shard id is carried for debug logging only; the query itself is shard-agnostic.
    private final int shardId;
    private final ExactSearcher exactSearcher;

    ExactFilteredKNNVectorQuery(final Query filterQuery, final String field, final int k, final float[] queryVector, final int shardId) {
        this(filterQuery, field, k, queryVector, shardId, new ExactSearcher(ModelDao.OpenSearchKNNModelDao.getInstance()));
    }

    @VisibleForTesting
    ExactFilteredKNNVectorQuery(
        final Query filterQuery,
        final String field,
        final int k,
        final float[] queryVector,
        final int shardId,
        final ExactSearcher exactSearcher
    ) {
        this.filterQuery = filterQuery;
        this.field = field;
        this.k = k;
        this.queryVector = queryVector;
        this.shardId = shardId;
        this.exactSearcher = exactSearcher;
    }

    @Override
    public Weight createWeight(final IndexSearcher searcher, final ScoreMode scoreMode, final float boost) throws IOException {
        // The filter is only used to enumerate candidate docs; scores come from the vector scorer.
        final Query rewrittenFilter = searcher.rewrite(filterQuery);
        final Weight filterWeight = searcher.createWeight(rewrittenFilter, ScoreMode.COMPLETE_NO_SCORES, 1f);
        final TopDocs[] perLeafResults = searchAllLeaves(searcher, filterWeight);
        final TopDocs topK = TopDocs.merge(k, perLeafResults);
        if (topK.scoreDocs.length == 0) {
            return new MatchNoDocsQuery("k-NN exact filtered search matched no documents").createWeight(searcher, scoreMode, boost);
        }
        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), topK).createWeight(searcher, scoreMode, boost);
    }

    private TopDocs[] searchAllLeaves(final IndexSearcher indexSearcher, final Weight filterWeight) throws IOException {
        final List<LeafReaderContext> leaves = indexSearcher.getIndexReader().leaves();
        final List<Callable<TopDocs>> tasks = new ArrayList<>(leaves.size());
        for (final LeafReaderContext leaf : leaves) {
            tasks.add(() -> searchLeaf(filterWeight, leaf));
        }
        return indexSearcher.getTaskExecutor().invokeAll(tasks).toArray(TopDocs[]::new);
    }

    private TopDocs searchLeaf(final Weight filterWeight, final LeafReaderContext leaf) throws IOException {
        final Scorer scorer = filterWeight.scorer(leaf);
        if (scorer == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        // Intersect filter matches with live docs so deleted documents are never scored.
        final Bits liveDocs = leaf.reader().getLiveDocs();
        final BitSet matchingDocs = buildLiveMatchBitSet(scorer.iterator(), liveDocs, leaf.reader().maxDoc());
        final int cardinality = matchingDocs.cardinality();
        if (cardinality == 0) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        final ExactSearcherContext exactSearcherContext = ExactSearcherContext.builder()
            .matchedDocsIterator(new BitSetIterator(matchingDocs, cardinality))
            .numberOfMatchedDocs(cardinality)
            // Use full-precision float vectors (Lucene engine is not quantized in the KNN sense).
            .useQuantizedVectorsForSearch(false)
            .k(k)
            .field(field)
            .floatQueryVector(queryVector)
            .build();

        final TopDocs results = exactSearcher.searchLeaf(leaf, exactSearcherContext);
        if (leaf.docBase > 0) {
            for (final ScoreDoc scoreDoc : results.scoreDocs) {
                scoreDoc.doc += leaf.docBase;
            }
        }
        return results;
    }

    private static BitSet buildLiveMatchBitSet(final DocIdSetIterator matches, final Bits liveDocs, final int maxDoc) throws IOException {
        final FilteredDocIdSetIterator liveMatches = new FilteredDocIdSetIterator(matches) {
            @Override
            protected boolean match(final int doc) {
                return liveDocs == null || liveDocs.get(doc);
            }
        };
        return BitSet.of(liveMatches, maxDoc);
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
        final ExactFilteredKNNVectorQuery other = (ExactFilteredKNNVectorQuery) obj;
        return k == other.k
            && shardId == other.shardId
            && Objects.equals(field, other.field)
            && Objects.equals(filterQuery, other.filterQuery)
            && Objects.deepEquals(queryVector, other.queryVector);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), field, k, filterQuery, shardId, Arrays.hashCode(queryVector));
    }
}
