/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Benchmark-only (test-source) extractor of candidate filter-locality features from the bounded filter
 * matches. Kept out of production {@code src/main} deliberately: per iteration-4 Phase 8, locality
 * signals are integrated into the production planner only if they generalize to held-out / adversarial
 * distributions, which this class exists to test.
 *
 * <p>Group A (segment distribution) and Group B (document-id scatter) are computed purely from the
 * collected per-leaf doc ids — no vector reads. Group C (vector-space dispersion) reads a small bounded,
 * deterministic sample of vectors via {@link FloatVectorValues} using the field's mapped similarity, and
 * is therefore more expensive.
 *
 * <p><b>Interpretation warning:</b> document-id metrics (Group B) are only meaningful if document
 * ordering happens to preserve dataset structure; they are experimental proxies and must be validated
 * under shuffled insertion order (see the adversarial distributions in the benchmark).
 */
final class FilterLocalityFeatures {

    private FilterLocalityFeatures() {}

    /** Per-leaf ascending sample of matched doc ids (retained even past a threshold, for study). */
    static final class LeafSample {
        final LeafReaderContext leaf;
        final int[] docIds;
        final int count;

        LeafSample(final LeafReaderContext leaf, final int[] docIds, final int count) {
            this.leaf = leaf;
            this.docIds = docIds;
            this.count = count;
        }
    }

    /** Group A + B features (query-independent), computed from the collected sample. */
    static final class ScatterFeatures {
        final long sampledMatches;
        final int numLeavesWithMatches;
        final double fracLeavesWithMatches;
        final double maxLeafConcentration; // max share of matches in a single leaf
        final double leafEntropy;          // normalized Shannon entropy of per-leaf distribution
        final double meanPerLeaf;
        final double varPerLeaf;
        final double weightedNormalizedSpan; // Group B, count-weighted across leaves
        final double weightedMeanGap;
        final double weightedMaxGap;
        final double fracInContiguousRuns;   // matches belonging to runs of length >= 2
        final double contiguousRunDensity;   // runs / matches (low => long runs / contiguous)

        ScatterFeatures(
            long sampledMatches,
            int numLeavesWithMatches,
            double fracLeavesWithMatches,
            double maxLeafConcentration,
            double leafEntropy,
            double meanPerLeaf,
            double varPerLeaf,
            double weightedNormalizedSpan,
            double weightedMeanGap,
            double weightedMaxGap,
            double fracInContiguousRuns,
            double contiguousRunDensity
        ) {
            this.sampledMatches = sampledMatches;
            this.numLeavesWithMatches = numLeavesWithMatches;
            this.fracLeavesWithMatches = fracLeavesWithMatches;
            this.maxLeafConcentration = maxLeafConcentration;
            this.leafEntropy = leafEntropy;
            this.meanPerLeaf = meanPerLeaf;
            this.varPerLeaf = varPerLeaf;
            this.weightedNormalizedSpan = weightedNormalizedSpan;
            this.weightedMeanGap = weightedMeanGap;
            this.weightedMaxGap = weightedMaxGap;
            this.fracInContiguousRuns = fracInContiguousRuns;
            this.contiguousRunDensity = contiguousRunDensity;
        }
    }

    /** Group C features (query-dependent), in the field's Euclidean metric. */
    static final class VectorDispersion {
        final int sampleSize;
        final double meanDistToQuery;
        final double minDistToQuery;
        final double maxDistToQuery;
        final double varDistToQuery;
        final double meanDistToCentroid; // spatial dispersion of the filtered sample

        VectorDispersion(
            int sampleSize,
            double meanDistToQuery,
            double minDistToQuery,
            double maxDistToQuery,
            double varDistToQuery,
            double meanDistToCentroid
        ) {
            this.sampleSize = sampleSize;
            this.meanDistToQuery = meanDistToQuery;
            this.minDistToQuery = minDistToQuery;
            this.maxDistToQuery = maxDistToQuery;
            this.varDistToQuery = varDistToQuery;
            this.meanDistToCentroid = meanDistToCentroid;
        }
    }

    /**
     * Collects up to {@code cap} live matches (ascending per leaf), retaining them even when the cap is
     * reached — unlike {@link BoundedFilterResult}, which is tuned for the planner and releases on
     * exceed. This mirrors what a bounded planner pass could keep for feature extraction.
     */
    static List<LeafSample> collectSample(final IndexSearcher searcher, final Weight filterWeight, final int cap) throws IOException {
        final List<LeafSample> out = new ArrayList<>();
        long observed = 0;
        outer: for (final LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
            final Scorer scorer = filterWeight.scorer(leaf);
            if (scorer == null) {
                continue;
            }
            final Bits liveDocs = leaf.reader().getLiveDocs();
            final DocIdSetIterator it = scorer.iterator();
            int[] docIds = new int[16];
            int count = 0;
            for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
                if (liveDocs != null && liveDocs.get(doc) == false) {
                    continue;
                }
                if (count == docIds.length) {
                    docIds = java.util.Arrays.copyOf(docIds, docIds.length * 2);
                }
                docIds[count++] = doc;
                if (++observed >= cap) {
                    out.add(new LeafSample(leaf, docIds, count));
                    break outer;
                }
            }
            if (count > 0) {
                out.add(new LeafSample(leaf, docIds, count));
            }
        }
        return out;
    }

    static ScatterFeatures scatterFeatures(final List<LeafSample> sample, final int totalLeaves) {
        long total = 0;
        for (final LeafSample ls : sample) {
            total += ls.count;
        }
        if (total == 0) {
            return new ScatterFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }
        // Group A: per-leaf distribution.
        double maxConc = 0;
        double entropy = 0;
        double sumPerLeaf = 0;
        double sumSqPerLeaf = 0;
        for (final LeafSample ls : sample) {
            final double p = (double) ls.count / total;
            maxConc = Math.max(maxConc, p);
            if (p > 0) {
                entropy -= p * Math.log(p);
            }
            sumPerLeaf += ls.count;
            sumSqPerLeaf += (double) ls.count * ls.count;
        }
        final int leavesWithMatches = sample.size();
        final double meanPerLeaf = sumPerLeaf / leavesWithMatches;
        final double varPerLeaf = sumSqPerLeaf / leavesWithMatches - meanPerLeaf * meanPerLeaf;
        final double normEntropy = leavesWithMatches > 1 ? entropy / Math.log(leavesWithMatches) : 0.0;

        // Group B: doc-id scatter, count-weighted across leaves.
        double wSpan = 0, wMeanGap = 0, wMaxGap = 0, wRunFrac = 0, wRunDensity = 0;
        for (final LeafSample ls : sample) {
            final double w = (double) ls.count / total;
            if (ls.count == 1) {
                // single doc: no gaps; treat as maximally contiguous
                wRunFrac += w * 0.0;
                wRunDensity += w * 1.0;
                continue;
            }
            final int span = ls.docIds[ls.count - 1] - ls.docIds[0];
            final int leafMax = Math.max(1, ls.leaf.reader().maxDoc());
            wSpan += w * ((double) span / leafMax);
            long sumGap = 0;
            int maxGap = 0;
            int runs = 1; // number of maximal ascending-contiguous runs
            int inRun = 0; // matches that are part of a run of length >= 2
            int currentRun = 1;
            for (int i = 1; i < ls.count; i++) {
                final int gap = ls.docIds[i] - ls.docIds[i - 1];
                sumGap += gap;
                maxGap = Math.max(maxGap, gap);
                if (gap == 1) {
                    currentRun++;
                } else {
                    if (currentRun >= 2) {
                        inRun += currentRun;
                    }
                    runs++;
                    currentRun = 1;
                }
            }
            if (currentRun >= 2) {
                inRun += currentRun;
            }
            wMeanGap += w * ((double) sumGap / (ls.count - 1));
            wMaxGap += w * maxGap;
            wRunFrac += w * ((double) inRun / ls.count);
            wRunDensity += w * ((double) runs / ls.count);
        }

        return new ScatterFeatures(
            total,
            leavesWithMatches,
            (double) leavesWithMatches / Math.max(1, totalLeaves),
            maxConc,
            normEntropy,
            meanPerLeaf,
            varPerLeaf,
            wSpan,
            wMeanGap,
            wMaxGap,
            wRunFrac,
            wRunDensity
        );
    }

    /**
     * Group C: reads a bounded, deterministic sample of at most {@code sampleSize} matched vectors and
     * computes Euclidean dispersion relative to {@code queryVector} and the sample centroid.
     */
    static VectorDispersion vectorDispersion(
        final List<LeafSample> sample,
        final String field,
        final float[] queryVector,
        final int sampleSize
    ) throws IOException {
        // Flatten (leaf, docId) pairs, then pick evenly spaced indices for determinism and spread.
        final List<int[]> flat = new ArrayList<>(); // [leafOrdInList, docId]
        for (int li = 0; li < sample.size(); li++) {
            final LeafSample ls = sample.get(li);
            for (int i = 0; i < ls.count; i++) {
                flat.add(new int[] { li, ls.docIds[i] });
            }
        }
        if (flat.isEmpty()) {
            return new VectorDispersion(0, 0, 0, 0, 0, 0);
        }
        final int take = Math.min(sampleSize, flat.size());
        final int stride = Math.max(1, flat.size() / take);
        final List<float[]> vectors = new ArrayList<>(take);
        for (int idx = 0; idx < flat.size() && vectors.size() < take; idx += stride) {
            final int[] pair = flat.get(idx);
            final LeafSample ls = sample.get(pair[0]);
            final FloatVectorValues fvv = ls.leaf.reader().getFloatVectorValues(field);
            if (fvv == null) {
                continue;
            }
            final KnnVectorValues.DocIndexIterator it = fvv.iterator();
            if (it.advance(pair[1]) == pair[1]) {
                vectors.add(fvv.vectorValue(it.index()).clone());
            }
        }
        final int n = vectors.size();
        if (n == 0) {
            return new VectorDispersion(0, 0, 0, 0, 0, 0);
        }
        double sum = 0, sumSq = 0, min = Double.MAX_VALUE, max = 0;
        final float[] centroid = new float[queryVector.length];
        for (final float[] v : vectors) {
            final double d = l2(queryVector, v);
            sum += d;
            sumSq += d * d;
            min = Math.min(min, d);
            max = Math.max(max, d);
            for (int j = 0; j < centroid.length; j++) {
                centroid[j] += v[j];
            }
        }
        for (int j = 0; j < centroid.length; j++) {
            centroid[j] /= n;
        }
        double dispSum = 0;
        for (final float[] v : vectors) {
            dispSum += l2(centroid, v);
        }
        final double mean = sum / n;
        final double var = sumSq / n - mean * mean;
        return new VectorDispersion(n, mean, min, max, var, dispSum / n);
    }

    /** Euclidean distance — the metric behind {@link VectorSimilarityFunction#EUCLIDEAN}; not a new impl. */
    private static double l2(final float[] a, final float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            final double d = a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }
}
