/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

/**
 * Strategy chosen by the {@link BoundedExactSearchDecider} for a <em>filtered</em> top-k vector query.
 *
 * <p>Scoped to the Lucene engine, FLOAT vectors, non-nested fields, top-k queries and rescore-disabled
 * requests; every other configuration bypasses the decider entirely.
 */
enum KNNFilterExecutionStrategy {
    /**
     * Run the existing filtered approximate-nearest-neighbor (ANN) path unchanged. This is also the
     * fallback whenever the request is outside the supported MVP surface or shard-local cardinality
     * cannot be estimated.
     */
    APPROXIMATE,

    /**
     * Score every filtered, live document exactly and return the top-k. Chosen for highly selective
     * filters where a brute-force scan is expected to be faster and perfectly recall-accurate.
     */
    EXACT,

    /**
     * The filter matches zero live documents on this shard, so the query is replaced with a
     * {@code MatchNoDocsQuery} and no vector work is performed.
     */
    MATCH_NONE
}
