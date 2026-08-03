/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.knn.index.query;

import lombok.Getter;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Locale;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.FILTERED_SEARCH_MODE_ACORN;
import static org.opensearch.knn.common.KNNConstants.FILTERED_SEARCH_MODE_STANDARD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_FILTERED_SEARCH_MODE;

/**
 * EXPERIMENTAL (POC). Filtered HNSW traversal policy for the native Faiss engine.
 *
 * <ul>
 *   <li>{@link #STANDARD} — the existing stock Faiss HNSW filtered traversal
 *       (the beam routes through predicate-rejected nodes; selector gates only
 *       the result heap).</li>
 *   <li>{@link #ACORN} — selector-aware predicate-subgraph traversal: only
 *       predicate-passing nodes enter the result heap; rejected nodes act as
 *       two-hop routing bridges. Intended for selective, vector-locality
 *       correlated filters.</li>
 * </ul>
 *
 * The wire value travels in the query {@code method_parameters} map under
 * {@code filtered_search_mode} and is read natively in
 * {@code jni/src/faiss_wrapper.cpp}. There is NO automatic strategy selection:
 * behaviour is entirely explicit. Omitting the parameter preserves existing
 * behaviour (equivalent to {@link #STANDARD}).
 */
@Getter
public enum FilteredSearchMode {
    STANDARD(FILTERED_SEARCH_MODE_STANDARD),
    ACORN(FILTERED_SEARCH_MODE_ACORN);

    private static final String ALLOWED = "[standard, acorn]";

    private final String wireName;

    FilteredSearchMode(final String wireName) {
        this.wireName = wireName;
    }

    /** True for any selector-aware traversal (everything except {@link #STANDARD}). */
    public boolean isSelectorAware() {
        return this != STANDARD;
    }

    /**
     * Parse the canonical wire value. Unknown values fail (no silent fallback).
     * @throws IllegalArgumentException on any value not in {@value #ALLOWED}.
     */
    public static FilteredSearchMode fromWireName(final String value) {
        if (value == null) {
            throw new IllegalArgumentException(METHOD_PARAMETER_FILTERED_SEARCH_MODE + " must be one of " + ALLOWED);
        }
        for (final FilteredSearchMode mode : values()) {
            if (mode.wireName.equals(value)) {
                return mode;
            }
        }
        throw new IllegalArgumentException(
            String.format(
                Locale.ROOT,
                "%s must be one of %s but was [%s]",
                METHOD_PARAMETER_FILTERED_SEARCH_MODE,
                ALLOWED,
                value
            )
        );
    }

    /**
     * Extract the mode from a method-parameters map. Absent → {@link #STANDARD}
     * (existing behaviour). Present → validated (unknown values throw).
     */
    public static FilteredSearchMode fromMethodParameters(final Map<String, ?> methodParameters) {
        if (methodParameters == null) {
            return STANDARD;
        }
        final Object raw = methodParameters.get(METHOD_PARAMETER_FILTERED_SEARCH_MODE);
        if (raw == null) {
            return STANDARD;
        }
        return fromWireName(String.valueOf(raw));
    }

    /**
     * Query-time validation of an explicit {@code acorn} selection. These checks
     * need query context (engine, method, filter, query type) that is not
     * available at the REST method-parameter layer, so they run in
     * {@code KNNQueryBuilder#doToQuery} once the engine is resolved. There is
     * NO silent fallback: an invalid {@code acorn} request is rejected.
     *
     * @param methodParameters the query method parameters (may be null)
     * @param knnEngine        resolved engine for the target field
     * @param method           resolved method name (e.g. "hnsw"), may be null
     * @param hasFilter        whether the query carries a filter
     * @param isRadial         whether this is a radial (min_score/max_distance) query
     */
    public static void validateForQuery(
        final Map<String, ?> methodParameters,
        final KNNEngine knnEngine,
        final String method,
        final boolean hasFilter,
        final boolean isRadial
    ) {
        final FilteredSearchMode mode = fromMethodParameters(methodParameters);
        if (!mode.isSelectorAware()) { // STANDARD (or omitted): existing behaviour, no constraints
            return;
        }
        final String m = mode.getWireName();
        if (knnEngine != KNNEngine.FAISS) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "%s [%s] is only supported for the [%s] engine, but field uses [%s]",
                    METHOD_PARAMETER_FILTERED_SEARCH_MODE,
                    m,
                    KNNEngine.FAISS.getName(),
                    knnEngine.getName()
                )
            );
        }
        if (method != null && !METHOD_HNSW.equalsIgnoreCase(method)) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "%s [%s] is only supported for the [%s] method, but field uses [%s]",
                    METHOD_PARAMETER_FILTERED_SEARCH_MODE,
                    m,
                    METHOD_HNSW,
                    method
                )
            );
        }
        if (isRadial) {
            throw new IllegalArgumentException(
                METHOD_PARAMETER_FILTERED_SEARCH_MODE + " [" + m + "] is not supported for radial (min_score/max_distance) queries"
            );
        }
        if (!hasFilter) {
            throw new IllegalArgumentException(
                METHOD_PARAMETER_FILTERED_SEARCH_MODE + " [" + m + "] requires a filter; add a filter or omit the parameter"
            );
        }
    }
}
