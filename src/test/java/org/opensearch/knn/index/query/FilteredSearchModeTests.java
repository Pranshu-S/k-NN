/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.knn.index.query;

import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.query.request.MethodParameter;

import java.util.HashMap;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_FILTERED_SEARCH_MODE;
import static org.opensearch.knn.index.query.parser.MethodParametersParser.validateMethodParameters;

/** Unit tests for the experimental {@code filtered_search_mode} POC parameter. */
public class FilteredSearchModeTests extends KNNTestCase {

    public void testFromWireName() {
        assertEquals(FilteredSearchMode.STANDARD, FilteredSearchMode.fromWireName("standard"));
        assertEquals(FilteredSearchMode.ACORN, FilteredSearchMode.fromWireName("acorn"));
    }

    public void testSelectorAwareFlag() {
        assertFalse(FilteredSearchMode.STANDARD.isSelectorAware());
        assertTrue(FilteredSearchMode.ACORN.isSelectorAware());
    }

    public void testFromWireNameUnknownFails() {
        expectThrows(IllegalArgumentException.class, () -> FilteredSearchMode.fromWireName("ACORN"));   // case-sensitive
        expectThrows(IllegalArgumentException.class, () -> FilteredSearchMode.fromWireName("hybrid"));
        expectThrows(IllegalArgumentException.class, () -> FilteredSearchMode.fromWireName(null));
    }

    public void testFromMethodParametersDefaultsToStandard() {
        assertEquals(FilteredSearchMode.STANDARD, FilteredSearchMode.fromMethodParameters(null));
        assertEquals(FilteredSearchMode.STANDARD, FilteredSearchMode.fromMethodParameters(Map.of("ef_search", 100)));
        assertEquals(
            FilteredSearchMode.ACORN,
            FilteredSearchMode.fromMethodParameters(Map.of(METHOD_PARAMETER_FILTERED_SEARCH_MODE, "acorn"))
        );
    }

    public void testMethodParameterParseCanonicalizesAndRejects() {
        // valid values pass rest-layer validation
        assertNull(validateMethodParameters(Map.of(METHOD_PARAMETER_FILTERED_SEARCH_MODE, "acorn")));
        assertNull(validateMethodParameters(Map.of(METHOD_PARAMETER_FILTERED_SEARCH_MODE, "standard")));
        // unknown value is rejected at the rest layer (no silent fallback)
        final ValidationException ve = validateMethodParameters(Map.of(METHOD_PARAMETER_FILTERED_SEARCH_MODE, "nope"));
        assertNotNull(ve);
        assertTrue(ve.getMessage().contains(METHOD_PARAMETER_FILTERED_SEARCH_MODE));
        // parse canonicalizes to the wire value
        assertEquals("acorn", MethodParameter.FILTERED_SEARCH_MODE.parse("acorn"));
    }

    public void testValidateForQueryStandardAlwaysOk() {
        // omitted / standard never fails, regardless of engine / filter / query type
        FilteredSearchMode.validateForQuery(null, KNNEngine.LUCENE, "hnsw", false, true);
        FilteredSearchMode.validateForQuery(
            Map.of(METHOD_PARAMETER_FILTERED_SEARCH_MODE, "standard"),
            KNNEngine.LUCENE,
            "hnsw",
            false,
            false
        );
    }

    public void testValidateForQueryAcornHappyPath() {
        // acorn + faiss + hnsw + filter + non-radial → no exception
        FilteredSearchMode.validateForQuery(acorn(), KNNEngine.FAISS, "hnsw", true, false);
    }

    public void testValidateForQueryAcornRequiresFilter() {
        final IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> FilteredSearchMode.validateForQuery(acorn(), KNNEngine.FAISS, "hnsw", false, false)
        );
        assertTrue(e.getMessage().contains("requires a filter"));
    }

    public void testValidateForQueryAcornRequiresFaiss() {
        final IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> FilteredSearchMode.validateForQuery(acorn(), KNNEngine.LUCENE, "hnsw", true, false)
        );
        assertTrue(e.getMessage().contains("engine"));
    }

    public void testValidateForQueryAcornRequiresHnsw() {
        final IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> FilteredSearchMode.validateForQuery(acorn(), KNNEngine.FAISS, "ivf", true, false)
        );
        assertTrue(e.getMessage().contains("method"));
    }

    public void testValidateForQueryAcornRejectsRadial() {
        final IllegalArgumentException e = expectThrows(
            IllegalArgumentException.class,
            () -> FilteredSearchMode.validateForQuery(acorn(), KNNEngine.FAISS, "hnsw", true, true)
        );
        assertTrue(e.getMessage().contains("radial"));
    }

    private static Map<String, ?> acorn() {
        final Map<String, Object> m = new HashMap<>();
        m.put(METHOD_PARAMETER_FILTERED_SEARCH_MODE, "acorn");
        return m;
    }
}
