/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query;

/**
 * Mutable, single-use probe that {@link BoundedExactKnnFloatVectorQuery} fills in during weight
 * creation so tests can assert on planning behavior (strategy chosen, bounded-count early termination,
 * filter reuse). Production passes {@code null} and pays only a null-check, keeping observability
 * lightweight — the debug log line remains the production-facing signal.
 */
final class KNNFilterPlanningStats {
    private KNNFilterExecutionStrategy strategy;
    private long observedMatches;
    private boolean thresholdExceeded;
    private long threshold;
    private long filterAdvances;
    private int filterScorersCreated;
    private long exactVectorsScored;
    private boolean annDelegateExecuted;

    KNNFilterExecutionStrategy getStrategy() {
        return strategy;
    }

    void setStrategy(final KNNFilterExecutionStrategy strategy) {
        this.strategy = strategy;
    }

    long getObservedMatches() {
        return observedMatches;
    }

    void setObservedMatches(final long observedMatches) {
        this.observedMatches = observedMatches;
    }

    boolean isThresholdExceeded() {
        return thresholdExceeded;
    }

    void setThresholdExceeded(final boolean thresholdExceeded) {
        this.thresholdExceeded = thresholdExceeded;
    }

    long getThreshold() {
        return threshold;
    }

    void setThreshold(final long threshold) {
        this.threshold = threshold;
    }

    long getFilterAdvances() {
        return filterAdvances;
    }

    void setFilterAdvances(final long filterAdvances) {
        this.filterAdvances = filterAdvances;
    }

    int getFilterScorersCreated() {
        return filterScorersCreated;
    }

    void setFilterScorersCreated(final int filterScorersCreated) {
        this.filterScorersCreated = filterScorersCreated;
    }

    long getExactVectorsScored() {
        return exactVectorsScored;
    }

    void setExactVectorsScored(final long exactVectorsScored) {
        this.exactVectorsScored = exactVectorsScored;
    }

    boolean isAnnDelegateExecuted() {
        return annDelegateExecuted;
    }

    void setAnnDelegateExecuted(final boolean annDelegateExecuted) {
        this.annDelegateExecuted = annDelegateExecuted;
    }
}
