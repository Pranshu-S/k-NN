/*
 * ACORN-γ research prototype for native Faiss HNSW (OpenSearch k-NN).
 *
 * This header declares a self-contained, selector-aware ACORN traversal
 * (Phase 2) and an ACORN-γ graph builder (Phase 3) that operate directly on
 * faiss 1.11.0's `faiss::HNSW` link structure and an `faiss::IndexFlat`
 * storage. It intentionally does NOT depend on faiss::IndexHNSW/IndexIDMap so
 * that the prototype links against a minimal, deterministic (single-thread)
 * subset of Faiss.
 *
 * Faithful to the ACORN reference implementation (TAG-Research/ACORN, a Faiss
 * fork) but adapted to:
 *   - reuse faiss::IDSelector::is_member(id) as THE filter predicate (matching
 *     the OpenSearch efficient-filter path) instead of a dense char* map;
 *   - store the graph as a plain faiss::HNSW (one graph type for standard and
 *     ACORN-γ), so the ACORN traversal runs over EITHER graph.
 *
 * TEST-ONLY. Not a production API. See research/acorn/ACORN_GAMMA_DESIGN.md.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/IDSelector.h>

namespace acorn {

// Internal research search mode (Phase 2 requirement).
enum class FilteredHnswSearchMode {
    STANDARD, // stock faiss HNSW beam search (faiss::HNSW::search)
    ACORN     // ACORN predicate-subgraph traversal (this prototype)
};

// Graph construction descriptor. gamma==1 => standard-density control; the
// ACORN-γ builder densifies level-0 to (M_beta + 1.5*M) and upper levels to
// (M*gamma), matching the ACORN reference.
struct AcornGammaBuildParameters {
    bool enabled = false; // false => standard HNSW build
    int gamma = 1;
    int M = 32;
    int max_degree = 0;   // 0 => derive from gamma/M (level-0 = M_beta+1.5M)
    int M_beta = -1;      // -1 => default 2*M (ACORN reference default)
    int efConstruction = 0; // 0 => M*gamma (ACORN) or 40 (standard)
    int seed = 12345;
};

// Per-query search statistics (Phase measurement requirements).
struct SearchStats {
    uint64_t dist_computations = 0;   // distance evals
    uint64_t nodes_visited = 0;       // graph nodes touched (visited-set sets)
    uint64_t first_hop_expansions = 0;// direct neighbours examined
    uint64_t second_hop_expansions = 0;// two-hop (bridge) neighbours examined
    uint64_t selector_checks = 0;     // is_member() calls
    uint64_t accepted = 0;            // predicate-passing nodes evaluated
    uint64_t rejected = 0;            // predicate-failing selector checks
    uint64_t hops = 0;                // beam steps (nstep)
    uint64_t visit_limit_terminations = 0;
    uint64_t results_returned = 0;
    uint64_t bridges_used = 0;         // RACORN: filter-failing bridge nodes admitted
    uint64_t aef_switches = 0;         // RACORN-1+: exact-fallback activations (0/1 per query)

    void add(const SearchStats& o) {
        dist_computations += o.dist_computations;
        nodes_visited += o.nodes_visited;
        first_hop_expansions += o.first_hop_expansions;
        second_hop_expansions += o.second_hop_expansions;
        selector_checks += o.selector_checks;
        accepted += o.accepted;
        rejected += o.rejected;
        hops += o.hops;
        visit_limit_terminations += o.visit_limit_terminations;
        results_returned += o.results_returned;
        bridges_used += o.bridges_used;
        aef_switches += o.aef_switches;
    }
};

// RACORN-1 / RACORN-1+ search parameters (arXiv:2607.00768). Search-time only;
// runs over a STANDARD HNSW graph (no build change), like ACORN-1.
struct RacornParams {
    double bridge_ratio = 1.0;   // ASF bridge allocation per unvisited 1-hop (0 => ACORN-1)
    bool stride_sampling = true; // diversity-preserving selection vs prefix truncation
    bool enable_aef = false;     // RACORN-1+ : Adaptive Exact Fallback
    double aef_threshold = 0.003;// switch to exact when running pass-ratio < this
    int aef_min_probe = 0;       // min visited before AEF may trigger (0 => derive from ef)
};

// Index-build statistics.
struct BuildStats {
    double build_ms = 0;
    uint64_t candidate_edges_considered = 0;
    uint64_t edges_retained = 0;
    uint64_t edges_pruned = 0;
    size_t graph_bytes = 0;           // neighbors[] + offsets[] bytes
    double avg_degree_l0 = 0;
    int max_degree_l0 = 0;
    int levels = 0;
};

// Bundles a flat vector storage with a faiss::HNSW link structure.
struct AcornIndex {
    int d = 0;
    faiss::MetricType metric = faiss::METRIC_L2;
    bool is_acorn_gamma = false;
    int gamma = 1;
    int M = 32;
    int M_beta = 64;
    std::unique_ptr<faiss::IndexFlat> storage; // owns the raw float vectors
    faiss::HNSW hnsw{32};
    BuildStats build_stats;

    faiss::idx_t ntotal() const { return storage ? storage->ntotal : 0; }
};

/* ----------------------------- Phase 3: build ---------------------------- */

// Build a STANDARD faiss HNSW graph (faithful faiss::HNSW::add_with_locks path,
// single-threaded/deterministic). Used for baselines A, B, D, E.
void build_standard_hnsw(
        AcornIndex& index,
        int d,
        faiss::MetricType metric,
        int M,
        int efConstruction,
        faiss::idx_t n,
        const float* x,
        int seed = 12345);

// Build an ACORN-γ graph (denser level-0, two-hop compression pruning). Used
// for baseline C. gamma==1 reproduces the ACORN-1 graph density.
void build_acorn_gamma(
        AcornIndex& index,
        int d,
        faiss::MetricType metric,
        const AcornGammaBuildParameters& params,
        faiss::idx_t n,
        const float* x);

/* --------------------------- Phase 2: search ----------------------------- */

// Unified filtered top-k search entry. mode selects stock faiss traversal vs
// ACORN traversal. `sel` is the filter predicate (nullptr => unfiltered).
// Returns number of results written (<= k). out_ids/out_dis must hold k slots.
int filtered_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        FilteredHnswSearchMode mode,
        const faiss::IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats = nullptr);

/* ------------------- RACORN-1 / RACORN-1+ (Phase: RACORN) ---------------- */

// RACORN traversal over a standard HNSW graph. Faithful to arXiv:2607.00768:
// ACORN-SEARCH-LAYER beam + RACORN1-EXPAND (Adaptive Search Fallback with
// filter-failing bridge nodes + stride sampling); optional Adaptive Exact
// Fallback (RACORN-1+). Bridges enter the candidate queue only, never results.
int racorn_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        const faiss::IDSelector* sel,
        const RacornParams& params,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats = nullptr);

// Faithful HNSW In-filtering (distance-first) baseline, per arXiv:2607.00768 §2.1:
// predicate-agnostic traversal, predicate applied only to the result heap, search
// continues until ef PASSING results are collected. Recall-preserving but latency
// grows ~ef/selectivity. This is the paper's "HNSW" reference (contrast with the
// faiss native early-termination path exposed via FilteredHnswSearchMode::STANDARD).
int hnsw_infilter_search(
        const AcornIndex& index,
        const float* query,
        int k,
        int efSearch,
        const faiss::IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats = nullptr);

/* ----------------------- Exact filtered oracle (F) ----------------------- */

int exact_filtered_search(
        const AcornIndex& index,
        const float* query,
        int k,
        const faiss::IDSelector* sel,
        faiss::idx_t* out_ids,
        float* out_dis,
        SearchStats* stats = nullptr);

/* ------------------------- Prototype serialization ----------------------- */
// Compact, PROTOTYPE-ONLY graph format with an explicit marker/version. Fails
// loudly on mismatch. Not production/faiss-compatible (see design doc).
void save_acorn_index(const AcornIndex& index, const std::string& path);
void load_acorn_index(AcornIndex& index, const std::string& path);

// Degree helpers for correctness tests.
int graph_max_degree(const faiss::HNSW& hnsw, int level);
double graph_avg_degree(const faiss::HNSW& hnsw, int level);

} // namespace acorn
