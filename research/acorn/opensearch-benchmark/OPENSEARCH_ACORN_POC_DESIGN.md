# OpenSearch ACORN / RACORN POC — Design

OpenSearch-integrated proof-of-concept for **experimental filtered-HNSW traversal
policies** on the native Faiss engine, selected explicitly per query. Adds
**ACORN**, **RACORN-1**, and **RACORN-1+** as opt-in modes alongside the existing
`standard` traversal; determines (via the collective benchmark) whether any is
strong enough for an experimental RFC.

Scope (this POC): Faiss engine, HNSW, `IndexHNSWFlat`, float vectors, filtered
top-k, non-nested fields, CPU, L2 + inner-product, single-node, multi-segment.
Deferred: memory-optimized search, Lucene/NMSLIB, IVF, SQ/PQ, byte/binary,
nested, radial, GPU/CAGRA, automatic strategy selection.

## 1. End-to-end filtered-search flow (with the new hook)

```
REST k-NN query (method_parameters.filtered_search_mode = standard|acorn|racorn|racorn_plus)
  → KNNQueryBuilder.doToQuery  — validates the mode against engine/method/filter/query-type
  → KNNWeight.searchLeaf       — builds the filter BitSet, exact/ANN gate (unchanged)
  → DefaultKNNWeight.doANNSearch → FilterIdsSelector (BITMAP/BATCH)  (unchanged)
  → JNIService.queryIndex → FaissService.queryIndexWithFilter        (mode rides method_parameters)
  → QueryIndex_WithFilter (faiss_wrapper.cpp)
        IDSelectorJlongBitmap / IDSelectorBatch  (existing, reused)
        read filtered_search_mode → dispatch:
          standard    → indexReader->search(... SearchParametersHNSW.sel ...)   [unchanged Faiss path]
          acorn       → knn_jni::acorn::search(...)                              [Faiss-layer policy]
          racorn      → knn_jni::acorn::racorn(... bridge_ratio, aef=false ...)  [Faiss-layer policy]
          racorn_plus → knn_jni::acorn::racorn(... aef=true ...)                 [Faiss-layer policy]
  → segment result collection → OpenSearch top-k response
```

The mode travels in the existing `method_parameters` map (the same channel as
`ef_search`), so **no new JNI method signature** is required. Omitting the
parameter preserves current behaviour exactly.

## 2. Experimental API

```json
{ "knn": { "field": "v", "vector": [...], "k": 10,
           "filter": { "term": { "category": "shoes" } },
           "method_parameters": { "filtered_search_mode": "racorn" } } }
```

Supported values: `standard` (= omitted), `acorn`, `racorn`, `racorn_plus`.
Rules (all enforced, **no silent fallback**):

| rule | where |
|---|---|
| unknown value → fail (parse) | `MethodParameter.FILTERED_SEARCH_MODE.parse` / `FilteredSearchMode.fromWireName` |
| non-standard mode requires a filter | `FilteredSearchMode.validateForQuery` (in `doToQuery`) |
| only FAISS engine | same |
| only HNSW method | same |
| not for radial (min_score/max_distance) | same |
| omitted → existing behaviour | `fromMethodParameters` → `STANDARD` |

## 3. Files changed

**Production Java** (minimal, feature-flag-style, default-off):
- `src/main/java/org/opensearch/knn/common/KNNConstants.java` — mode name + values.
- `src/main/java/org/opensearch/knn/index/query/FilteredSearchMode.java` — **new**: enum
  `{STANDARD, ACORN, RACORN, RACORN_PLUS}` + `validateForQuery`.
- `src/main/java/org/opensearch/knn/index/query/request/MethodParameter.java` — register
  `filtered_search_mode` (parse/validate, unknown fails).
- `src/main/java/org/opensearch/knn/index/query/KNNQueryBuilder.java` — `FILTERED_SEARCH_MODE_FIELD`
  + `validateForQuery(...)` call in `doToQuery`.
- `src/test/java/org/opensearch/knn/index/query/FilteredSearchModeTests.java` — **new**: 13 tests
  (parse, validation rules, all modes). **Pass.**

**Native (JNI / Faiss layer)** — builds in a full JNI environment; the traversal
modules compile cleanly against the bundled Faiss 1.11.0 here:
- `jni/include/acorn_hnsw.h`, `jni/src/acorn_hnsw.cpp` — **new**: ACORN traversal +
  RACORN-1/1+ traversal (`knn_jni::acorn::search` / `::racorn`) over a
  faiss::IndexIDMap-wrapped IndexHNSW, reusing `IDSelector`; env-gated benchmark
  instrumentation.
- `jni/src/faiss_wrapper.cpp` — read `filtered_search_mode`, dispatch to the policy.
- `jni/src/commons.cpp`, `jni/include/commons.h` — `getStringMethodParameter`.
- `jni/src/jni_util.cpp`, `jni/include/jni_util.h` — mode constants.
- `jni/CMakeLists.txt` — add `acorn_hnsw.cpp` to the faiss JNI lib.

**Research** (out of production source): `research/acorn/src/*` (validated standalone
implementations — `racorn.cpp` is the reference RACORN-1/1+; `acorn_hnsw.cpp` +
`acorn_gamma_builder.cpp` for ACORN / ACORN-γ), `research/acorn/opensearch-benchmark/*`.

**Faiss submodule: unchanged** (no new patch). The traversal policies are separate
translation units in the OpenSearch native layer, not edits to Faiss internals.

## 4. Architecture conformance

- **Java**: parses + validates the experimental parameter, threads the stable
  wire value through the existing `method_parameters` structure, preserves
  behaviour when omitted, adds parsing/validation tests. ✔
- **JNI**: translates the mode to a Faiss-layer policy call, reuses the existing
  `IDSelector`, attaches it as today via `SearchParametersHNSW.sel`; **no graph
  traversal is duplicated in JNI glue** — it lives in `acorn_hnsw.cpp`. ✔
- **Faiss layer**: standard HNSW traversal unchanged; ACORN/RACORN are **separate
  filtered-traversal policies** using `IDSelector::is_member`; only
  selector-approved vectors enter the result heap; rejected vectors act as routing
  bridges (RACORN ASF); first/second-hop candidates deduplicated; `efSearch` and
  termination respected; read-only over a `const` graph (thread-safe); no graph
  mutation during search. ✔

## 5. Validation & compatibility matrix

| condition | result |
|---|---|
| mode omitted | existing standard behaviour (byte-identical path) |
| `standard` | forces existing Faiss traversal |
| `acorn` / `racorn` / `racorn_plus` | selector-aware policy |
| unknown value (`"hybrid"`, `"ACORN"`) | rejected at REST parse |
| non-standard mode, no filter | rejected in `doToQuery` |
| non-FAISS engine (Lucene/NMSLIB) | rejected |
| non-HNSW method (IVF) | rejected |
| radial (min_score/max_distance) | rejected |
| nested / parent grouping | not supported under these modes (rejected) — POC scope |
| node-to-node serialization | rides `MethodParametersParser` stream in/out (version-gated) |

## 6. Instrumentation (Phase 2)

`AcornSearchStats` (thread-local, populated only when env `KNN_ACORN_BENCH_STATS=1`;
negligible/zero overhead otherwise): search mode, segment vector count, nodes
visited, first/second-hop neighbours, selector checks/matches/rejections, distance
computations, eligible discovered, bridges used, AEF switches, results returned.
Native traversal time and OpenSearch request time are captured by the harness, not
per-query INFO logs.

## 7. ACORN-γ note

ACORN-γ requires a **build-time** graph change (denser neighbourhoods). It has no
production OpenSearch construction path in this POC; it is evaluated as a
**test-only artifact** (built in the benchmark harness). RACORN-1/1+ and ACORN are
search-time only over the standard OpenSearch-built graph and therefore integrate
cleanly with no index change.
