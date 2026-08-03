# Session handoff — ACORN / RACORN filtered-kNN POC (resume-safe)

## Git state
- Working branch: `claude/acorn-gamma-faiss-hnsw-729026` (all committed at `be4fb55`).
- Fresh clean baseline branch: `acorn-baseline-clean` (= `main`, no changes).
- Submodules `jni/external/{faiss,nmslib}`: hard-reset to pinned commits + `git config
  submodule.<path>.ignore all` so they never pollute `git status`.
- Production changes = 13 files (+927/−2) under `src/` + `jni/` (the `filtered_search_mode`
  POC). `research/` = benchmark tooling (separate). `bench_param.cpp` may be untracked.

## CRITICAL correction (do not repeat the old claims)
- Earlier "OpenSearch is slow / ~100k visits / 37ms / recall 0" was a **harness bug**:
  `research/acorn/src/acorn_hnsw.cpp: standard_search` didn't set `SearchParametersHNSW.
  bounded_queue`, so faiss ran its **unbounded** path (ignores the selector, scans all).
  **Fixed** (bounded_queue=true). RETRACTED the "slow" framing + `proof_slow` "today" curve.
- Corrected faiss-native (today): **fast, ~0.1–0.2ms, ~1,800 visits**, recall perfect under
  positive/no correlation; **recall≈0 only under negative (cross-domain) correlation** because
  the bounded beam can't reach far-away eligible nodes. Real OpenSearch then hits its post-ANN
  **exact fallback** → correct results at exact's O(cand) cost.
- Corrected numbers: `FINAL_SELECTIVITY_TABLE.md`. Docs still carrying WRONG "today is slow"
  figures and needing correction: `POC_SUMMARY.md`, `OPENSEARCH_ACORN_RESULTS.md`,
  `PROOF_OPENSEARCH_KNN_SLOW.md`.

## Honest thesis now
Filtered graph-ANN gives no benefit for negative-correlation filters (walk goes toward query,
eligible docs are elsewhere) → OpenSearch falls back to exact, whose cost grows with the
eligible set + index. Best fix = smarter exact routing (pure Java, `KNNWeight`/`ExactSearcher`).
RACORN keeps graph-ANN bounded there but only partially recovers recall (0.87–0.90 at low sel,
degrades to 0.08–0.21 at 25–50% neg) and its wall-clock needs an optimized traversal.

## Approach (agreed): standalone Faiss-subset harness, NOT a live cluster
The harness's `standard` mode == vanilla faiss `IndexHNSW::search`+`IDSelector` = "fresh
OpenSearch". Faithful for recall / distance-comps / native latency. Java-layer overheads
(routing, exact-fallback, segments) modeled/noted, not live-measured.

## Gated plan (get user opinion after EACH step)
- Step 2 (IN PROGRESS): fresh/standard baseline, all query knobs as ARGS, sweep
  ef_search × selectivity 0.1–50% × {neg,no,pos} at 100k + 1M.
- Step 3: ACORN (mode=acorn; args gamma, m_beta). Show better at higher sel / correlated, fails low sel.
- Step 4: RACORN (mode=racorn; args bridge_ratio, stride). Show recall/latency.
- Step 5: RACORN-1+ (mode=racorn_plus; arg aef_thr).
Keep ALL tuning params as query-analog args (never hard-code) so we can iterate.

## Run
```
bash research/acorn/scripts/build.sh bench_param       # (rebuilds faiss subset if needed)
./research/acorn/build/bench_param <n> <mode> [ef_list] [gamma] [m_beta] [bridge_ratio] [aef_thr]
#   mode: standard|acorn|racorn|racorn_plus ; writes raw-results/param_<mode>_n<n>.csv
```
Faithful native integration: `jni/src/acorn_hnsw.cpp` (ACORN + racorn), dispatched by
`faiss_wrapper.cpp` on `method_parameters.filtered_search_mode`. Compiles vs Faiss 1.11.0.
33/33 native tests + 13 Java tests pass.
```
```
