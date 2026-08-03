# Customer-Value Analysis — where ACORN / RACORN help, and where they must not be enabled

The strongest argument for these modes is not that they win everywhere. It is
that OpenSearch can offer a **targeted, explicit** execution mode for workloads
where **filter metadata aligns (or anti-aligns) with vector-space structure** and
the standard filtered path performs poorly. The governing variables are
**query–filter correlation** and **absolute eligible-candidate count** — not
selectivity percentage alone.

## The map (from the collective benchmark)

| regime | what the standard/faiss path does | best mode | why |
|---|---|---|---|
| eligible count ≲ 1–2K (any correlation) | exact gate already handles it | **exact** (existing) | O(cand·d) is cheap and exact |
| positive correlation (valid near query) | works fine | **standard** | valid nodes are on the query's greedy path |
| negative / cross-domain correlation, low selectivity, large N | **collapses to recall ~0** (faiss native) or **explodes 20–40ms** (HNSW in-filter) | **RACORN-1 / RACORN-1+** | filter-failing bridges reconnect the severed predicate subgraph; AEF switches to exact when even bridges are starved |
| random / weakly-correlated filters | standard/HNSW fine; ACORN-family can hurt or waste work | **standard** (or exact if small) | predicate-subgraph restriction gives no benefit and adds overhead |

## Scenarios likely to benefit (correlation present)

### Product search — semantic query + category + availability
- **Why correlation:** a category ("running shoes") is a coherent embedding region;
  an availability flag is often *anti*-correlated with a query for discontinued-but-
  semantically-near items (negative correlation).
- **Expected candidate count:** category filters are frequently 0.5–5% of the corpus
  → thousands of candidates at ≥1M scale (above the exact crossover).
- **Expected benefit:** high when the eligible category is a *different* region than
  the query (negative correlation) — the standard path collapses/explodes; RACORN-1
  recovers recall at a fraction of HNSW's visits.
- **Alternative:** if the category is small (≲1–2K docs), exact is better (and OpenSearch
  already routes there).

### Multi-domain / multi-language document search
- **Why correlation:** language/domain strongly shapes embedding locality; a query in
  one language filtered to another domain is a negative-correlation case.
- **Expected candidate count:** domain filters are broad (5–30%) → large eligible sets
  at scale; **RACORN-1** (bridges) is the fit, RACORN-1+ only at the extreme tail.

### Clustered tenant search
- **Why correlation:** a tenant whose content is a distinct semantic domain makes the
  tenant filter a cluster filter. Querying with a cross-tenant vector is negative
  correlation. **RACORN-1** applies.
- **Contrast:** hash/random tenant assignment (below) is a *losing* case.

### Region-aware recommendations
- **Why correlation:** region eligibility can correlate with content/preferences.
  Positive correlation → standard is fine; negative (recommend from an
  under-served region) → RACORN-1.

## Required losing scenarios (where ACORN/RACORN must NOT be enabled)

| losing scenario | measured behaviour | recommendation |
|---|---|---|
| random ACL filters | no correlation; predicate-subgraph restriction adds cost without recall benefit | **standard** |
| random / hash tenant assignment | same as random | **standard** |
| broad filters (≥30–50%) | plenty of eligible near the query; standard already fast/accurate | **standard** |
| extremely small candidate sets (≲1–2K) | exact is fastest and exact | **exact** (existing gate) |
| weakly-correlated metadata | marginal/no benefit, added overhead | **standard** |
| CPU-saturated workloads | RACORN's per-node overhead (2-hop scan) worsens tail under saturation | **standard**, or exact if selective |

## The explicit, no-auto-selection stance

Because the right mode depends on correlation (which OpenSearch cannot cheaply know
a priori) and on candidate count, this POC deliberately exposes the mode as an
**explicit, opt-in** per-query parameter with **no automatic selection**. The
customer (who knows their filter semantics) chooses `racorn` for known
negative-correlation workloads; everyone else keeps `standard`. Automatic
selection is left to a future RFC only if a cheap correlation signal proves robust.
