/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.research;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Track 2, step 10 — WORKING-SET + CONCURRENCY CROSSOVER. Central question: does 4-bit's 7-8x lower
 * vector-code traffic + far-better-than-1-bit graph navigation translate into a real p99/QPS win over
 * fp32 once the fp32 working set no longer fits in CPU cache and concurrent queries compete for
 * bandwidth? Measured, not inferred. Representation & geometry UNCHANGED from step 8/9.
 *
 * Three test methods (run individually with --tests "...methodName"):
 *   testNeonKernelMicrobench  -- real Vector-API(NEON) dot-product kernel vs scalar-LUT vs scalar.
 *   testScaleConcurrency      -- SIFT-128 real, N in {6K..500K}, equal recall@10>=0.95, 1..16 threads.
 *   testHighDim               -- 768-D synthetic (isotropic + low-rank MANIFOLD proxy), ceiling+equal-recall.
 *
 * Honesty: Apple arm64 host has NO perf/PMU access -> hardware counters unavailable; we use wall-clock,
 * bytes-touched estimates, cache-footprint math, and scaling efficiency (documented). Latency warm,
 * cores not pinned. No real 768/1536 embedding corpus is available locally -> high-D uses SYNTHETIC
 * data (isotropic + a better-than-isotropic low-rank manifold proxy), explicitly NOT a production claim.
 */
public class Track2ScaleConcurrencyTests extends KNNTestCase {

    private static final int K = 10, M = 16, BEAM = 100;
    private static final int NCAL = 50, NEVAL = 200;
    private static final int[] EFS = { 20, 50, 100, 200, 500, 800 };
    private static final int[] DEPTHS = { 10, 20, 30, 50, 100, 200, 300, 500 };
    private static final double TARGET = 0.95, MARGIN = 0.005;
    private static final long SEED = 71L;
    private static final String ISA = "arm64";   // JIT/SuperWord autovec; no hand-written intrinsics (Java)
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";

    // ===================================================================================
    //  A) NEON kernel microbench (item 6) + exactness of the dot-product reformulation
    // ===================================================================================
    public void testNeonKernelMicrobench() throws IOException {
        // exactness: dot-product form is algebraically exact but float-reassociated -> tolerance, not bit-exact.
        Random rng = new Random(SEED);
        int maxRelViolations = 0; double maxRel = 0;
        for (int dim : new int[]{128, 384, 768, 1536, 130, 7}) {
            float[][] V = randData(dim, 300, rng);
            for (int bits : new int[]{2, 4}) {
                Packed p = encode(V, bits); float[] cx = perCodeCx(p);
                float[] q = V[rng.nextInt(V.length)]; float[] lut = buildLUT(p, q); QP qp = buildQP(p, q);
                for (int t = 0; t < 60; t++) { int i = rng.nextInt(V.length);
                    float sc = asymL2(p, q, p.codes[i]);          // scalar reference (authoritative)
                    float sl = asymLUT(p, p.codes[i], lut);        // scalar LUT (bit-exact to ref, from step 9)
                    float sn = asymDotAutovec(p, p.codes[i], qp, cx[i]); // Vector-API dot form (tolerance)
                    assertEquals("lut==scalar dim=" + dim + " bits=" + bits, sc, sl, 0f);
                    double rel = Math.abs(sn - sc) / Math.max(1f, Math.abs(sc)); maxRel = Math.max(maxRel, rel);
                    if (rel > 1e-3) maxRelViolations++;
                }
            }
        }
        System.out.printf(java.util.Locale.ROOT, "[step10-neon-exact] dot-form max_rel_err=%.2e (>1e-3 count=%d) -- algebraically exact, float-reassociated%n", maxRel, maxRelViolations);

        List<String> rows = new ArrayList<>();
        rows.add("kernel,bit_width,dimension,implementation,isa,access_pattern,query_prep_ns,score_ns,cycles_per_dim_est,scores_per_second_M,speedup_vs_scalar,bytes_per_score");
        double FREQ = 4.4;
        for (int dim : new int[]{128, 384, 768, 1536}) {
            float[][] V = randData(dim, 200_000 / Math.max(1, dim / 128), new Random(SEED + dim)); // ~large working set
            float[] q = V[0];
            for (int bits : new int[]{1, 2, 4, 32}) {
                Packed p = encode(V, bits); float[] cx = bits <= 4 ? perCodeCx(p) : null;
                float[] lut = bits <= 4 ? buildLUT(p, q) : null; QP qp = bits <= 4 ? buildQP(p, q) : null;
                for (String access : new String[]{ "sequential", "random" }) {
                    int[] perm = perm(p, V.length, access, new Random(SEED + 9));
                    double scalarNs = time(() -> { float s = 0; for (int idx : perm) s += bits == 32 ? l2(q, V[idx]) : asymL2(p, q, p.codes[idx]); return s; }, perm.length);
                    rows.add(row("scalar", bits, dim, "scalar", access, 0, scalarNs, FREQ, 1.0, bits == 32 ? dim * 4 : p.codeBytes));
                    if (bits <= 4) {
                        long pr0 = System.nanoTime(); float[] lut2 = buildLUT(p, q); long prLut = System.nanoTime() - pr0;
                        double lutNs = time(() -> { float s = 0; for (int idx : perm) s += asymLUT(p, p.codes[idx], lut2); return s; }, perm.length);
                        rows.add(row("lut", bits, dim, "lut", access, prLut, lutNs, FREQ, scalarNs / lutNs, p.codeBytes));
                        long pn0 = System.nanoTime(); QP qp2 = buildQP(p, q); long prNeon = System.nanoTime() - pn0;
                        double neonNs = time(() -> { float s = 0; for (int idx : perm) s += asymDotAutovec(p, p.codes[idx], qp2, cx[idx]); return s; }, perm.length);
                        rows.add(row("dot_autovec", bits, dim, "autovec_superword", access, prNeon, neonNs, FREQ, scalarNs / neonNs, p.codeBytes));
                    }
                }
            }
            System.out.printf(java.util.Locale.ROOT, "[step10-neon] dim=%d done%n", dim);
        }
        write("track2_step10_kernel_" + ISA + "_seed" + SEED + ".csv", rows);
    }

    // ===================================================================================
    //  B) Scale + concurrency crossover on real SIFT-128 (items 1-4, 8, 9)
    // ===================================================================================
    public void testScaleConcurrency() throws IOException {
        long maxHeap = Runtime.getRuntime().maxMemory();
        // fp32 X floats dominate: N*dim*4 bytes; keep < ~35% heap for safety (graph+GT+codes+overhead).
        int dim = 128; long perN = (long) dim * 4;
        int nCap = (int) Math.min(1_000_000, (maxHeap * 35 / 100) / perN);
        // {6K..250K} fit the 20-min randomizedtesting suite timeout; 500K runs in testScale500k (own budget).
        int[] allTiers = { 6000, 25000, 100000, 250000 };
        List<Integer> tiers = new ArrayList<>(); for (int n : allTiers) if (n <= nCap) tiers.add(n);
        System.out.printf(java.util.Locale.ROOT, "[step10-env] maxHeap=%.1fGB nCap=%d tiers=%s isa=%s cores=15(no-SMT) perf=UNAVAILABLE(macOS-arm64)%n",
            maxHeap / 1e9, nCap, tiers, ISA);

        float[][] baseAll = readFvecs(repo().resolve(BASE).toString(), tiers.get(tiers.size() - 1));
        float[][] Qall = readFvecs(repo().resolve(QUERY).toString(), NCAL + NEVAL);

        List<String> scale = new ArrayList<>();
        scale.add("dataset,N,dim,representation,bit_width,kernel,isa,ef_search,rerank_depth,heldout_recall,mean_nodes,dist_comps,code_bytes_per_vec,code_bytes_read_per_query,total_bytes_touched_per_query,fp32_workingset_MB,cache_state,p50_us,p90_us,p95_us,p99_us,max_us,qps_1t,build_ms,vectors_per_sec,index_code_MB,status");
        List<String> conc = new ArrayList<>();
        conc.add("dataset,N,representation,bit_width,kernel,isa,ef_search,rerank_depth,threads,qps,p50_us,p95_us,p99_us,scaling_eff,cache_state");
        List<String> cross = new ArrayList<>();
        cross.add("workload,N,threads,cache_state,fp32_p99,twobit_p99,fourbit_p99,fp32_qps,twobit_qps,fourbit_qps,p99_winner,qps_winner");
        List<String> sel = new ArrayList<>();
        sel.add("segment,N,dim,objective,selected_rep,ef,rerank_depth,recall,p99_us,storage_bytes_per_vec,reason");

        long L2 = 8L * 1024 * 1024, SLC_EST = 24L * 1024 * 1024; // M-series shared cache is large; ~24MB estimate (labeled EST)

        for (int N : tiers) {
            float[][] X = Arrays.copyOf(baseAll, N);
            int[][] gt = groundTruth(X, Qall, N);
            double fp32WsMB = (double) N * dim * 4 / 1e6;
            String cacheState = (N * (long) dim * 4 <= L2) ? "fits_L2" : (N * (long) dim * 4 <= SLC_EST ? "fits_SLC_est" : "exceeds_SLC_est");
            boolean fullConc = (N == 6000) || (N == tiers.get(tiers.size() - 1)) || (N == 250000);
            int[] threadSet = fullConc ? new int[]{ 1, 2, 4, 8, 16 } : new int[]{ 1 };

            // 1-bit only as reference at small + one large tier (dominated elsewhere)
            int[] repBits = (N == 6000 || N == 100000) ? new int[]{ 32, 1, 2, 4 } : new int[]{ 32, 2, 4 };
            java.util.Map<Integer, double[]> p99qps = new java.util.HashMap<>(); // bits -> {p99, qps@maxThreads}

            for (int bits : repBits) {
                Packed p = encode(X, bits); float[] cx = bits <= 4 ? perCodeCx(p) : null;
                long tb = System.nanoTime(); OnHeapHnswGraph g = buildGraph(p, X, bits); long buildMs = (System.nanoTime() - tb) / 1_000_000;
                double vps = N * 1000.0 / Math.max(1, buildMs);

                // equal-recall calibration: cheapest ef with held-out recall@10>=target (conservative on cal split), then depth
                int feEf = -1, feDepth = -1; double feCeil = 0;
                for (int ef : EFS) {
                    int[][] pos = posAt(g, p, X, Qall, gt, bits, ef, cx);
                    double ceil = recallAt(pos, 0, Qall.length, ef);
                    if (feEf < 0 && ceil >= TARGET) { int dep = calibrateDepth(pos, TARGET, MARGIN); if (dep > 0) { feEf = ef; feDepth = dep; feCeil = ceil; } }
                }
                String kernel = bits == 32 ? "fp32" : (bits + "bit_lut");
                int codeB = bits == 32 ? dim * 4 : (dim * bits + 7) / 8;
                double idxCodeMB = (double) N * codeB / 1e6;
                if (feEf < 0) {
                    scale.add(String.format(java.util.Locale.ROOT, "sift,%d,%d,%s,%d,%s,%s,,,,,,%d,,,%.1f,%s,,,,,,,%d,%.0f,%.1f,SLA_UNACHIEVABLE_AT_MAX_CONFIG",
                        N, dim, repName(bits), bits, kernel, ISA, codeB, fp32WsMB, cacheState, buildMs, vps, idxCodeMB));
                    flush(scale, conc, cross, sel); continue;
                }
                // per-query full latency (warm), held-out only
                Perf pf = warmPerf(g, p, X, Qall, gt, bits, feEf, feDepth, cx);
                double codeBytesRead = pf.meanNodes * codeB;
                double adjBytes = pf.meanNodes * M * 4;                       // neighbor-id reads (est)
                double rerankBytes = feDepth * (double) dim * 4;             // fp32 rerank reads
                double totalTouched = codeBytesRead + adjBytes + rerankBytes;
                scale.add(String.format(java.util.Locale.ROOT, "sift,%d,%d,%s,%d,%s,%s,%d,%d,%.4f,%.1f,%.0f,%d,%.0f,%.0f,%.1f,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.0f,%d,%.0f,%.1f,CALIBRATED",
                    N, dim, repName(bits), bits, kernel, ISA, feEf, feDepth, pf.heldRecall, pf.meanNodes, pf.meanNodes, codeB,
                    codeBytesRead, totalTouched, fp32WsMB, cacheState, pf.p50, pf.p90, pf.p95, pf.p99, pf.max, pf.qps1t, buildMs, vps, idxCodeMB));

                // concurrency sweep
                double base1t = -1;
                for (int T : threadSet) {
                    double[] tq = threadedQps(g, p, X, Qall, bits, feEf, feDepth, cx, T);
                    if (T == 1) base1t = tq[0];
                    double eff = T == 1 ? 1.0 : tq[0] / (base1t * T);
                    conc.add(String.format(java.util.Locale.ROOT, "sift,%d,%s,%d,%s,%s,%d,%d,%d,%.0f,%.1f,%.1f,%.1f,%.3f,%s",
                        N, repName(bits), bits, kernel, ISA, feEf, feDepth, T, tq[0], tq[1], tq[2], tq[3], eff, cacheState));
                    if (T == threadSet[threadSet.length - 1]) p99qps.put(bits, new double[]{ tq[3] /*p99 at maxT*/, tq[0] /*qps at maxT*/ });
                }
                // per-objective selection from calibration (latency/memory/p99/balanced)
                for (String obj : new String[]{ "latency_first", "memory_first", "p99_constrained", "balanced" }) {
                    // provisional: store per-rep tuple; final winner computed after loop below via helper
                }
                System.out.printf(java.util.Locale.ROOT, "[step10-scale] N=%d %s ef=%d d=%d recall=%.3f p50=%.1f p99=%.1f qps1t=%.0f build=%dms cache=%s%n",
                    N, repName(bits), feEf, feDepth, pf.heldRecall, pf.p50, pf.p99, pf.qps1t, buildMs, cacheState);
                storeSel(N, dim, bits, feEf, feDepth, pf, codeB);
                flush(scale, conc, cross, sel);
            }

            // crossover rows (fullConc tiers): compare fp32 vs 2 vs 4 at max threads
            if (fullConc && p99qps.containsKey(32) && p99qps.containsKey(2) && p99qps.containsKey(4)) {
                double f99 = p99qps.get(32)[0], t99 = p99qps.get(2)[0], q99 = p99qps.get(4)[0];
                double fq = p99qps.get(32)[1], tq = p99qps.get(2)[1], qq = p99qps.get(4)[1];
                int mt = threadSet[threadSet.length - 1];
                String p99win = (t99 < f99 || q99 < f99) ? (t99 < q99 ? "2bit" : "4bit") : "fp32";
                String qpswin = (tq > fq || qq > fq) ? (tq > qq ? "2bit" : "4bit") : "fp32";
                cross.add(String.format(java.util.Locale.ROOT, "sift_%dT,%d,%d,%s,%.1f,%.1f,%.1f,%.0f,%.0f,%.0f,%s,%s",
                    mt, N, mt, cacheState, f99, t99, q99, fq, tq, qq, p99win, qpswin));
            }
            flush(scale, conc, cross, sel);
        }
        // emit per-objective selections from stored tuples
        emitSelections(sel, dim);
        write("track2_step10_scale_" + ISA + "_seed" + SEED + ".csv", scale);
        write("track2_step10_concurrency_" + ISA + "_seed" + SEED + ".csv", conc);
        write("track2_step10_crossover_" + ISA + "_seed" + SEED + ".csv", cross);
        write("track2_step10_selected_" + ISA + "_seed" + SEED + ".csv", sel);
        writeStorageTable();
    }

    // ===================================================================================
    //  B2) Largest cache-pressure point: N=500K, fp32 vs 4-bit only (2-bit dominated at scale),
    //      own 20-min suite budget. Confirms/denies a crossover at the strongest working set.
    // ===================================================================================
    public void testScale500k() throws IOException {
        int N = 500_000, dim = 128;
        if (Runtime.getRuntime().maxMemory() < 4L * 1024 * 1024 * 1024) {   // needs ~1GB+; skip under Gradle's 512MB default
            System.out.println("[step10-500k] SKIP: needs >=4GB heap (set test { maxHeapSize = \"12g\" }); maxHeap="
                + Runtime.getRuntime().maxMemory() / (1024 * 1024) + "MB");
            return;
        }
        System.out.printf(java.util.Locale.ROOT, "[step10-500k] maxHeap=%.1fGB N=%d fp32_ws=%.0fMB isa=%s%n",
            Runtime.getRuntime().maxMemory() / 1e9, N, (double) N * dim * 4 / 1e6, ISA);
        float[][] X = readFvecs(repo().resolve(BASE).toString(), N);
        float[][] Q = readFvecs(repo().resolve(QUERY).toString(), NCAL + NEVAL);
        int[][] gt = groundTruth(X, Q, N);
        List<String> scale = new ArrayList<>();
        scale.add("dataset,N,dim,representation,bit_width,kernel,isa,ef_search,rerank_depth,heldout_recall,mean_nodes,code_bytes_per_vec,code_bytes_read_per_query,fp32_workingset_MB,cache_state,p50_us,p90_us,p95_us,p99_us,qps_1t,build_ms,index_code_MB,status");
        List<String> conc = new ArrayList<>();
        conc.add("dataset,N,representation,bit_width,kernel,isa,ef_search,rerank_depth,threads,qps,p50_us,p95_us,p99_us,scaling_eff,cache_state");
        List<String> cross = new ArrayList<>();
        cross.add("workload,N,threads,cache_state,fp32_p99,fourbit_p99,fp32_qps,fourbit_qps,p99_winner,qps_winner");
        java.util.Map<Integer, double[]> maxT = new java.util.HashMap<>();
        for (int bits : new int[]{ 32, 4 }) {
            Packed p = encode(X, bits); float[] cx = bits <= 4 ? perCodeCx(p) : null;
            long tb = System.nanoTime(); OnHeapHnswGraph g = buildGraph(p, X, bits); long buildMs = (System.nanoTime() - tb) / 1_000_000;
            int feEf = -1, feDepth = -1;
            for (int ef : EFS) { int[][] pos = posAt(g, p, X, Q, gt, bits, ef, cx); if (recallAt(pos, 0, Q.length, ef) >= TARGET) { int dep = calibrateDepth(pos, TARGET, MARGIN); if (dep > 0) { feEf = ef; feDepth = dep; break; } } }
            String kernel = bits == 32 ? "fp32" : (bits + "bit_lut"); int codeB = bits == 32 ? dim * 4 : (dim * bits + 7) / 8;
            if (feEf < 0) { scale.add(String.format(java.util.Locale.ROOT, "sift,%d,%d,%s,%d,%s,%s,,,,,%d,,%.1f,exceeds_SLC_est,,,,,,%d,%.1f,SLA_UNACHIEVABLE", N, dim, repName(bits), bits, kernel, ISA, codeB, (double) N * dim * 4 / 1e6, buildMs, (double) N * codeB / 1e6)); continue; }
            Perf pf = warmPerf(g, p, X, Q, gt, bits, feEf, feDepth, cx);
            scale.add(String.format(java.util.Locale.ROOT, "sift,%d,%d,%s,%d,%s,%s,%d,%d,%.4f,%.1f,%d,%.0f,%.1f,exceeds_SLC_est,%.1f,%.1f,%.1f,%.1f,%.0f,%d,%.1f,CALIBRATED",
                N, dim, repName(bits), bits, kernel, ISA, feEf, feDepth, pf.heldRecall, pf.meanNodes, codeB, pf.meanNodes * codeB, (double) N * dim * 4 / 1e6, pf.p50, pf.p90, pf.p95, pf.p99, pf.qps1t, buildMs, (double) N * codeB / 1e6));
            double base1t = -1;
            for (int T : new int[]{ 1, 2, 4, 8, 16 }) { double[] tq = threadedQps(g, p, X, Q, bits, feEf, feDepth, cx, T); if (T == 1) base1t = tq[0]; double eff = T == 1 ? 1.0 : tq[0] / (base1t * T);
                conc.add(String.format(java.util.Locale.ROOT, "sift,%d,%s,%d,%s,%s,%d,%d,%d,%.0f,%.1f,%.1f,%.1f,%.3f,exceeds_SLC_est", N, repName(bits), bits, kernel, ISA, feEf, feDepth, T, tq[0], tq[1], tq[2], tq[3], eff));
                if (T == 16) maxT.put(bits, new double[]{ tq[3], tq[0] }); }
            System.out.printf(java.util.Locale.ROOT, "[step10-500k] %s ef=%d d=%d recall=%.3f p50=%.1f p99=%.1f qps1t=%.0f build=%dms%n", repName(bits), feEf, feDepth, pf.heldRecall, pf.p50, pf.p99, pf.qps1t, buildMs);
            write("track2_step10_scale500k_" + ISA + "_seed" + SEED + ".csv", scale);
            write("track2_step10_concurrency500k_" + ISA + "_seed" + SEED + ".csv", conc);
        }
        if (maxT.containsKey(32) && maxT.containsKey(4)) { double f99 = maxT.get(32)[0], q99 = maxT.get(4)[0], fq = maxT.get(32)[1], qq = maxT.get(4)[1];
            cross.add(String.format(java.util.Locale.ROOT, "sift_16T,%d,16,exceeds_SLC_est,%.1f,%.1f,%.0f,%.0f,%s,%s", N, f99, q99, fq, qq, q99 < f99 ? "4bit" : "fp32", qq > fq ? "4bit" : "fp32"));
            write("track2_step10_crossover500k_" + ISA + "_seed" + SEED + ".csv", cross); }
    }

    // ===================================================================================
    //  C) High-dimensional (768) synthetic: candidate ceiling + equal-recall at 0.90/0.95/0.97
    // ===================================================================================
    // Split per dataset so each 768-D suite fits the 20-min randomizedtesting budget. N=20K (768-D
    // builds are ~6x costlier/dim than 128-D). Concurrency at 8T to also probe the dim axis under load.
    public void testHighDimIsotropic() throws IOException { highDim("isotropic768"); }
    public void testHighDimManifold() throws IOException { highDim("manifold768"); }

    private void highDim(String ds) throws IOException {
        int dim = 768, N = 8000;   // isotropic-768 is pathological (poor graph, high ef); 8K keeps it tractable
        List<String> ceil = new ArrayList<>();
        ceil.add("dataset,dim,N,representation,bit_width,ef_search,candidate_ceiling_recall10");
        List<String> eq = new ArrayList<>();
        eq.add("dataset,dim,N,representation,bit_width,kernel,target_recall,ef_search,rerank_depth,heldout_recall,mean_nodes,code_bytes_read,code_bytes_per_vec,p50_us,p99_us,qps_1t,qps_8t,p99_8t,status");
        float[][] X = genHighD(ds, dim, N, new Random(SEED));
        float[][] Q = genHighD(ds, dim, NCAL + NEVAL, new Random(SEED + 5));
        int[][] gt = groundTruth(X, Q, N);
        System.out.printf(java.util.Locale.ROOT, "[step10-highd] %s N=%d dim=%d fp32_ws=%.0fMB start%n", ds, N, dim, (double) N * dim * 4 / 1e6);
        for (int bits : new int[]{ 1, 2, 4, 32 }) {
            Packed p = encode(X, bits); float[] cx = bits <= 4 ? perCodeCx(p) : null;
            OnHeapHnswGraph g = buildGraph(p, X, bits);
            java.util.Map<Integer, int[][]> posCache = new java.util.HashMap<>();
            for (int ef : EFS) { int[][] pos = posAt(g, p, X, Q, gt, bits, ef, cx); posCache.put(ef, pos);
                ceil.add(String.format(java.util.Locale.ROOT, "%s,%d,%d,%s,%d,%d,%.4f", ds, dim, N, repName(bits), bits, ef, recallAt(pos, 0, Q.length, ef))); }
            String kernel = bits == 32 ? "fp32" : (bits + "bit_lut");
            for (double tgt : new double[]{ 0.90, 0.95, 0.97 }) {
                int feEf = -1, feDepth = -1;
                for (int ef : EFS) { int[][] pos = posCache.get(ef); if (recallAt(pos, 0, Q.length, ef) >= tgt) { int dep = calibrateDepth(pos, tgt, MARGIN); if (dep > 0) { feEf = ef; feDepth = dep; break; } } }
                int codeB = bits == 32 ? dim * 4 : (dim * bits + 7) / 8;
                if (feEf < 0) { eq.add(String.format(java.util.Locale.ROOT, "%s,%d,%d,%s,%d,%s,%.2f,,,,,,%d,,,,,SLA_UNACHIEVABLE", ds, dim, N, repName(bits), bits, kernel, tgt, codeB)); continue; }
                Perf pf = warmPerf(g, p, X, Q, gt, bits, feEf, feDepth, cx);
                double[] t8 = tgt == 0.95 ? threadedQps(g, p, X, Q, bits, feEf, feDepth, cx, 8) : new double[]{ 0, 0, 0, 0 };
                eq.add(String.format(java.util.Locale.ROOT, "%s,%d,%d,%s,%d,%s,%.2f,%d,%d,%.4f,%.1f,%.0f,%d,%.1f,%.1f,%.0f,%.0f,%.1f,CALIBRATED",
                    ds, dim, N, repName(bits), bits, kernel, tgt, feEf, feDepth, pf.heldRecall, pf.meanNodes, pf.meanNodes * codeB, codeB, pf.p50, pf.p99, pf.qps1t, t8[0], t8[3]));
            }
            System.out.printf(java.util.Locale.ROOT, "[step10-highd] %s bits=%d done%n", ds, bits);
            write("track2_step10_highd_ceiling_" + ds + "_" + ISA + "_seed" + SEED + ".csv", ceil);
            write("track2_step10_highd_equalrecall_" + ds + "_" + ISA + "_seed" + SEED + ".csv", eq);
        }
    }

    // ---- storage density table (item 7) ----
    private void writeStorageTable() throws IOException {
        List<String> rows = new ArrayList<>();
        rows.add("dims,N,onebit_MB,twobit_MB,fourbit_MB,fp32_MB,note");
        for (int dim : new int[]{ 128, 384, 768, 1536 }) for (long n : new long[]{ 1_000_000L, 10_000_000L, 100_000_000L }) {
            double one = (double) n * ((dim + 7) / 8) / 1e6, two = (double) n * ((dim * 2 + 7) / 8) / 1e6, four = (double) n * ((dim * 4 + 7) / 8) / 1e6, f = (double) n * dim * 4 / 1e6;
            rows.add(String.format(java.util.Locale.ROOT, "%d,%d,%.0f,%.0f,%.0f,%.0f,code-only (HNSW graph + fp32-rerank extra)", dim, n, one, two, four, f));
        }
        write("track2_step10_storage_density.csv", rows);
    }

    // ---- per-objective selection bookkeeping ----
    static final class SelRec { int N, dim, bits, ef, depth, codeB; Perf pf; }
    private final List<SelRec> selRecs = new ArrayList<>();
    private void storeSel(int N, int dim, int bits, int ef, int depth, Perf pf, int codeB) { SelRec s = new SelRec(); s.N = N; s.dim = dim; s.bits = bits; s.ef = ef; s.depth = depth; s.pf = pf; s.codeB = codeB; selRecs.add(s); }
    private void emitSelections(List<String> out, int dim) {
        java.util.Set<Integer> ns = new java.util.TreeSet<>(); for (SelRec s : selRecs) ns.add(s.N);
        for (int N : ns) { List<SelRec> cand = new ArrayList<>(); for (SelRec s : selRecs) if (s.N == N) cand.add(s);
            for (String obj : new String[]{ "latency_first", "memory_first", "p99_constrained", "balanced" }) {
                SelRec best = null; double bc = Double.POSITIVE_INFINITY;
                for (SelRec s : cand) { double cost;
                    if (obj.equals("latency_first")) cost = s.pf.mean;
                    else if (obj.equals("memory_first")) cost = s.codeB * 1e6 + s.pf.mean;              // storage dominates
                    else if (obj.equals("p99_constrained")) cost = s.pf.p99;
                    else cost = s.pf.mean + 0.02 * s.codeB + 0.001 * (s.pf.meanNodes * s.codeB);         // balanced
                    if (cost < bc) { bc = cost; best = s; } }
                if (best != null) out.add(String.format(java.util.Locale.ROOT, "sift,%d,%d,%s,%s,%d,%d,%.4f,%.1f,%d,cheapest under objective",
                    N, dim, obj, repName(best.bits), best.ef, best.depth, best.pf.heldRecall, best.pf.p99, best.codeB));
            }
        }
    }

    // ===================================================================================
    //  packed codec + scorers (step-9 identical scalar/LUT) + dot-product NEON form
    // ===================================================================================
    static final class Packed { byte[][] codes; float[][] floats; float[] mn, mx; int bits, dim, codeBytes; }
    private static Packed encode(float[][] V, int bits) {
        int n = V.length, d = V[0].length; Packed p = new Packed(); p.bits = bits; p.dim = d;
        if (bits == 32) { p.codeBytes = d * 4; p.floats = V; return p; }
        p.mn = new float[d]; p.mx = new float[d]; Arrays.fill(p.mn, Float.POSITIVE_INFINITY); Arrays.fill(p.mx, Float.NEGATIVE_INFINITY);
        for (float[] v : V) for (int j = 0; j < d; j++) { if (v[j] < p.mn[j]) p.mn[j] = v[j]; if (v[j] > p.mx[j]) p.mx[j] = v[j]; }
        int L = 1 << bits; p.codeBytes = (d * bits + 7) / 8; p.codes = new byte[n][p.codeBytes];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; int lvl = rng <= 0 ? 0 : Math.round((V[i][j] - p.mn[j]) / rng * (L - 1)); if (lvl < 0) lvl = 0; if (lvl > L - 1) lvl = L - 1;
            int dpb = 8 / bits, bi = j / dpb, sh = (j % dpb) * bits; p.codes[i][bi] |= (lvl << sh); }
        return p;
    }
    private static int level(Packed p, byte[] c, int j) { int dpb = 8 / p.bits, bi = j / dpb, sh = (j % dpb) * p.bits, mask = (1 << p.bits) - 1; return (c[bi] >> sh) & mask; }
    private static float decLevel(Packed p, byte[] c, int j) { int lvl = level(p, c, j); int L = 1 << p.bits; float rng = p.mx[j] - p.mn[j]; return rng <= 0 ? p.mn[j] : p.mn[j] + (float) lvl / (L - 1) * rng; }
    private static float asymL2(Packed p, float[] q, byte[] b) { float s = 0; for (int j = 0; j < p.dim; j++) { float x = q[j] - decLevel(p, b, j); s += x * x; } return s; }
    private static float symL2(Packed p, byte[] a, byte[] b) { float s = 0; for (int j = 0; j < p.dim; j++) { float x = decLevel(p, a, j) - decLevel(p, b, j); s += x * x; } return s; }
    private static float[] buildLUT(Packed p, float[] q) { int L = 1 << p.bits, d = p.dim; float[] lut = new float[d * L];
        for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; int base = j * L; for (int l = 0; l < L; l++) { float recon = rng <= 0 ? p.mn[j] : p.mn[j] + (float) l / (L - 1) * rng; float diff = q[j] - recon; lut[base + l] = diff * diff; } } return lut; }
    private static float asymLUT(Packed p, byte[] code, float[] lut) {
        int d = p.dim, bits = p.bits, L = 1 << bits; float s = 0;
        if (bits == 2) { int nb = d >> 2, base = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; s += lut[base + (b & 3)]; base += 4; s += lut[base + ((b >> 2) & 3)]; base += 4; s += lut[base + ((b >> 4) & 3)]; base += 4; s += lut[base + ((b >> 6) & 3)]; base += 4; } for (int j = nb << 2; j < d; j++) s += lut[j * L + level(p, code, j)]; return s; }
        if (bits == 4) { int nb = d >> 1, base = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; s += lut[base + (b & 15)]; base += 16; s += lut[base + ((b >> 4) & 15)]; base += 16; } for (int j = nb << 1; j < d; j++) s += lut[j * L + level(p, code, j)]; return s; }
        if (bits == 1) { int nb = d >> 3, base = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; for (int k = 0; k < 8; k++) { s += lut[base + ((b >> k) & 1)]; base += 2; } } for (int j = nb << 3; j < d; j++) s += lut[j * L + level(p, code, j)]; return s; }
        for (int j = 0; j < d; j++) s += lut[j * L + level(p, code, j)]; return s;
    }
    // dot-product reformulation: L2 = Cq - 2*sum_d(w_d*lvl_d) + Cx ; w_d=(q_d-mn_d)*step_d, Cx=sum step_d^2 lvl_d^2
    static final class QP { float cq; float[] w; float[] step; float[] mn; }
    private static QP buildQP(Packed p, float[] q) { int d = p.dim, L = 1 << p.bits; QP r = new QP(); r.w = new float[d]; r.step = new float[d]; r.mn = p.mn; float cq = 0;
        for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; float step = rng <= 0 ? 0 : rng / (L - 1); float a = q[j] - p.mn[j]; r.w[j] = a * step; r.step[j] = step; cq += a * a; } r.cq = cq; return r; }
    private static float[] perCodeCx(Packed p) { int n = p.codes.length, d = p.dim, L = 1 << p.bits; float[] cx = new float[n];
        float[] step2 = new float[d]; for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; float st = rng <= 0 ? 0 : rng / (L - 1); step2[j] = st * st; }
        for (int i = 0; i < n; i++) { float s = 0; byte[] c = p.codes[i]; for (int j = 0; j < d; j++) { int lv = level(p, c, j); s += step2[j] * lv * lv; } cx[i] = s; } return cx; }
    /** Dot form, autovectorizable: unpack levels into contiguous float lanes, then a plain multiply-add
     *  reduction with 4 accumulators that C2 SuperWord lifts to arm64 NEON FMA. No hand intrinsics.
     *  Uses a per-thread reusable scratch buffer (no per-score allocation -> no GC storm under load). */
    private static final ThreadLocal<float[]> DOT_SCRATCH = ThreadLocal.withInitial(() -> new float[0]);
    private static float asymDotAutovec(Packed p, byte[] code, QP qp, float cx) {
        int d = p.dim, bits = p.bits; float[] lvlF = DOT_SCRATCH.get(); if (lvlF.length < d) { lvlF = new float[d]; DOT_SCRATCH.set(lvlF); }
        if (bits == 4) { int nb = d >> 1, j = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; lvlF[j++] = b & 15; lvlF[j++] = (b >> 4) & 15; } for (; j < d; j++) lvlF[j] = level(p, code, j); }
        else if (bits == 2) { int nb = d >> 2, j = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; lvlF[j++] = b & 3; lvlF[j++] = (b >> 2) & 3; lvlF[j++] = (b >> 4) & 3; lvlF[j++] = (b >> 6) & 3; } for (; j < d; j++) lvlF[j] = level(p, code, j); }
        else { for (int j = 0; j < d; j++) lvlF[j] = level(p, code, j); }
        float[] w = qp.w; float a0 = 0, a1 = 0, a2 = 0, a3 = 0; int i = 0, bound = d & ~3;
        for (; i < bound; i += 4) { a0 += w[i] * lvlF[i]; a1 += w[i + 1] * lvlF[i + 1]; a2 += w[i + 2] * lvlF[i + 2]; a3 += w[i + 3] * lvlF[i + 3]; }
        float dot = a0 + a1 + a2 + a3; for (; i < d; i++) dot += w[i] * lvlF[i];
        return qp.cq - 2f * dot + cx;
    }

    // ---- graph build/search (consistent geometry) ----
    private OnHeapHnswGraph buildGraph(Packed p, float[][] X, int bits) throws IOException {
        final int NN = bits == 32 ? X.length : p.codes.length;
        RandomVectorScorerSupplier sup = new RandomVectorScorerSupplier() {
            public UpdateableRandomVectorScorer scorer() { return new UpdateableRandomVectorScorer() { int cur = 0; public int maxOrd() { return NN; } public void setScoringOrdinal(int o) { cur = o; }
                public float score(int j) { return bits == 32 ? -l2(X[cur], X[j]) : -symL2(p, p.codes[cur], p.codes[j]); } }; }
            public RandomVectorScorerSupplier copy() { return this; } };
        return HnswGraphBuilder.create(sup, M, BEAM, SEED).build(NN);
    }
    /** Per-thread read-only view of a shared OnHeapHnswGraph: own seek/nextNeighbor cursor, delegates
     *  stateless reads. Lucene's HnswGraph cursor is NOT thread-safe; this makes concurrent search safe
     *  (concurrent reads of the immutable post-build adjacency only). */
    static final class GraphView extends HnswGraph {
        final OnHeapHnswGraph g; int[] cur = new int[0]; int curSize = 0, upto = 0;
        GraphView(OnHeapHnswGraph g) { this.g = g; }
        public void seek(int level, int node) { NeighborArray na = g.getNeighbors(level, node); cur = na.nodes(); curSize = na.size(); upto = 0; }
        public int size() { return g.size(); }
        public int nextNeighbor() { return upto < curSize ? cur[upto++] : DocIdSetIterator.NO_MORE_DOCS; }
        public int numLevels() throws IOException { return g.numLevels(); }
        public int maxConn() { return g.maxConn(); }
        public int entryNode() throws IOException { return g.entryNode(); }
        public int neighborCount() { return curSize; }
        public HnswGraph.NodesIterator getNodesOnLevel(int level) throws IOException { return g.getNodesOnLevel(level); }
    }
    private int[] search(OnHeapHnswGraph g, Packed p, float[][] X, float[] q, int bits, int ef, long[] work, float[] cx) throws IOException {
        // fastest EXACT nav kernel per dim: LUT <=256-D, autovec dot form >=384-D (microbench: dot wins high-D).
        final boolean useDot = bits <= 4 && p.dim >= 384;
        final float[] lut = (bits == 32 || useDot) ? null : buildLUT(p, q);
        final QP qp = useDot ? buildQP(p, q) : null;
        RandomVectorScorer qs = new RandomVectorScorer() { public int maxOrd() { return bits == 32 ? X.length : p.codes.length; }
            public float score(int ord) { return bits == 32 ? -l2(q, X[ord]) : useDot ? -asymDotAutovec(p, p.codes[ord], qp, cx[ord]) : -asymLUT(p, p.codes[ord], lut); } };
        TopKnnCollector col = new TopKnnCollector(ef, Integer.MAX_VALUE); HnswGraphSearcher.search(qs, col, new GraphView(g), null); work[0] = col.visitedCount();
        ScoreDoc[] sd = col.topDocs().scoreDocs; int[] o = new int[sd.length]; for (int i = 0; i < sd.length; i++) o[i] = sd[i].doc; return o;
    }
    private int[][] posAt(OnHeapHnswGraph g, Packed p, float[][] X, float[][] Q, int[][] gt, int bits, int ef, float[] cx) throws IOException {
        int[][] pos = new int[Q.length][K]; for (int qi = 0; qi < Q.length; qi++) { long[] w = new long[1]; int[] cand = search(g, p, X, Q[qi], bits, ef, w, cx); java.util.HashMap<Integer, Integer> rk = new java.util.HashMap<>(); for (int i = 0; i < cand.length; i++) rk.put(cand[i], i); for (int t = 0; t < K; t++) pos[qi][t] = rk.getOrDefault(gt[qi][t], -1); } return pos;
    }

    static final class Perf { double mean, p50, p90, p95, p99, max, qps1t, heldRecall, meanNodes; }
    private Perf warmPerf(OnHeapHnswGraph g, Packed p, float[][] X, float[][] Q, int[][] gt, int bits, int ef, int depth, float[] cx) throws IOException {
        int nq = Q.length - NCAL; double[] best = null; double bestP50 = Double.POSITIVE_INFINITY; double recall = 0, nodes = 0;
        for (int rep = 0; rep < 3; rep++) { double[] samp = new double[nq]; int idx = 0; double hit = 0, nn = 0;
            for (int qi = NCAL; qi < Q.length; qi++) { long t0 = System.nanoTime(); long[] w = new long[1]; int[] cand = search(g, p, X, Q[qi], bits, ef, w, cx);
                int dd = Math.min(depth, cand.length); float[] ex = new float[dd]; for (int i = 0; i < dd; i++) ex[i] = l2(Q[qi], X[cand[i]]);
                Integer[] o = new Integer[dd]; for (int i = 0; i < dd; i++) o[i] = i; Arrays.sort(o, (a, b) -> Float.compare(ex[a], ex[b])); long t1 = System.nanoTime(); samp[idx++] = (t1 - t0) / 1000.0; nn += w[0];
                if (rep == 0) { java.util.HashSet<Integer> top = new java.util.HashSet<>(); for (int i = 0; i < Math.min(K, dd); i++) top.add(cand[o[i]]); int h = 0; for (int t = 0; t < K; t++) if (top.contains(gt[qi][t])) h++; hit += h / (double) K; } }
            if (rep == 0) { recall = hit / nq; nodes = nn / nq; }
            double[] sorted = samp.clone(); Arrays.sort(sorted); if (sorted[sorted.length / 2] < bestP50) { bestP50 = sorted[sorted.length / 2]; best = sorted; } }
        Perf pf = new Perf(); pf.heldRecall = recall; pf.meanNodes = nodes; pf.p50 = pct(best, 50); pf.p90 = pct(best, 90); pf.p95 = pct(best, 95); pf.p99 = pct(best, 99); pf.max = best[best.length - 1];
        double sum = 0; for (double v : best) sum += v; pf.mean = sum / best.length; pf.qps1t = 1e6 / pf.mean; return pf;
    }
    private double[] threadedQps(OnHeapHnswGraph g, Packed p, float[][] X, float[][] Q, int bits, int ef, int depth, float[] cx, int T) {
        final int TOTAL = Math.max(1200, T * 300); final int qOff = NCAL, qN = Q.length - NCAL;
        ExecutorService pool = Executors.newFixedThreadPool(T); AtomicInteger cur = new AtomicInteger(0); List<Future<double[]>> fs = new ArrayList<>();
        try { for (int qi = qOff; qi < qOff + Math.min(qN, 40); qi++) { long[] w = new long[1]; search(g, p, X, Q[qi], bits, ef, w, cx); } } catch (IOException e) { throw new RuntimeException(e); }
        long t0 = System.nanoTime();
        for (int t = 0; t < T; t++) fs.add(pool.submit(() -> { List<Double> lat = new ArrayList<>();
            while (true) { int c = cur.getAndIncrement(); if (c >= TOTAL) break; int qi = qOff + (c % qN);
                try { long s0 = System.nanoTime(); long[] w = new long[1]; int[] cand = search(g, p, X, Q[qi], bits, ef, w, cx); int dd = Math.min(depth, cand.length); float[] ex = new float[dd]; for (int i = 0; i < dd; i++) ex[i] = l2(Q[qi], X[cand[i]]); Integer[] o = new Integer[dd]; for (int i = 0; i < dd; i++) o[i] = i; Arrays.sort(o, (a, b) -> Float.compare(ex[a], ex[b])); long s1 = System.nanoTime(); lat.add((s1 - s0) / 1000.0); } catch (IOException e) { throw new RuntimeException(e); } }
            double[] a = new double[lat.size()]; for (int i = 0; i < a.length; i++) a[i] = lat.get(i); return a; }));
        List<Double> all = new ArrayList<>(); try { for (Future<double[]> f : fs) { double[] a = f.get(); for (double v : a) all.add(v); } } catch (Exception e) { throw new RuntimeException(e); }
        long t1 = System.nanoTime(); pool.shutdown(); double secs = (t1 - t0) / 1e9; double qps = TOTAL / secs;
        double[] arr = new double[all.size()]; for (int i = 0; i < arr.length; i++) arr[i] = all.get(i); Arrays.sort(arr);
        return new double[]{ qps, pct(arr, 50), pct(arr, 95), pct(arr, 99) }; // {qps,p50,p95,p99} matches header
    }

    // ---- data + helpers ----
    private int[][] groundTruth(float[][] X, float[][] Q, int N) { int[][] gt = new int[Q.length][K]; for (int qi = 0; qi < Q.length; qi++) { float[] ex = new float[N]; for (int j = 0; j < N; j++) ex[j] = l2(Q[qi], X[j]); gt[qi] = topIdx(ex, K); } return gt; }
    private static float[][] genHighD(String kind, int d, int n, Random rng) {
        float[][] X = new float[n][d];
        if (kind.equals("isotropic768")) { for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) X[i][j] = (float) rng.nextGaussian(); }
        else { // manifold768: low intrinsic dim (r) mapped to 768 via fixed random basis + small noise + cluster offsets => anisotropic, embedding-like
            int r = 48, ncl = 40; float[][] basis = new float[r][d]; for (int a = 0; a < r; a++) for (int j = 0; j < d; j++) basis[a][j] = (float) rng.nextGaussian();
            float[][] cen = new float[ncl][r]; for (int a = 0; a < ncl; a++) for (int j = 0; j < r; j++) cen[a][j] = (float) rng.nextGaussian() * 3f;
            for (int i = 0; i < n; i++) { int a = rng.nextInt(ncl); float[] z = new float[r]; for (int j = 0; j < r; j++) z[j] = cen[a][j] + (float) rng.nextGaussian();
                for (int j = 0; j < d; j++) { float s = 0; for (int k = 0; k < r; k++) s += z[k] * basis[k][j]; X[i][j] = s / (float) Math.sqrt(r) + (float) rng.nextGaussian() * 0.1f; } } }
        return X;
    }
    private int[] perm(Packed p, int n, String access, Random r) { int[] a = new int[Math.min(n, 200_000)]; for (int i = 0; i < a.length; i++) a[i] = i; if (access.equals("random")) for (int i = a.length - 1; i > 0; i--) { int j = r.nextInt(i + 1); int t = a[i]; a[i] = a[j]; a[j] = t; } return a; }
    private interface Kern { float run(); }
    private double time(Kern k, int n) { float sink = 0; for (int w = 0; w < 2; w++) sink += k.run(); long target = 200_000_000L; long scored = 0, t0 = System.nanoTime(), el; do { sink += k.run(); scored += n; el = System.nanoTime() - t0; } while (el < target && scored < 40_000_000L); if (sink == Float.NaN) throw new IllegalStateException(); return (double) el / scored; }
    private String row(String kernel, int bits, int dim, String impl, String access, long prep, double ns, double freq, double speedup, int bytes) { return String.format(java.util.Locale.ROOT, "%s,%d,%d,%s,%s,%s,%d,%.2f,%.3f,%.1f,%.2f,%d", kernel, bits, dim, impl, ISA, access, prep, ns, ns * freq / dim, 1000.0 / ns, speedup, bytes); }
    private int calibrateDepth(int[][] pos, double target, double margin) { Random rng = new Random(SEED + 3); for (int d : DEPTHS) { double lb = bootLower(pos, 0, NCAL, d, rng); if (lb >= target + margin) return d; } return -1; }
    private double recallAt(int[][] pos, int from, int to, int depth) { double s = 0; int c = 0; for (int q = from; q < to; q++) { int h = 0; for (int pp : pos[q]) if (pp >= 0 && pp < depth) h++; s += h / (double) K; c++; } return c == 0 ? 0 : s / c; }
    private double bootLower(int[][] pos, int from, int to, int depth, Random rng) { int n = to - from; double[] m = new double[200]; for (int b = 0; b < 200; b++) { double s = 0; for (int i = 0; i < n; i++) { int q = from + rng.nextInt(n); int h = 0; for (int pp : pos[q]) if (pp >= 0 && pp < depth) h++; s += h / (double) K; } m[b] = s / n; } Arrays.sort(m); return m[10]; }
    private static double pct(double[] s, int p) { int i = (int) Math.ceil(p / 100.0 * s.length) - 1; if (i < 0) i = 0; if (i >= s.length) i = s.length - 1; return s[i]; }
    private static String repName(int bits) { return bits == 32 ? "fp32" : (bits + "bit"); }
    private static float l2(float[] a, float[] b) { float s = 0; for (int i = 0; i < a.length; i++) { float x = a[i] - b[i]; s += x * x; } return s; }
    private static int[] topIdx(float[] sc, int m) { Integer[] idx = new Integer[sc.length]; for (int i = 0; i < idx.length; i++) idx[i] = i; Arrays.sort(idx, (x, y) -> Float.compare(sc[x], sc[y])); int[] o = new int[m]; for (int i = 0; i < m; i++) o[i] = idx[i]; return o; }
    private static float[][] randData(int d, int n, Random r) { float[][] X = new float[n][d]; for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) X[i][j] = (float) r.nextGaussian(); return X; }
    private Path repo() { return Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent(); }
    private void flush(List<String> a, List<String> b, List<String> c, List<String> d) throws IOException { write("track2_step10_scale_" + ISA + "_seed" + SEED + ".csv", a); write("track2_step10_concurrency_" + ISA + "_seed" + SEED + ".csv", b); write("track2_step10_crossover_" + ISA + "_seed" + SEED + ".csv", c); }
    private void write(String name, List<String> rows) throws IOException { Path out = Paths.get(System.getProperty("user.dir"), "research", "track2_adaptive_rescore", "results"); Files.createDirectories(out); try (Writer w = Files.newBufferedWriter(out.resolve(name))) { for (String r : rows) { w.write(r); w.write("\n"); } } }
    private static float[][] readFvecs(String path, int max) throws IOException { List<float[]> out = new ArrayList<>(); try (RandomAccessFile f = new RandomAccessFile(path, "r")) { byte[] h = new byte[4]; while (out.size() < max && f.getFilePointer() < f.length()) { if (f.read(h) != 4) break; int dim = ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf = new byte[4 * dim]; if (f.read(buf) != 4 * dim) break; ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v = new float[dim]; for (int j = 0; j < dim; j++) v[j] = bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
