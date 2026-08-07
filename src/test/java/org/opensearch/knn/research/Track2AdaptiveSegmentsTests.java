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
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Track 2, step 11 — ADAPTIVE PER-SEGMENT PRECISION PoC. Tests the economic hypothesis: in a
 * heterogeneous multi-segment index, only genuinely hard segments need 2/4-bit, so per-segment
 * minimum-precision selection keeps the average bits/dim near 1-2 while retaining ~4-bit quality.
 *
 * Bit width is chosen at segment BUILD time (calibration sample) and baked into the segment (we never
 * "search a 1-bit segment as 4-bit"). Representation & geometry unchanged from steps 8-10.
 *
 * Heterogeneous corpus: 50 segments, 8 difficulty types (easy-128-clustered, real SIFT-128,
 * medium-384, manifold-768, isotropic-768, clustered-768, mixed-density-256, hard-1536), heavy-tailed
 * sizes. Per segment: build all 4 reps (1/2/4-bit + fp32), sweep ef, select LOWEST precision whose
 * conservative bootstrap-LB held-out-predicted recall@10 >= target, at cheapest ef. Emits long-format
 * CSVs (feasibility x targets{0.90,0.95,0.97} x calN{25,50,100,250}, per-rep latency, predictors) that
 * a Python aggregator turns into the step-4..14 tables. Selection is CALIBRATION-only; eval is held out.
 *
 * Runs in APPEND/RESUME mode (-Dstart=k) to survive the 20-min randomizedtesting suite timeout: it
 * flushes per segment and appends; rerun with -Dstart=<next> to continue. Merge simulation is a
 * separate method. Warm latency only (sandbox); default heap is enough (one segment resident at a time).
 */
public class Track2AdaptiveSegmentsTests extends KNNTestCase {

    private static final int K = 10, M = 16, BEAM = 100;
    private static final int NCALPOOL = 150, NEVAL = 60, NQ = NCALPOOL + NEVAL;
    private static final int[] EFS = { 20, 50, 100, 200, 500 };
    private static final int[] DEPTHS = { 10, 20, 30, 50, 100, 200, 300, 500 };
    private static final double[] TARGETS = { 0.90, 0.95, 0.97 };
    private static final int[] CALNS = { 25, 50, 100, 150 };
    private static final double MARGIN = 0.005;
    private static final int PRIMARY_CAL = 150;   // primary selection uses full calibration budget; cal{25,50,100} = stability study
    private static final long SEED = 71L;
    private static final int NSEG = 50;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";

    // ---- corpus spec: {typeIdx, dim, N, segSeed, siftOffset} ----
    static final String[] TYPES = { "easy128", "sift128", "medium384", "manifold768", "isotropic768", "clustered768", "mixdense256", "hard1536" };
    private static List<int[]> buildCorpus() {
        // deterministic heavy-tailed mix; counts chosen for heterogeneity, NOT to force a result
        int[] typeCounts = { 10, 8, 7, 7, 6, 6, 4, 2 };            // sums to 50
        Random r = new Random(SEED + 999);
        List<int[]> specs = new ArrayList<>();
        int siftOff = 0;
        for (int t = 0; t < TYPES.length; t++) {
            int dim = dimOf(t);
            for (int c = 0; c < typeCounts[t]; c++) {
                int[] cap = capOf(dim);                            // {min,max}
                double u = r.nextDouble();
                int N = cap[0] + (int) ((cap[1] - cap[0]) * u * u * u); // cubic -> heavy tail toward small
                int off = 0;
                if (t == 1) { off = siftOff; siftOff += N; }        // disjoint real SIFT slices
                specs.add(new int[]{ t, dim, N, (int) (SEED + t * 131 + c * 17), off });
            }
        }
        return specs;
    }
    private static int dimOf(int t) { switch (t) { case 0: return 128; case 1: return 128; case 2: return 384; case 3: case 4: case 5: return 768; case 6: return 256; default: return 1536; } }
    private static int[] capOf(int dim) { if (dim == 128) return new int[]{ 5000, 22000 }; if (dim == 256) return new int[]{ 5000, 16000 }; if (dim == 384) return new int[]{ 5000, 13000 }; if (dim == 768) return new int[]{ 4000, 10000 }; return new int[]{ 3000, 5000 }; }

    // ===================================================================================
    //  MAIN: profile segments [start, NSEG); append to CSVs; flush per segment.
    // ===================================================================================
    public void testProfileAll() throws IOException {
        // AUTO-RESUME: continue from #segments already in adaptive_predictors.csv (Gradle does not forward
        // -Dstart to the test worker, so we derive resume position from the flushed CSV itself).
        Path resumeDir = outDir();
        int existing = 0;
        Path prog = resumeDir.resolve("adaptive_predictors.csv");
        if (Files.exists(prog)) existing = Math.max(0, (int) Files.lines(prog).count() - 1);
        int start = Integer.getInteger("start", existing);
        List<int[]> corpus = buildCorpus();
        System.out.printf(java.util.Locale.ROOT, "[step11] corpus=%d segs, start=%d, heap=%.1fGB%n", corpus.size(), start, Runtime.getRuntime().maxMemory() / 1e9);
        Path dir = outDir();
        // headers (write once, when starting fresh)
        String hFeas = "seg_id,type,dim,N,bits,code_bytes_per_vec,ceiling_maxef,target,calN,feasible_ef,sel_depth,cal_lb,eval_recall";
        String hPerf = "seg_id,type,dim,N,bits,code_bytes_per_vec,feasible_095,sel_ef,sel_depth,eval_recall,warm_p50,warm_p99,mean_nodes,bytes_per_query";
        String hPred = "seg_id,type,dim,N,anisotropy,recon_err_1b,recon_err_2b,recon_err_4b,intrinsic_dim,nn_spread,cluster_sep,probe_ceil_1b_ef50,sel_bits_095";
        if (start == 0) { writeFresh(dir, "adaptive_feasible.csv", hFeas); writeFresh(dir, "adaptive_rep_perf.csv", hPerf); writeFresh(dir, "adaptive_predictors.csv", hPred); }

        for (int si = start; si < corpus.size(); si++) {
            int[] spec = corpus.get(si);
            Seg s = materialize(spec, capOf(dimOf(spec[0]))[1]);
            int dim = s.X[0].length, N = s.X.length;
            int[][] gt = groundTruth(s.X, s.Q);
            List<String> feas = new ArrayList<>(), perf = new ArrayList<>();
            double probeCeil1b = 0; int[] selBits095 = { 32 };   // min feasible bits at 0.95/cal50 (default fp32)
            // per rep
            java.util.Map<Integer, Boolean> repFeas095 = new java.util.HashMap<>();
            for (int bits : new int[]{ 1, 2, 4, 32 }) {
                Packed p = encode(s.X, bits); float[] cx = bits <= 4 ? perCodeCx(p) : null;
                OnHeapHnswGraph g = buildGraph(p, s.X, bits);
                int codeB = bits == 32 ? dim * 4 : (dim * bits + 7) / 8;
                // ef sweep: cache pos + ceiling (candidate recall over eval queries at depth=ef)
                java.util.Map<Integer, int[][]> pos = new java.util.HashMap<>();
                double ceilMax = 0;
                for (int ef : EFS) { int[][] pp = posAt(g, p, s.X, s.Q, gt, bits, ef, cx); pos.put(ef, pp);
                    ceilMax = Math.max(ceilMax, recallAt(pp, NCALPOOL, NQ, ef)); }
                if (bits == 1) probeCeil1b = recallAt(pos.get(50), NCALPOOL, NQ, 50);
                // feasibility per (target, calN): cheapest ef whose bootstrap-LB over calN cal queries has a depth with LB>=target+margin
                for (double tgt : TARGETS) for (int calN : CALNS) {
                    int feEf = -1, feDepth = -1; double calLb = 0, evalRec = 0;
                    for (int ef : EFS) { int[][] pp = pos.get(ef); int[] dl = bestDepth(pp, calN, tgt); if (dl[0] > 0) { feEf = ef; feDepth = dl[0]; calLb = dl[1] / 1e6; evalRec = recallAt(pp, NCALPOOL, NQ, feDepth); break; } }
                    feas.add(String.format(java.util.Locale.ROOT, "%d,%s,%d,%d,%d,%d,%.4f,%.2f,%d,%d,%d,%.4f,%.4f",
                        si, TYPES[spec[0]], dim, N, bits, codeB, ceilMax, tgt, calN, feEf, feDepth, calLb, evalRec));
                    if (tgt == 0.95 && calN == PRIMARY_CAL && feEf > 0 && bits < selBits095[0]) selBits095[0] = bits;
                }
                // per-rep latency at 0.95/cal50 selected config (fallback: max ef if infeasible)
                int[] sel = selectAt(pos, 0.95, PRIMARY_CAL); boolean feasible = sel[0] > 0;
                int useEf = feasible ? sel[0] : EFS[EFS.length - 1], useDepth = feasible ? sel[1] : DEPTHS[DEPTHS.length - 1];
                double[] lat = warmLatency(g, p, s.X, s.Q, gt, bits, useEf, useDepth, cx);
                repFeas095.put(bits, feasible);
                perf.add(String.format(java.util.Locale.ROOT, "%d,%s,%d,%d,%d,%d,%d,%d,%d,%.4f,%.1f,%.1f,%.1f,%.0f",
                    si, TYPES[spec[0]], dim, N, bits, codeB, feasible ? 1 : 0, useEf, useDepth, lat[2], lat[0], lat[1], lat[3], lat[3] * codeB));
            }
            double[] pr = predictors(s.X);
            List<String> pred = new ArrayList<>();
            pred.add(String.format(java.util.Locale.ROOT, "%d,%s,%d,%d,%.3f,%.4f,%.4f,%.4f,%.1f,%.3f,%.3f,%.4f,%d",
                si, TYPES[spec[0]], dim, N, pr[0], pr[1], pr[2], pr[3], pr[4], pr[5], pr[6], probeCeil1b, selBits095[0]));
            append(dir, "adaptive_feasible.csv", feas); append(dir, "adaptive_rep_perf.csv", perf); append(dir, "adaptive_predictors.csv", pred);
            System.out.printf(java.util.Locale.ROOT, "[step11] seg %d/%d %s dim=%d N=%d -> sel@0.95=%s (feas 1b=%s 2b=%s 4b=%s)%n",
                si, corpus.size() - 1, TYPES[spec[0]], dim, N, selBits095[0] == 32 ? "fp32" : selBits095[0] + "bit", repFeas095.get(1), repFeas095.get(2), repFeas095.get(4));
        }
        System.out.println("[step11] DONE profiling through " + (corpus.size() - 1));
    }

    // ===================================================================================
    //  MERGE simulation (step 12): merge same-dim pairs, re-profile combined, compare bits.
    // ===================================================================================
    public void testMerge() throws IOException {
        Path dir = outDir(); writeFresh(dir, "adaptive_merge.csv", "merge_id,dim,type_a,type_b,Na,Nb,srcA_bits,srcB_bits,merged_N,merged_bits,merged_ef");
        // same-dim pairs, small N, avoiding the pathological isotropic-768 build. The key cases: 1-bit(SIFT)
        // + 4-bit(iid) [does merge escalate?], 1-bit+1-bit [retain?], 4-bit+4-bit, and cross-dim structured.
        // types: 0=iid128, 1=sift128. 128-D only (fast; high-D isotropic builds are pathologically slow).
        // Covers the key cases: 1-bit(SIFT)+4-bit(iid) [escalate?], 1-bit+1-bit [retain], 4-bit+4-bit.
        int[][] pairs = { {1,0,128},{1,1,128},{0,0,128},{1,0,128},{0,1,128} };
        List<String> rows = new ArrayList<>();
        for (int mi = 0; mi < pairs.length; mi++) {
            int ta = pairs[mi][0], tb = pairs[mi][1], dim = pairs[mi][2];
            int Na = 6000 + mi * 500, Nb = 7000 + mi * 400;
            Seg a = materialize(new int[]{ ta, dim, Na, (int) (SEED + 700 + mi), mi * 6000 }, Na);
            Seg b = materialize(new int[]{ tb, dim, Nb, (int) (SEED + 800 + mi), 200000 + mi * 7000 }, Nb);
            int aBits = selectSegBits(a), bBits = selectSegBits(b);
            // merge vectors + queries
            float[][] mX = concat(a.X, b.X); float[][] mQ = a.Q;   // use A's queries (in-distribution enough for the mixed set)
            Seg m = new Seg(); m.X = mX; m.Q = mQ;
            int[] mSel = selectSegBitsEf(m);
            rows.clear();
            rows.add(String.format(java.util.Locale.ROOT, "%d,%d,%s,%s,%d,%d,%s,%s,%d,%s,%d",
                mi, dim, TYPES[ta], TYPES[tb], Na, Nb, bitName(aBits), bitName(bBits), mX.length, bitName(mSel[0]), mSel[1]));
            append(dir, "adaptive_merge.csv", rows);   // flush per merge
            System.out.printf(java.util.Locale.ROOT, "[step11-merge] %s(%s)+%s(%s) -> %s%n", TYPES[ta], bitName(aBits), TYPES[tb], bitName(bBits), bitName(mSel[0]));
        }
    }

    private int selectSegBits(Seg s) throws IOException { return selectSegBitsEf(s)[0]; }
    private int[] selectSegBitsEf(Seg s) throws IOException {
        int[][] gt = groundTruth(s.X, s.Q);
        for (int bits : new int[]{ 1, 2, 4, 32 }) {
            Packed p = encode(s.X, bits); float[] cx = bits <= 4 ? perCodeCx(p) : null; OnHeapHnswGraph g = buildGraph(p, s.X, bits);
            java.util.Map<Integer, int[][]> pos = new java.util.HashMap<>();
            for (int ef : EFS) pos.put(ef, posAt(g, p, s.X, s.Q, gt, bits, ef, cx));
            int[] sel = selectAt(pos, 0.95, PRIMARY_CAL); if (sel[0] > 0) return new int[]{ bits, sel[0] };
        }
        return new int[]{ 32, EFS[EFS.length - 1] };
    }
    private static String bitName(int b) { return b == 32 ? "fp32" : b + "bit"; }

    // ---- selection helpers ----
    /** cheapest ef whose bootstrap-LB over first calN cal queries has a depth with LB>=target+margin; {ef,depth} or {-1,-1}. */
    private int[] selectAt(java.util.Map<Integer, int[][]> pos, double target, int calN) {
        for (int ef : EFS) { int[] dl = bestDepth(pos.get(ef), calN, target); if (dl[0] > 0) return new int[]{ ef, dl[0] }; }
        return new int[]{ -1, -1 };
    }
    /** smallest depth whose bootstrap p10 LB over first calN cal queries >= target+margin; returns {depth, lb*1e6} or {-1,..}. */
    private int[] bestDepth(int[][] pos, int calN, double target) {
        Random rng = new Random(SEED + 3);
        for (int d : DEPTHS) { double lb = bootLower(pos, 0, calN, d, rng); if (lb >= target + MARGIN) return new int[]{ d, (int) (lb * 1e6) }; }
        return new int[]{ -1, -1 };
    }

    // ===================================================================================
    //  predictors (step 10): anisotropy, recon err 1/2/4-bit, intrinsic dim (two-NN), nn spread, cluster sep
    // ===================================================================================
    private double[] predictors(float[][] X) {
        int n = X.length, d = X[0].length; Random r = new Random(SEED + 5);
        int S = Math.min(n, 2000); int[] idx = sample(n, S, r);
        // anisotropy = max per-dim var / mean per-dim var
        double[] mean = new double[d]; for (int i : idx) for (int j = 0; j < d; j++) mean[j] += X[i][j]; for (int j = 0; j < d; j++) mean[j] /= S;
        double[] var = new double[d]; for (int i : idx) for (int j = 0; j < d; j++) { double e = X[i][j] - mean[j]; var[j] += e * e; }
        double vmax = 0, vsum = 0; for (int j = 0; j < d; j++) { var[j] /= S; vmax = Math.max(vmax, var[j]); vsum += var[j]; }
        double aniso = vsum > 0 ? vmax / (vsum / d) : 1;
        // recon error per bit width (relative)
        double e1 = reconErr(X, idx, 1), e2 = reconErr(X, idx, 2), e4 = reconErr(X, idx, 4);
        // nn distances over a small ref set
        int SR = Math.min(n, 1200); int[] ref = sample(n, SR, new Random(SEED + 6)); int SP = Math.min(SR, 400);
        double[] r1 = new double[SP], r2 = new double[SP];
        for (int a = 0; a < SP; a++) { float[] q = X[ref[a]]; double b1 = Double.MAX_VALUE, b2 = Double.MAX_VALUE;
            for (int b = 0; b < SR; b++) { if (ref[b] == ref[a]) continue; double dd = Math.sqrt(l2(q, X[ref[b]])); if (dd < b1) { b2 = b1; b1 = dd; } else if (dd < b2) b2 = dd; }
            r1[a] = b1; r2[a] = b2; }
        // intrinsic dim (two-NN / Facco): d = (SP) / sum(log(r2/r1))
        double sumlog = 0; int cnt = 0; for (int a = 0; a < SP; a++) if (r1[a] > 1e-9 && r2[a] > r1[a]) { sumlog += Math.log(r2[a] / r1[a]); cnt++; }
        double intrinsic = cnt > 0 && sumlog > 0 ? cnt / sumlog : d;
        // nn spread = std/mean of 1-NN dist
        double m1 = 0; for (double v : r1) m1 += v; m1 /= SP; double s1 = 0; for (double v : r1) s1 += (v - m1) * (v - m1); s1 = Math.sqrt(s1 / SP);
        double nnSpread = m1 > 0 ? s1 / m1 : 0;
        // cluster separation = mean pairwise dist / mean 1-NN dist (high -> sparse/separated)
        double mp = 0; int pc = 0; for (int a = 0; a < 200; a++) { int i = ref[r.nextInt(SR)], j = ref[r.nextInt(SR)]; if (i != j) { mp += Math.sqrt(l2(X[i], X[j])); pc++; } }
        mp = pc > 0 ? mp / pc : 0; double clusterSep = m1 > 0 ? mp / m1 : 0;
        return new double[]{ aniso, e1, e2, e4, intrinsic, nnSpread, clusterSep };
    }
    private double reconErr(float[][] X, int[] idx, int bits) { int d = X[0].length, L = 1 << bits;
        float[] mn = new float[d], mx = new float[d]; Arrays.fill(mn, Float.POSITIVE_INFINITY); Arrays.fill(mx, Float.NEGATIVE_INFINITY);
        for (int i : idx) for (int j = 0; j < d; j++) { if (X[i][j] < mn[j]) mn[j] = X[i][j]; if (X[i][j] > mx[j]) mx[j] = X[i][j]; }
        double num = 0, den = 0; for (int i : idx) { float[] x = X[i]; for (int j = 0; j < d; j++) { float rng = mx[j] - mn[j]; int lvl = rng <= 0 ? 0 : Math.round((x[j] - mn[j]) / rng * (L - 1)); float rec = rng <= 0 ? mn[j] : mn[j] + (float) lvl / (L - 1) * rng; double e = x[j] - rec; num += e * e; den += (double) x[j] * x[j]; } }
        return den > 0 ? num / den : 0;
    }
    private static int[] sample(int n, int s, Random r) { int[] a = new int[n]; for (int i = 0; i < n; i++) a[i] = i; for (int i = n - 1; i > 0 && i >= n - s; i--) { int j = r.nextInt(i + 1); int t = a[i]; a[i] = a[j]; a[j] = t; } int[] o = new int[s]; System.arraycopy(a, n - s, o, 0, s); return o; }

    // ===================================================================================
    //  corpus materialization
    // ===================================================================================
    static final class Seg { float[][] X, Q; }
    private Seg materialize(int[] spec, int capMax) throws IOException {
        int type = spec[0], dim = spec[1], N = Math.min(spec[2], capMax), off = spec[4]; long sd = spec[3];
        Seg s = new Seg();
        if (type == 1) { // real SIFT slice
            Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
            s.X = readFvecs(repo.resolve(BASE).toString(), off, N);
            s.Q = readFvecs(repo.resolve(QUERY).toString(), (off / 1000) % 9000, NQ);
            if (s.Q.length < NQ) s.Q = readFvecs(repo.resolve(QUERY).toString(), 0, NQ);
        } else {
            s.X = genSeg(TYPES[type], dim, N, new Random(sd));
            s.Q = genSeg(TYPES[type], dim, NQ, new Random(sd + 12345));
        }
        return s;
    }
    /* Corpus difficulty is driven by QUANTIZATION structure (intrinsic dim / isotropy / ambient dim),
     * NOT by cluster separation -- all distributions are graph-navigable so fp32 reaches high recall and
     * the variable of interest is "how many bits to approach fp32". Low-rank manifolds (few high-variance
     * directions) are anisotropic -> quantization-friendly (1-2 bit); iid Gaussian is isotropic/full-rank
     * -> quantization-hard (4-bit); higher ambient dim amplifies difficulty. */
    private static float[][] genSeg(String type, int d, int n, Random r) {
        // NOT engineered to a desired mix: a natural hardness spectrum. iid Gaussian across ambient dims
        // (axis-aligned quantization: harder as dim grows) + a couple structured manifolds. Real SIFT (type
        // "sift128", handled in materialize) is the empirically 1-bit-friendly anchor. Report what falls out.
        switch (type) {
            case "manifold768": return manifold(d, 64,  0.10f, n, r);   // structured high-D (moderate)
            case "clustered768":return manifold(d, 192, 0.15f, n, r);   // near-full-rank high-D (harder)
            default:            { float[][] X = new float[n][d]; for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) X[i][j] = (float) r.nextGaussian(); return X; } // iid Gaussian, dim-driven hardness
        }
    }
    /** n points on an r-dim linear manifold embedded in R^d (fixed random orthogonal-ish basis) + gaussian noise. */
    private static float[][] manifold(int d, int r, float noise, int n, Random rng) {
        r = Math.min(r, d); float[][] basis = new float[r][d];
        for (int a = 0; a < r; a++) { for (int j = 0; j < d; j++) basis[a][j] = (float) rng.nextGaussian(); }
        float[][] X = new float[n][d]; float inv = 1f / (float) Math.sqrt(r);
        for (int i = 0; i < n; i++) { float[] z = new float[r]; for (int k = 0; k < r; k++) z[k] = (float) rng.nextGaussian();
            for (int j = 0; j < d; j++) { float s = 0; for (int k = 0; k < r; k++) s += z[k] * basis[k][j]; X[i][j] = s * inv + (float) rng.nextGaussian() * noise; } }
        return X;
    }
    private static float[][] concat(float[][] a, float[][] b) { float[][] o = new float[a.length + b.length][]; System.arraycopy(a, 0, o, 0, a.length); System.arraycopy(b, 0, o, a.length, b.length); return o; }

    // ===================================================================================
    //  codec + scorers + graph (from step 10; LUT nav <=256-D, autovec dot >=384-D)
    // ===================================================================================
    static final class Packed { byte[][] codes; float[][] floats; float[] mn, mx; int bits, dim, codeBytes; }
    private static Packed encode(float[][] V, int bits) { int n = V.length, d = V[0].length; Packed p = new Packed(); p.bits = bits; p.dim = d;
        if (bits == 32) { p.codeBytes = d * 4; p.floats = V; return p; }
        p.mn = new float[d]; p.mx = new float[d]; Arrays.fill(p.mn, Float.POSITIVE_INFINITY); Arrays.fill(p.mx, Float.NEGATIVE_INFINITY);
        for (float[] v : V) for (int j = 0; j < d; j++) { if (v[j] < p.mn[j]) p.mn[j] = v[j]; if (v[j] > p.mx[j]) p.mx[j] = v[j]; }
        int L = 1 << bits; p.codeBytes = (d * bits + 7) / 8; p.codes = new byte[n][p.codeBytes];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; int lvl = rng <= 0 ? 0 : Math.round((V[i][j] - p.mn[j]) / rng * (L - 1)); if (lvl < 0) lvl = 0; if (lvl > L - 1) lvl = L - 1; int dpb = 8 / bits, bi = j / dpb, sh = (j % dpb) * bits; p.codes[i][bi] |= (lvl << sh); } return p; }
    private static int level(Packed p, byte[] c, int j) { int dpb = 8 / p.bits, bi = j / dpb, sh = (j % dpb) * p.bits, mask = (1 << p.bits) - 1; return (c[bi] >> sh) & mask; }
    private static float decLevel(Packed p, byte[] c, int j) { int lvl = level(p, c, j); int L = 1 << p.bits; float rng = p.mx[j] - p.mn[j]; return rng <= 0 ? p.mn[j] : p.mn[j] + (float) lvl / (L - 1) * rng; }
    private static float symL2(Packed p, byte[] a, byte[] b) { float s = 0; for (int j = 0; j < p.dim; j++) { float x = decLevel(p, a, j) - decLevel(p, b, j); s += x * x; } return s; }
    private static float asymLUT(Packed p, byte[] code, float[] lut) { int d = p.dim, bits = p.bits, L = 1 << bits; float s = 0;
        if (bits == 2) { int nb = d >> 2, base = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; s += lut[base + (b & 3)]; base += 4; s += lut[base + ((b >> 2) & 3)]; base += 4; s += lut[base + ((b >> 4) & 3)]; base += 4; s += lut[base + ((b >> 6) & 3)]; base += 4; } for (int j = nb << 2; j < d; j++) s += lut[j * L + level(p, code, j)]; return s; }
        if (bits == 4) { int nb = d >> 1, base = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; s += lut[base + (b & 15)]; base += 16; s += lut[base + ((b >> 4) & 15)]; base += 16; } for (int j = nb << 1; j < d; j++) s += lut[j * L + level(p, code, j)]; return s; }
        if (bits == 1) { int nb = d >> 3, base = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; for (int k = 0; k < 8; k++) { s += lut[base + ((b >> k) & 1)]; base += 2; } } for (int j = nb << 3; j < d; j++) s += lut[j * L + level(p, code, j)]; return s; }
        for (int j = 0; j < d; j++) s += lut[j * L + level(p, code, j)]; return s; }
    private static float[] buildLUT(Packed p, float[] q) { int L = 1 << p.bits, d = p.dim; float[] lut = new float[d * L]; for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; int base = j * L; for (int l = 0; l < L; l++) { float recon = rng <= 0 ? p.mn[j] : p.mn[j] + (float) l / (L - 1) * rng; float diff = q[j] - recon; lut[base + l] = diff * diff; } } return lut; }
    static final class QP { float cq; float[] w; }
    private static QP buildQP(Packed p, float[] q) { int d = p.dim, L = 1 << p.bits; QP r = new QP(); r.w = new float[d]; float cq = 0; for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; float step = rng <= 0 ? 0 : rng / (L - 1); float a = q[j] - p.mn[j]; r.w[j] = a * step; cq += a * a; } r.cq = cq; return r; }
    private static float[] perCodeCx(Packed p) { int n = p.codes.length, d = p.dim, L = 1 << p.bits; float[] cx = new float[n]; float[] step2 = new float[d]; for (int j = 0; j < d; j++) { float rng = p.mx[j] - p.mn[j]; float st = rng <= 0 ? 0 : rng / (L - 1); step2[j] = st * st; } for (int i = 0; i < n; i++) { float s = 0; byte[] c = p.codes[i]; for (int j = 0; j < d; j++) { int lv = level(p, c, j); s += step2[j] * lv * lv; } cx[i] = s; } return cx; }
    private static final ThreadLocal<float[]> DOT_SCRATCH = ThreadLocal.withInitial(() -> new float[0]);
    private static float asymDot(Packed p, byte[] code, QP qp, float cx) { int d = p.dim, bits = p.bits; float[] lvlF = DOT_SCRATCH.get(); if (lvlF.length < d) { lvlF = new float[d]; DOT_SCRATCH.set(lvlF); }
        if (bits == 4) { int nb = d >> 1, j = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; lvlF[j++] = b & 15; lvlF[j++] = (b >> 4) & 15; } for (; j < d; j++) lvlF[j] = level(p, code, j); }
        else if (bits == 2) { int nb = d >> 2, j = 0; for (int bi = 0; bi < nb; bi++) { int b = code[bi] & 0xFF; lvlF[j++] = b & 3; lvlF[j++] = (b >> 2) & 3; lvlF[j++] = (b >> 4) & 3; lvlF[j++] = (b >> 6) & 3; } for (; j < d; j++) lvlF[j] = level(p, code, j); }
        else { for (int j = 0; j < d; j++) lvlF[j] = level(p, code, j); }
        float[] w = qp.w; float a0 = 0, a1 = 0, a2 = 0, a3 = 0; int i = 0, bound = d & ~3; for (; i < bound; i += 4) { a0 += w[i] * lvlF[i]; a1 += w[i + 1] * lvlF[i + 1]; a2 += w[i + 2] * lvlF[i + 2]; a3 += w[i + 3] * lvlF[i + 3]; } float dot = a0 + a1 + a2 + a3; for (; i < d; i++) dot += w[i] * lvlF[i]; return qp.cq - 2f * dot + cx; }

    private OnHeapHnswGraph buildGraph(Packed p, float[][] X, int bits) throws IOException { final int NN = bits == 32 ? X.length : p.codes.length;
        RandomVectorScorerSupplier sup = new RandomVectorScorerSupplier() { public UpdateableRandomVectorScorer scorer() { return new UpdateableRandomVectorScorer() { int cur = 0; public int maxOrd() { return NN; } public void setScoringOrdinal(int o) { cur = o; } public float score(int j) { return bits == 32 ? -l2(X[cur], X[j]) : -symL2(p, p.codes[cur], p.codes[j]); } }; } public RandomVectorScorerSupplier copy() { return this; } };
        return HnswGraphBuilder.create(sup, M, BEAM, SEED).build(NN); }
    static final class GraphView extends HnswGraph { final OnHeapHnswGraph g; int[] cur = new int[0]; int curSize = 0, upto = 0; GraphView(OnHeapHnswGraph g) { this.g = g; }
        public void seek(int level, int node) { NeighborArray na = g.getNeighbors(level, node); cur = na.nodes(); curSize = na.size(); upto = 0; } public int size() { return g.size(); } public int nextNeighbor() { return upto < curSize ? cur[upto++] : DocIdSetIterator.NO_MORE_DOCS; } public int numLevels() throws IOException { return g.numLevels(); } public int maxConn() { return g.maxConn(); } public int entryNode() throws IOException { return g.entryNode(); } public int neighborCount() { return curSize; } public HnswGraph.NodesIterator getNodesOnLevel(int l) throws IOException { return g.getNodesOnLevel(l); } }
    private int[] search(OnHeapHnswGraph g, Packed p, float[][] X, float[] q, int bits, int ef, long[] work, float[] cx) throws IOException {
        final boolean useDot = bits <= 4 && p.dim >= 384; final float[] lut = (bits == 32 || useDot) ? null : buildLUT(p, q); final QP qp = useDot ? buildQP(p, q) : null;
        RandomVectorScorer qs = new RandomVectorScorer() { public int maxOrd() { return bits == 32 ? X.length : p.codes.length; } public float score(int ord) { return bits == 32 ? -l2(q, X[ord]) : useDot ? -asymDot(p, p.codes[ord], qp, cx[ord]) : -asymLUT(p, p.codes[ord], lut); } };
        TopKnnCollector col = new TopKnnCollector(ef, Integer.MAX_VALUE); HnswGraphSearcher.search(qs, col, new GraphView(g), null); work[0] = col.visitedCount();
        ScoreDoc[] sd = col.topDocs().scoreDocs; int[] o = new int[sd.length]; for (int i = 0; i < sd.length; i++) o[i] = sd[i].doc; return o; }
    private int[][] posAt(OnHeapHnswGraph g, Packed p, float[][] X, float[][] Q, int[][] gt, int bits, int ef, float[] cx) throws IOException {
        int[][] pos = new int[Q.length][K]; for (int qi = 0; qi < Q.length; qi++) { long[] w = new long[1]; int[] cand = search(g, p, X, Q[qi], bits, ef, w, cx); java.util.HashMap<Integer, Integer> rk = new java.util.HashMap<>(); for (int i = 0; i < cand.length; i++) rk.put(cand[i], i); for (int t = 0; t < K; t++) pos[qi][t] = rk.getOrDefault(gt[qi][t], -1); } return pos; }
    /** {p50,p99,eval_recall,mean_nodes} over held-out queries at (ef,depth). */
    private double[] warmLatency(OnHeapHnswGraph g, Packed p, float[][] X, float[][] Q, int[][] gt, int bits, int ef, int depth, float[] cx) throws IOException {
        int nq = NEVAL; double[] samp = new double[nq]; int idx = 0; double hit = 0, nodes = 0;
        for (int qi = NCALPOOL; qi < NQ; qi++) { long t0 = System.nanoTime(); long[] w = new long[1]; int[] cand = search(g, p, X, Q[qi], bits, ef, w, cx); int dd = Math.min(depth, cand.length); float[] ex = new float[dd]; for (int i = 0; i < dd; i++) ex[i] = l2(Q[qi], X[cand[i]]); Integer[] o = new Integer[dd]; for (int i = 0; i < dd; i++) o[i] = i; Arrays.sort(o, (a, b) -> Float.compare(ex[a], ex[b])); long t1 = System.nanoTime(); samp[idx++] = (t1 - t0) / 1000.0; nodes += w[0];
            java.util.HashSet<Integer> top = new java.util.HashSet<>(); for (int i = 0; i < Math.min(K, dd); i++) top.add(cand[o[i]]); int h = 0; for (int t = 0; t < K; t++) if (top.contains(gt[qi][t])) h++; hit += h / (double) K; }
        double[] sorted = samp.clone(); Arrays.sort(sorted); return new double[]{ pct(sorted, 50), pct(sorted, 99), hit / nq, nodes / nq }; }

    // ---- misc ----
    private int[][] groundTruth(float[][] X, float[][] Q) { int N = X.length; int[][] gt = new int[Q.length][K]; for (int qi = 0; qi < Q.length; qi++) { float[] ex = new float[N]; for (int j = 0; j < N; j++) ex[j] = l2(Q[qi], X[j]); gt[qi] = topIdx(ex, K); } return gt; }
    private double recallAt(int[][] pos, int from, int to, int depth) { double s = 0; int c = 0; for (int q = from; q < to; q++) { int h = 0; for (int pp : pos[q]) if (pp >= 0 && pp < depth) h++; s += h / (double) K; c++; } return c == 0 ? 0 : s / c; }
    private double bootLower(int[][] pos, int from, int to, int depth, Random rng) { int n = to - from; double[] m = new double[200]; for (int b = 0; b < 200; b++) { double s = 0; for (int i = 0; i < n; i++) { int q = from + rng.nextInt(n); int h = 0; for (int pp : pos[q]) if (pp >= 0 && pp < depth) h++; s += h / (double) K; } m[b] = s / n; } Arrays.sort(m); return m[10]; }
    private static double pct(double[] s, int p) { int i = (int) Math.ceil(p / 100.0 * s.length) - 1; if (i < 0) i = 0; if (i >= s.length) i = s.length - 1; return s[i]; }
    private static float l2(float[] a, float[] b) { float s = 0; for (int i = 0; i < a.length; i++) { float x = a[i] - b[i]; s += x * x; } return s; }
    private static int[] topIdx(float[] sc, int m) { Integer[] idx = new Integer[sc.length]; for (int i = 0; i < idx.length; i++) idx[i] = i; Arrays.sort(idx, (x, y) -> Float.compare(sc[x], sc[y])); int[] o = new int[m]; for (int i = 0; i < m; i++) o[i] = idx[i]; return o; }
    private Path outDir() throws IOException { Path out = Paths.get(System.getProperty("user.dir"), "research", "track2_adaptive_rescore", "results"); Files.createDirectories(out); return out; }
    private void writeFresh(Path dir, String name, String header) throws IOException { try (Writer w = Files.newBufferedWriter(dir.resolve(name))) { w.write(header); w.write("\n"); } }
    private void append(Path dir, String name, List<String> rows) throws IOException { try (Writer w = Files.newBufferedWriter(dir.resolve(name), StandardOpenOption.CREATE, StandardOpenOption.APPEND)) { for (String r : rows) { w.write(r); w.write("\n"); } } }
    private static float[][] readFvecs(String path, int skip, int max) throws IOException { List<float[]> out = new ArrayList<>(); try (RandomAccessFile f = new RandomAccessFile(path, "r")) { byte[] h = new byte[4]; int seen = 0; while (out.size() < max && f.getFilePointer() < f.length()) { if (f.read(h) != 4) break; int dim = ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf = new byte[4 * dim]; if (f.read(buf) != 4 * dim) break; if (seen++ < skip) continue; ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v = new float[dim]; for (int j = 0; j < dim; j++) v[j] = bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
