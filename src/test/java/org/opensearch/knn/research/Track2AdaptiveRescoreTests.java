/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.research;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.quantization.models.quantizationOutput.BinaryQuantizationOutput;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;
import org.opensearch.knn.quantization.models.requests.TrainingRequest;
import org.opensearch.knn.quantization.quantizer.OneBitScalarQuantizer;

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

/**
 * Track 2, step 2 — bound-based ADAPTIVE rerank-depth policy simulator (flat scan; no OpenSearch
 * integration; no per-vector correction estimator — those are explicitly out of scope).
 *
 * SAFE-STOP RULE (correct, all-remaining form). Lower distance = better. After exact-reranking the
 * first r pool candidates (in ranking-estimator order), with exact_k = exact distance of the current
 * k-th best reranked result:
 *     STOP is certified safe  ⟺  min over ALL remaining candidates j>=r of  lower_j  >  exact_k
 * where lower_j = pred_j - z * sigma  (calibrated bound-estimator prediction ± z·σ).
 * Because calibration showed σ ≈ homoscedastic, we use a single global σ per (dataset,bound-estimator),
 * so min_j lower_j = (min_j pred_j) - z·σ, computed in O(1) per checkpoint from a precomputed
 * suffix-min-pred array. This accounts for EVERY remaining candidate, not just r+1.
 *
 * MULTIPLE-COMPARISON CORRECTION: the query-level safety claim spans m = (#remaining candidates)
 * comparisons. Bonferroni: per-comparison α' = α_query / m; two-sided z = Φ⁻¹(1 − α'/2), recomputed
 * as r grows (m shrinks). `none` mode uses α' = α_query (diagnostic; NOT a query-level guarantee).
 *
 * RANKING vs BOUND estimator are independent (e.g. ADC ranks, Hamming bounds). Bounds always use the
 * SAME candidate's bound-estimator score.
 *
 * Reuses the step-1 calibration model (affine slope/intercept + residual σ), fit per dataset/estimator.
 * Config combinations are run as a matrix; the CLI mapping is documented in TRACK2_DESIGN.md.
 */
public class Track2AdaptiveRescoreTests extends KNNTestCase {

    // config (CLI-equivalent constants; see TRACK2_DESIGN.md for --flag mapping)
    private static final int N = 10000, NQ = 100, K = 10, POOL = 1000;
    private static final int MIN_RERANK = K, MAX_RERANK = POOL, CHECK_STEP = 1;
    private static final double CONFIDENCE = 0.99;              // α_query = 1 - CONFIDENCE
    private static final int[] FIXED = { 10, 20, 30, 50, 100 }; // 1x,2x,3x,5x,min-100 (k=10)
    private static final long SEED = 11L;
    private static final String SIFT = "research/acorn/data/sift/sift_base.fvecs", SIFT_Q = "research/acorn/data/sift/sift_query.fvecs";

    enum Est { HAMMING, ADC }
    static final class Combo { final Est rank, bound; final String corr; Combo(Est r, Est b, String c){rank=r;bound=b;corr=c;} }

    public void testAdaptivePolicy() throws IOException {
        List<String> pq = new ArrayList<>();
        pq.add("dataset,dim,query_id,ranking_est,bound_est,correction,confidence,adaptive_depth,adaptive_factor,stopped_by_bound,"
            + "hit_max_fallback,false_safe,pool_has_true_topk,recall_at_k,pool_recall_at_k,oracle_depth,boundary_margin,rel_margin,difficulty");
        List<String> agg = new ArrayList<>();
        agg.add("dataset,dim,ranking_est,bound_est,correction,query_count,recall,pool_recall,false_safe_rate,min_depth_stop_rate,"
            + "max_fallback_rate,mean_depth,p50_depth,p95_depth,p99_depth,mean_oversample,oracle_mean_depth,"
            + "fixed10_rec,fixed20_rec,fixed50_rec,fixed100_rec,savings_vs_best_fixed_at_recall,cover95_boundary");

        String[][] datasets = { {"sift_real","128"}, {"isotropic","768"}, {"clustered","128"} };
        Combo[] combos = {
            new Combo(Est.HAMMING, Est.HAMMING, "none"), new Combo(Est.HAMMING, Est.HAMMING, "bonferroni"),
            new Combo(Est.ADC, Est.ADC, "none"),         new Combo(Est.ADC, Est.ADC, "bonferroni"),
            new Combo(Est.ADC, Est.HAMMING, "none"),     new Combo(Est.ADC, Est.HAMMING, "bonferroni"),
        };
        for (String[] ds : datasets) {
            String dt = ds[0]; int d = Integer.parseInt(ds[1]);
            Data data = load(dt, d);
            // per-query precompute: exact, hamming score, adc score over all N; ground-truth + pool later per combo
            float[][] hScore = new float[NQ][], aScore = new float[NQ][], exact = new float[NQ][];
            precomputeScores(data, hScore, aScore, exact);
            // calibration (affine + sigma) per bound estimator, reused across combos
            double[] calH = calibrate(hScore, exact), calA = calibrate(aScore, exact);   // {a,b,sigma}
            for (Combo c : combos) runCombo(pq, agg, dt, d, data, hScore, aScore, exact, calH, calA, c);
        }
        writeCsv("track2_adaptive_perquery.csv", pq);
        writeCsv("track2_adaptive_summary.csv", agg);
    }

    // ---------- per-combo simulation ----------
    private void runCombo(List<String> pq, List<String> agg, String dt, int d, Data data,
                          float[][] hScore, float[][] aScore, float[][] exact, double[] calH, double[] calA, Combo c) {
        double alpha = 1.0 - CONFIDENCE;
        float[][] rankScore = c.rank == Est.HAMMING ? hScore : aScore;
        float[][] boundScore = c.bound == Est.HAMMING ? hScore : aScore;
        double[] cal = c.bound == Est.HAMMING ? calH : calA; double aB = cal[0], bB = cal[1], sigma = cal[2];

        // pass 1: margins (for difficulty tertiles)
        double[] margin = new double[NQ];
        for (int q = 0; q < NQ; q++) { float[] ex = exact[q].clone(); Arrays.sort(ex); margin[q] = ex[K-1] - ex[K-2]; }
        double[] ms = margin.clone(); Arrays.sort(ms); double t1 = ms[NQ/3], t2 = ms[2*NQ/3];

        int nStop=0, nMax=0, nFalseSafe=0, poolHasAll=0; double recSum=0, poolRecSum=0; long depthSum=0, oracleSum=0;
        int[] depths = new int[NQ];
        double[] fixedRec = new double[FIXED.length];
        int boundaryCovIn=0, boundaryCovTot=0;
        for (int q = 0; q < NQ; q++) {
            // rank -> pool
            int[] cand = topIndices(rankScore[q], POOL);                 // ascending approx distance
            // exact ground truth top-k (whole corpus) and pool top-k
            int[] gtAll = topIndices(exact[q], K);
            java.util.HashSet<Integer> trueTopK = new java.util.HashSet<>(); for (int id : gtAll) trueTopK.add(id);
            // pool exact top-k (what the pool can achieve)
            int[] poolByExact = topExactWithin(cand, exact[q], K);
            java.util.HashSet<Integer> poolTopK = new java.util.HashSet<>(); for (int id : poolByExact) poolTopK.add(id);
            boolean hasAll = containsAll(cand, trueTopK);   // candidate-pool contains all true top-k?
            if (hasAll) poolHasAll++;
            // bounds + suffix-min-pred
            double[] pred = new double[POOL]; for (int r = 0; r < POOL; r++) pred[r] = aB * boundScore[q][cand[r]] + bB;
            double[] sufMinPred = new double[POOL + 1]; sufMinPred[POOL] = Double.POSITIVE_INFINITY;
            for (int r = POOL - 1; r >= 0; r--) sufMinPred[r] = Math.min(sufMinPred[r+1], pred[r]);
            // oracle depth: deepest rank of a poolTopK member + 1
            int oracle = K; for (int r = 0; r < POOL; r++) if (poolTopK.contains(cand[r])) oracle = r + 1;
            // incremental adaptive rerank with top-k max-heap of exact distances
            double[] heap = new double[K]; int hs = 0;   // max-heap (root = largest of the k smallest)
            int depth = MAX_RERANK; boolean stopped = false;
            for (int r = 1; r <= MAX_RERANK; r++) {
                double e = exact[q][cand[r-1]];
                if (hs < K) { heap[hs++] = e; if (hs == K) buildMax(heap); }
                else if (e < heap[0]) { heap[0] = e; siftDown(heap, K); }
                if (r >= MIN_RERANK && r % CHECK_STEP == 0 && hs == K) {
                    int m = POOL - r;                            // remaining comparisons
                    double z = zFor(alpha, m, c.corr);
                    double exactK = heap[0];                     // current k-th best exact
                    if (sufMinPred[r] - z * sigma > exactK) { depth = r; stopped = true; break; }  // strict >
                }
            }
            // final top-k by exact among reranked prefix [0,depth)
            java.util.HashSet<Integer> got = topExactSet(cand, exact[q], depth, K);
            double rec = overlap(got, trueTopK) / (double) K;
            double poolRec = overlap(got, poolTopK) / (double) K;
            boolean falseSafe = stopped && !got.equals(poolTopK);   // stopped by bound but missed a pool-achievable neighbour
            if (!stopped) nMax++;
            if (falseSafe) nFalseSafe++;
            recSum += rec; poolRecSum += poolRec; depthSum += depth; oracleSum += oracle; depths[q] = depth;
            // boundary conditional coverage: pred±z of candidates near exact k-th boundary
            float[] exSorted = exact[q].clone(); Arrays.sort(exSorted); double bnd = exSorted[K-1];
            for (int r = 0; r < POOL; r++) { double t = exact[q][cand[r]]; if (Math.abs(t - bnd) <= 0.05 * Math.abs(bnd) + 1e-6) {
                boundaryCovTot++; double z2 = zFor(alpha, 1, "none"); if (Math.abs(t - pred[r]) <= z2 * sigma) boundaryCovIn++; } }
            // fixed baselines (recall @ each fixed depth)
            for (int fi = 0; fi < FIXED.length; fi++) { java.util.HashSet<Integer> gf = topExactSet(cand, exact[q], Math.min(FIXED[fi], POOL), K); fixedRec[fi] += overlap(gf, trueTopK) / (double) K; }
            int diff = margin[q] <= t1 ? 0 : (margin[q] <= t2 ? 1 : 2);   // 0 hard(small margin),2 easy
            double relM = (exSorted.length > K ? (exSorted[K] - exSorted[K-1]) : 0) / Math.max(1e-9, Math.abs(exSorted[K-1]));
            pq.add(String.format(java.util.Locale.ROOT, "%s,%d,%d,%s,%s,%s,%.2f,%d,%.2f,%b,%b,%b,%b,%.3f,%.3f,%d,%.4g,%.4g,%s",
                dt, d, q, c.rank, c.bound, c.corr, CONFIDENCE, depth, depth/(double)K, stopped, !stopped, falseSafe,
                hasAll, rec, poolRec, oracle, margin[q], relM, diff==0?"hard":diff==1?"medium":"easy"));
        }
        // min-depth-stop rate: stopped exactly at MIN_RERANK
        int minStop = 0; for (int q = 0; q < NQ; q++) if (depths[q] == MIN_RERANK) minStop++;
        for (int fi = 0; fi < FIXED.length; fi++) fixedRec[fi] /= NQ;
        double recall = recSum/NQ, poolRecall = poolRecSum/NQ;
        // savings vs best fixed AT EQUIVALENT RECALL: smallest fixed depth whose recall >= adaptive recall
        int bestFixed = FIXED[FIXED.length-1]; for (int fi = 0; fi < FIXED.length; fi++) if (fixedRec[fi] >= recall - 1e-9) { bestFixed = FIXED[fi]; break; }
        double meanDepth = depthSum/(double)NQ;
        double savings = 1.0 - meanDepth / bestFixed;
        int[] ds2 = depths.clone(); Arrays.sort(ds2);
        agg.add(String.format(java.util.Locale.ROOT, "%s,%d,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%d,%d,%d,%.2f,%.1f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
            dt, d, c.rank, c.bound, c.corr, NQ, recall, poolRecall, nFalseSafe/(double)NQ, minStop/(double)NQ, nMax/(double)NQ,
            meanDepth, ds2[NQ/2], ds2[(int)(0.95*NQ)], ds2[(int)(0.99*NQ)], meanDepth/K, oracleSum/(double)NQ,
            fixedRec[0], fixedRec[1], fixedRec[3], fixedRec[4], savings, boundaryCovTot==0?0:boundaryCovIn/(double)boundaryCovTot));
        System.out.printf(java.util.Locale.ROOT, "%-10s d=%d %-7s/%-7s %-10s recall=%.3f falseSafe=%.3f maxFB=%.3f meanDepth=%.1f (oracle=%.1f) savingsVsBestFixed=%.2f%n",
            dt, d, c.rank, c.bound, c.corr, recall, nFalseSafe/(double)NQ, nMax/(double)NQ, meanDepth, oracleSum/(double)NQ, savings);
    }

    // ---------- z from confidence + correction ----------
    private static double zFor(double alphaQuery, int m, String corr) {
        double alphaPc = corr.equals("bonferroni") ? alphaQuery / Math.max(1, m) : alphaQuery;
        return invNorm(1.0 - alphaPc / 2.0);   // two-sided
    }
    // Acklam inverse normal CDF
    private static double invNorm(double p) {
        if (p <= 0) return -38; if (p >= 1) return 38;
        final double[] a = {-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00};
        final double[] b = {-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
        final double[] cc = {-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
        final double[] dd = {7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
        double plow = 0.02425, phigh = 1 - plow, q, r;
        if (p < plow) { q = Math.sqrt(-2*Math.log(p)); return (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5])/((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1); }
        if (p <= phigh) { q = p - 0.5; r = q*q; return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1); }
        q = Math.sqrt(-2*Math.log(1-p)); return -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5])/((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1);
    }

    // ---------- calibration (affine + residual sigma), reused from step 1 ----------
    private double[] calibrate(float[][] score, float[][] exact) {
        double n=0, se=0, st=0, see=0, set=0;
        for (int q = 0; q < NQ; q++) for (int j = 0; j < N; j += 5) { double e = score[q][j], t = exact[q][j]; n++; se+=e; st+=t; see+=e*e; set+=e*t; }
        double den = n*see - se*se; double a = den==0?0:(n*set - se*st)/den; double b = (st - a*se)/n;
        double v = 0; for (int q = 0; q < NQ; q++) for (int j = 0; j < N; j += 5) { double r = exact[q][j] - (a*score[q][j]+b); v += r*r; }
        return new double[] { a, b, Math.sqrt(v/n) };
    }

    // ---------- score precompute (REAL classes) ----------
    private void precomputeScores(Data data, float[][] hScore, float[][] aScore, float[][] exact) throws IOException {
        OneBitScalarQuantizer q = new OneBitScalarQuantizer(false);
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(data.X.length, false) {
            @Override public float[] getVectorAtThePosition(int p) { return data.X[p]; }
            @Override public void resetVectorValues() { }
        };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) q.train(req);
        byte[][] bits = new byte[N][]; for (int j = 0; j < N; j++) { BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); q.quantize(data.X[j].clone(), st, o); bits[j] = o.getQuantizedVector().clone(); }
        for (int qi = 0; qi < NQ; qi++) {
            float[] hs = new float[N], as = new float[N], ex = new float[N];
            byte[] qb = null; float[] qadc = data.Q[qi].clone(); q.transformWithADC(qadc, st, SpaceType.L2);
            { BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); q.quantize(data.Q[qi].clone(), st, o); qb = o.getQuantizedVector().clone(); }
            for (int j = 0; j < N; j++) { hs[j] = hamming(qb, bits[j]); as[j] = KNNScoringUtil.l2SquaredADC(qadc, bits[j]); ex[j] = l2sq(data.Q[qi], data.X[j]); }
            hScore[qi] = hs; aScore[qi] = as; exact[qi] = ex;
        }
    }

    // ---------- small helpers ----------
    private static int[] topIndices(float[] score, int m) {
        Integer[] idx = new Integer[score.length]; for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (x,y) -> Float.compare(score[x], score[y]));
        int[] o = new int[Math.min(m, idx.length)]; for (int i = 0; i < o.length; i++) o[i] = idx[i]; return o;
    }
    private static int[] topExactWithin(int[] cand, float[] exact, int k) {
        Integer[] idx = new Integer[cand.length]; for (int i = 0; i < idx.length; i++) idx[i] = cand[i];
        Arrays.sort(idx, (x,y) -> Float.compare(exact[x], exact[y])); int[] o = new int[Math.min(k, idx.length)]; for (int i = 0; i < o.length; i++) o[i] = idx[i]; return o;
    }
    private static java.util.HashSet<Integer> topExactSet(int[] cand, float[] exact, int depth, int k) {
        Integer[] idx = new Integer[Math.min(depth, cand.length)]; for (int i = 0; i < idx.length; i++) idx[i] = cand[i];
        Arrays.sort(idx, (x,y) -> Float.compare(exact[x], exact[y])); java.util.HashSet<Integer> s = new java.util.HashSet<>(); for (int i = 0; i < k && i < idx.length; i++) s.add(idx[i]); return s;
    }
    private static int overlap(java.util.HashSet<Integer> a, java.util.HashSet<Integer> b) { int c = 0; for (int x : a) if (b.contains(x)) c++; return c; }
    private static int intersectSize(int[] cand, java.util.HashSet<Integer> s) { int c = 0; for (int x : cand) if (s.contains(x)) c++; return c; }
    private static boolean containsAll(int[] cand, java.util.HashSet<Integer> s) { java.util.HashSet<Integer> cs = new java.util.HashSet<>(); for (int x : cand) cs.add(x); return cs.containsAll(s); }
    private static void buildMax(double[] h) { for (int i = h.length/2 - 1; i >= 0; i--) siftDownFrom(h, i, h.length); }
    private static void siftDown(double[] h, int n) { siftDownFrom(h, 0, n); }
    private static void siftDownFrom(double[] h, int i, int n) { while (true) { int l=2*i+1, r=2*i+2, m=i; if (l<n&&h[l]>h[m])m=l; if (r<n&&h[r]>h[m])m=r; if (m==i) break; double t=h[i];h[i]=h[m];h[m]=t; i=m; } }
    private static float l2sq(float[] a, float[] b) { float s=0; for (int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static int hamming(byte[] a, byte[] b) { int s=0; for (int i=0;i<a.length;i++) s+=Integer.bitCount((a[i]^b[i])&0xFF); return s; }

    private void writeCsv(String name, List<String> rows) throws IOException {
        Path out = Paths.get(System.getProperty("user.dir"), "research", "track2_adaptive_rescore", "results");
        Files.createDirectories(out);
        try (Writer w = Files.newBufferedWriter(out.resolve(name))) { for (String r : rows) { w.write(r); w.write("\n"); } }
        System.out.println("WROTE " + rows.size() + " rows -> " + out.resolve(name));
    }

    // ---------- data ----------
    static final class Data { float[][] X, Q; }
    private Data load(String dt, int d) throws IOException {
        Data data = new Data(); Random rng = new Random(SEED);
        if (dt.equals("sift_real")) {
            Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
            data.X = readFvecs(repo.resolve(SIFT).toString(), N); data.Q = readFvecs(repo.resolve(SIFT_Q).toString(), NQ);
        } else { data.X = gen(dt, d, N, rng); data.Q = gen(dt, d, NQ, rng); }
        return data;
    }
    private static float[][] gen(String kind, int d, int n, Random rng) {
        float[][] X = new float[n][d];
        if (kind.equals("isotropic")) { for (int i=0;i<n;i++) for (int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); }
        else if (kind.equals("clustered")) { int ncl=32; float[][] c=new float[ncl][d];
            for (int a=0;a<ncl;a++) for (int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f;
            for (int i=0;i<n;i++){ int a=rng.nextInt(ncl); for (int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } }
        else throw new IllegalArgumentException(kind);
        return X;
    }
    private static float[][] readFvecs(String path, int max) throws IOException {
        List<float[]> out = new ArrayList<>();
        try (RandomAccessFile f = new RandomAccessFile(path, "r")) { byte[] h = new byte[4];
            while (out.size() < max && f.getFilePointer() < f.length()) { if (f.read(h)!=4) break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt();
                byte[] buf = new byte[4*dim]; if (f.read(buf)!=4*dim) break; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);
                float[] v = new float[dim]; for (int j=0;j<dim;j++) v[j]=bb.getFloat(); out.add(v); } }
        return out.toArray(new float[0][]);
    }
}
