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
 * Track 2, step 4 — Elastic-style AUTO-CALIBRATED FIXED rerank depth (dataset/segment-level recall
 * SLA), the practical alternative to the per-query Bonferroni policy (which reranks ~full pool).
 * Benchmark only; the safe adaptive policy is untouched and cited as a baseline (~pool depth).
 *
 * Central question: can dataset/segment-level calibration pick a MUCH cheaper fixed depth that
 * reliably meets a target average recall, without per-query simultaneous confidence bounds?
 *
 * Method: recall@r = fraction of true top-k within the first r approximate candidates (monotone in
 * r). On a CALIBRATION query split, pick the smallest depth in a grid whose mean recall (or a
 * bootstrap lower bound, "conservative" mode) meets target; VALIDATE on disjoint held-out folds.
 * Ranking estimators: ADC, Hamming, Corrected (uses the step-3 8-byte metadata). Baselines:
 * min-100, best-fixed-in-hindsight, full-pool (~Bonferroni adaptive), per-query oracle.
 * Global (one depth) vs per-segment (per-dataset) calibration quantifies local-calibration savings.
 */
public class Track2AutoCalibrationTests extends KNNTestCase {

    private static final int N = 10000, NQ = 250, K = 10, POOL = 1000, FOLDS = 5, BOOT = 300;
    private static final int[] DEPTHS = { 10, 20, 30, 50, 100, 150, 200, 300, 500, 1000 };
    private static final double[] TARGETS = { 0.90, 0.95, 0.99 };
    private static final double MARGIN = 0.005;
    private static final long SEED = 31L;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";

    enum Est { ADC, HAMMING, CORRECTED }

    public void testAutoCalibration() throws IOException {
        List<String> summary = new ArrayList<>();
        summary.add("dataset,estimator,k,pool,target,mode,cal_qcount,mean_selected_depth,std_selected_depth,mean_heldout_recall,"
            + "worst_heldout_recall,sla_miss_rate,best_fixed_depth,oracle_mean_depth,min100_recall,savings_vs_min100,savings_vs_fullpool,gap_to_best_fixed,gap_to_oracle,pool_recall");
        List<String> seg = new ArrayList<>();
        seg.add("estimator,target,segment,per_segment_depth,per_segment_recall,global_depth,global_recall_on_segment,global_meets");

        String[][] datasets = { {"sift_real","128"}, {"isotropic","768"}, {"clustered","128"} };
        // positions[dataset][est][q][k] = rank of each true-top-k member in the estimator ordering (or >=POOL if outside)
        java.util.Map<String,int[][][]> POS = new java.util.LinkedHashMap<>();
        java.util.Map<String,Integer> DIM = new java.util.LinkedHashMap<>();
        for (String[] ds : datasets) { int[][][] p = computePositions(ds[0], Integer.parseInt(ds[1])); POS.put(ds[0], p); DIM.put(ds[0], p==null?0:Integer.parseInt(ds[1])); }

        // ---- global calibration per dataset x estimator x target x mode, with FOLDS ----
        for (String[] ds : datasets) {
            String dt = ds[0]; int[][][] pos = POS.get(dt);
            for (Est e : Est.values()) {
                int ei = e.ordinal();
                for (double target : TARGETS) for (String mode : new String[]{"mean","conservative"}) {
                    List<Integer> sel = new ArrayList<>(); List<Double> held = new ArrayList<>();
                    Random rng = new Random(SEED);
                    int[] perm = shuffle(NQ, rng);
                    for (int f = 0; f < FOLDS; f++) {
                        int[] calIdx = foldComplement(perm, f, FOLDS), evalIdx = fold(perm, f, FOLDS);
                        int depth = calibrate(pos, ei, calIdx, target, mode, rng);
                        double rec = meanRecallAt(pos, ei, evalIdx, depth);
                        sel.add(depth); held.add(rec);
                    }
                    // hindsight + oracle + pool on ALL queries
                    int bestFixed = bestFixedInHindsight(pos, ei, all(NQ), target);
                    double oracleMean = oracleMeanDepth(pos, ei, all(NQ));
                    double min100rec = meanRecallAt(pos, ei, all(NQ), 100);
                    double poolRec = meanRecallAt(pos, ei, all(NQ), POOL);
                    double meanSel = mean(sel), stdSel = std(sel), meanHeld = meanD(held), worstHeld = min(held);
                    double slaMiss = held.stream().filter(r -> r < target).count() / (double) FOLDS;
                    summary.add(String.format(java.util.Locale.ROOT, "%s,%s,%d,%d,%.2f,%s,%d,%.1f,%.1f,%.4f,%.4f,%.2f,%d,%.1f,%.4f,%.3f,%.3f,%d,%.1f,%.4f",
                        dt, e, K, POOL, target, mode, (FOLDS-1)*NQ/FOLDS, meanSel, stdSel, meanHeld, worstHeld, slaMiss,
                        bestFixed, oracleMean, min100rec, 1.0-meanSel/100.0, 1.0-meanSel/POOL, (int)Math.round(meanSel)-bestFixed, meanSel-oracleMean, poolRec));
                    System.out.printf(java.util.Locale.ROOT, "%-10s %-9s tgt=%.2f %-12s selDepth=%.0f±%.0f heldRecall=%.3f(worst %.3f) slaMiss=%.2f bestFixed=%d oracle=%.0f%n",
                        dt, e, target, mode, meanSel, stdSel, meanHeld, worstHeld, slaMiss, bestFixed, oracleMean);
                }
            }
        }
        // ---- segment (per-dataset) vs single GLOBAL depth across all datasets ----
        for (Est e : Est.values()) for (double target : TARGETS) {
            // global depth: smallest that meets target on the POOLED calibration queries of all datasets
            int global = globalAcrossDatasets(POS, e.ordinal(), target);
            for (String[] ds : datasets) { String dt = ds[0]; int[][][] pos = POS.get(dt);
                int local = calibrate(pos, e.ordinal(), all(NQ), target, "mean", new Random(SEED));
                double localRec = meanRecallAt(pos, e.ordinal(), all(NQ), local);
                double globalRec = meanRecallAt(pos, e.ordinal(), all(NQ), global);
                seg.add(String.format(java.util.Locale.ROOT, "%s,%.2f,%s,%d,%.4f,%d,%.4f,%b", e, target, dt, local, localRec, global, globalRec, globalRec>=target-1e-9));
            }
        }
        write("track2_autocal_summary.csv", summary);
        write("track2_autocal_segments.csv", seg);
    }

    // smallest depth whose (mean or bootstrap-lower-bound) recall >= target(+margin conservative)
    private int calibrate(int[][][] pos, int ei, int[] qIdx, double target, String mode, Random rng) {
        for (int depth : DEPTHS) {
            if (mode.equals("mean")) { if (meanRecallAt(pos, ei, qIdx, depth) >= target) return depth; }
            else { double lb = bootstrapLower(pos, ei, qIdx, depth, rng); if (lb >= target + MARGIN) return depth; }
        }
        return POOL;
    }
    private double meanRecallAt(int[][][] pos, int ei, int[] qIdx, int depth) {
        double s = 0; for (int q : qIdx) { int hit = 0; for (int r : pos[ei][q]) if (r < depth) hit++; s += hit / (double) K; } return s / qIdx.length;
    }
    private double bootstrapLower(int[][][] pos, int ei, int[] qIdx, int depth, Random rng) {
        double[] means = new double[BOOT];
        for (int b = 0; b < BOOT; b++) { double s = 0; for (int i = 0; i < qIdx.length; i++) { int q = qIdx[rng.nextInt(qIdx.length)]; int hit = 0; for (int r : pos[ei][q]) if (r < depth) hit++; s += hit / (double) K; } means[b] = s / qIdx.length; }
        Arrays.sort(means); return means[(int) (0.05 * BOOT)];   // 5th percentile lower bound
    }
    private int bestFixedInHindsight(int[][][] pos, int ei, int[] qIdx, double target) { for (int d : DEPTHS) if (meanRecallAt(pos, ei, qIdx, d) >= target) return d; return POOL; }
    private double oracleMeanDepth(int[][][] pos, int ei, int[] qIdx) { double s = 0; for (int q : qIdx) { int mx = K; for (int r : pos[ei][q]) if (r < POOL && r + 1 > mx) mx = r + 1; s += mx; } return s / qIdx.length; }
    private int globalAcrossDatasets(java.util.Map<String,int[][][]> POS, int ei, double target) {
        for (int d : DEPTHS) { double s = 0; int c = 0; for (int[][][] pos : POS.values()) { for (int q = 0; q < NQ; q++) { int hit = 0; for (int r : pos[ei][q]) if (r < d) hit++; s += hit / (double) K; c++; } } if (s / c >= target) return d; } return POOL;
    }

    // ---- compute per-query true-top-k positions under each estimator ordering ----
    private int[][][] computePositions(String dt, int d) throws IOException {
        float[][] X, Q; int dim;
        if (dt.equals("sift_real")) { Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
            if (!Files.exists(repo.resolve(BASE))) return null; X = readFvecs(repo.resolve(BASE).toString(), N); Q = readFvecs(repo.resolve(QUERY).toString(), NQ); dim = X[0].length;
        } else { Random r = new Random(SEED); X = gen(dt, d, N, r); Q = gen(dt, d, NQ, new Random(SEED+5)); dim = d; }
        OneBitScalarQuantizer quant = new OneBitScalarQuantizer(false); final float[][] Xf = X;
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(N, false) { public float[] getVectorAtThePosition(int p){return Xf[p];} public void resetVectorValues(){} };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) quant.train(req);
        float[] thr = st.getMeanThresholds(), above = st.getAboveThresholdMeans(), below = st.getBelowThresholdMeans();
        byte[][] bits = new byte[N][]; float[][] xhat = new float[N][dim]; double[] normSq = new double[N];
        for (int j = 0; j < N; j++) { BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); quant.quantize(X[j].clone(), st, o); bits[j] = o.getQuantizedVector().clone();
            double ns = 0; for (int i = 0; i < dim; i++) { float xh = X[j][i] > thr[i] ? above[i] : below[i]; xhat[j][i] = xh; ns += (double) X[j][i]*X[j][i]; } normSq[j] = ns; }
        int[][][] positions = new int[Est.values().length][NQ][K];
        for (int qi = 0; qi < NQ; qi++) {
            float[] ex = new float[N]; for (int j = 0; j < N; j++) ex[j] = l2sq(Q[qi], X[j]);
            int[] trueTopK = topIndices(ex, K);
            float[] adc = new float[N], ham = new float[N], cor = new float[N];
            float[] qadc = Q[qi].clone(); quant.transformWithADC(qadc, st, SpaceType.L2); byte[] qb = quantizeQ(quant, Q[qi], st);
            double qn2 = dot(Q[qi], Q[qi]);
            for (int j = 0; j < N; j++) { adc[j] = KNNScoringUtil.l2SquaredADC(qadc, bits[j]); ham[j] = hamming(qb, bits[j]); cor[j] = (float) (qn2 + normSq[j] - 2.0*dot(Q[qi], xhat[j])); }
            positions[Est.ADC.ordinal()][qi] = ranksOf(adc, trueTopK);
            positions[Est.HAMMING.ordinal()][qi] = ranksOf(ham, trueTopK);
            positions[Est.CORRECTED.ordinal()][qi] = ranksOf(cor, trueTopK);
        }
        return positions;
    }
    // rank (0-indexed) of each target id under ascending score order
    private static int[] ranksOf(float[] score, int[] targets) {
        int[] rank = new int[targets.length];
        for (int t = 0; t < targets.length; t++) { float sv = score[targets[t]]; int less = 0; for (int j = 0; j < score.length; j++) if (score[j] < sv || (score[j] == sv && j < targets[t])) less++; rank[t] = less; }
        return rank;
    }

    // ---- helpers ----
    private static int[] shuffle(int n, Random r){ int[] a=new int[n]; for(int i=0;i<n;i++)a[i]=i; for(int i=n-1;i>0;i--){int j=r.nextInt(i+1);int t=a[i];a[i]=a[j];a[j]=t;} return a; }
    private static int[] fold(int[] perm,int f,int folds){ List<Integer> l=new ArrayList<>(); for(int i=0;i<perm.length;i++) if(i%folds==f) l.add(perm[i]); return l.stream().mapToInt(Integer::intValue).toArray(); }
    private static int[] foldComplement(int[] perm,int f,int folds){ List<Integer> l=new ArrayList<>(); for(int i=0;i<perm.length;i++) if(i%folds!=f) l.add(perm[i]); return l.stream().mapToInt(Integer::intValue).toArray(); }
    private static int[] all(int n){ int[] a=new int[n]; for(int i=0;i<n;i++)a[i]=i; return a; }
    private static double mean(List<Integer> l){ double s=0; for(int x:l)s+=x; return s/l.size(); }
    private static double meanD(List<Double> l){ double s=0; for(double x:l)s+=x; return s/l.size(); }
    private static double std(List<Integer> l){ double m=mean(l),s=0; for(int x:l)s+=(x-m)*(x-m); return Math.sqrt(s/l.size()); }
    private static double min(List<Double> l){ double m=1e9; for(double x:l)m=Math.min(m,x); return m; }
    private static byte[] quantizeQ(OneBitScalarQuantizer q,float[] v,OneBitScalarQuantizationState st){ BinaryQuantizationOutput o=new BinaryQuantizationOutput(1); q.quantize(v.clone(),st,o); return o.getQuantizedVector().clone(); }
    private static int[] topIndices(float[] s,int m){ Integer[] idx=new Integer[s.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(s[x],s[y])); int[] o=new int[m]; for(int i=0;i<m;i++)o[i]=idx[i]; return o; }
    private static float l2sq(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static double dot(float[] a,float[] b){ double s=0; for(int i=0;i<a.length;i++) s+=(double)a[i]*b[i]; return s; }
    private static int hamming(byte[] a,byte[] b){ int s=0; for(int i=0;i<a.length;i++) s+=Integer.bitCount((a[i]^b[i])&0xFF); return s; }
    private void write(String name,List<String> rows) throws IOException { Path out=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(out); try(Writer w=Files.newBufferedWriter(out.resolve(name))){ for(String r:rows){ w.write(r); w.write("\n"); } } System.out.println("WROTE "+rows.size()+" -> "+out.resolve(name)); }
    private static float[][] gen(String kind,int d,int n,Random rng){ float[][] X=new float[n][d];
        if(kind.equals("isotropic")){ for(int i=0;i<n;i++) for(int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); }
        else if(kind.equals("clustered")){ int ncl=32; float[][] c=new float[ncl][d]; for(int a=0;a<ncl;a++) for(int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f; for(int i=0;i<n;i++){ int a=rng.nextInt(ncl); for(int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } }
        else throw new IllegalArgumentException(kind); return X; }
    private static float[][] readFvecs(String path,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++) v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
