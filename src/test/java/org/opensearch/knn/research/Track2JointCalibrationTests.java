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
 * Track 2, step 5 — JOINT (candidate_pool, rerank_depth) calibration under a total-cost model,
 * conservative held-out recall, safety-margin + calibration-sample-size sweeps, heterogeneous
 * segments, and explicit candidate-pool-ceiling / infeasibility handling. Benchmark only; the
 * per-query Bonferroni policy is untouched (baseline).
 *
 * total_cost(pool,depth,ranker) = CA[ranker]*pool + CE*depth  (raw pool/depth also reported).
 * Selection: smallest-cost (pool,depth) with depth<=pool, ceiling_cal(pool)>=target, and the
 * conservative lower bound of recall@depth on the calibration split >= target+margin. Status:
 * CALIBRATED / INCREASE_CANDIDATE_POOL / USE_BETTER_RANKER / SLA_UNACHIEVABLE_AT_MAX_POOL.
 *
 * NOTE (flat-scan honesty): recall@depth is exhaustive-ordering, so pool-independent for
 * depth<=pool; the cost-optimal pool therefore equals depth here, and the pool knob governs the
 * ceiling/feasibility. Pool>depth benefit needs graph-limited ef (native tier). We report this.
 */
public class Track2JointCalibrationTests extends KNNTestCase {

    private static final int N = 10000, NQ = 300, K = 10, FOLDS = 5, BOOT = 300;
    private static final int[] POOLS = { 100, 200, 300, 400, 600, 800, 1000 };
    private static final int[] DEPTHS = { 10, 20, 30, 50, 100, 150, 200, 300, 400, 600, 800, 1000 };
    private static final double CE = 10.0;                          // exact rerank cost / candidate
    private static final double[] CA = { 2.0, 1.0, 3.0 };           // approx cost/candidate: ADC, HAMMING, CORRECTED
    private static final long SEED = 41L;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";
    enum Est { ADC, HAMMING, CORRECTED }

    public void testJointCalibration() throws IOException {
        List<String> agg = new ArrayList<>();
        agg.add("sweep,dataset,ranker,target,margin,cal_size,mode,sel_pool_mean,sel_pool_std,sel_depth_mean,sel_depth_std,"
            + "mean_heldout_recall,worst_heldout_recall,sla_miss_rate,mean_total_cost,mean_raw_pool,mean_raw_depth,"
            + "pool_ceiling,best_fixed_depth,oracle_mean_depth,savings_vs_bestfixed,savings_vs_fullpool,status");
        List<String> segs = new ArrayList<>();
        segs.add("ranker,target,margin,segment,seg_pool,seg_depth,seg_cost,seg_heldout_recall,global_pool,global_depth,global_cost_on_seg,global_meets");

        String[][] datasets = { {"sift_real","128"}, {"isotropic","768"}, {"clustered","128"} };
        java.util.Map<String,int[][][]> POS = new java.util.LinkedHashMap<>();
        for (String[] ds : datasets) POS.put(ds[0], computePositions(ds[0], Integer.parseInt(ds[1])));

        // ---- main grid: target x ranker x dataset (margin 0.01, cal 100, conservative) ----
        for (double target : new double[]{0.90,0.95,0.99})
            for (Est e : Est.values())
                for (String[] ds : datasets)
                    agg.add(run("main", ds[0], POS.get(ds[0]), e, target, 0.01, 100, "conservative"));

        // ---- safety-margin sweep (target 0.95, ADC+Corrected, sift+isotropic) ----
        for (double margin : new double[]{0.0,0.005,0.01,0.02})
            for (Est e : new Est[]{Est.ADC, Est.CORRECTED})
                for (String dt : new String[]{"sift_real","isotropic"})
                    agg.add(run("margin", dt, POS.get(dt), e, 0.95, margin, 100, "conservative"));

        // ---- calibration-sample-size sweep (target 0.95, margin 0.01, Corrected, sift+isotropic) ----
        for (int cs : new int[]{25,50,100,250,500})
            for (String dt : new String[]{"sift_real","isotropic"})
                agg.add(run("calsize", dt, POS.get(dt), Est.CORRECTED, 0.95, 0.01, cs, "conservative"));

        // ---- per-segment vs global (heterogeneous = 3 datasets as segments), target 0.95 ----
        for (Est e : Est.values()) {
            double target = 0.95, margin = 0.01;
            int[] glob = selectGlobalAcross(POS, e.ordinal(), target, margin);   // {pool,depth}
            for (String[] ds : datasets) { String dt = ds[0]; int[][][] pos = POS.get(dt);
                int[] loc = (int[]) selectConfig(pos, e.ordinal(), all(NQ), all(NQ), target, margin)[0]; // {pool,depth}
                double locCost = CA[e.ordinal()]*loc[0] + CE*loc[1], locRec = meanRecallAt(pos, e.ordinal(), all(NQ), loc[1]);
                double gRecOnSeg = meanRecallAt(pos, e.ordinal(), all(NQ), glob[1]); double gCost = CA[e.ordinal()]*glob[0] + CE*glob[1];
                segs.add(String.format(java.util.Locale.ROOT,"%s,%.2f,%.3f,%s,%d,%d,%.0f,%.4f,%d,%d,%.0f,%b",
                    e,target,margin,dt,loc[0],loc[1],locCost,locRec,glob[0],glob[1],gCost,gRecOnSeg>=target-1e-9));
            }
        }
        write("track2_joint_summary.csv", agg);
        write("track2_joint_segments.csv", segs);
    }

    // one aggregated row over FOLDS
    private String run(String sweep, String dt, int[][][] pos, Est e, double target, double margin, int calSize, String mode) {
        int ei = e.ordinal(); Random rng = new Random(SEED + calSize + (long)(target*100));
        List<Integer> selP = new ArrayList<>(), selD = new ArrayList<>(); List<Double> held = new ArrayList<>(); List<Double> costs = new ArrayList<>();
        String status = "CALIBRATED"; int nMiss = 0;
        int[] perm = shuffle(NQ, rng);
        for (int f = 0; f < FOLDS; f++) {
            int[] evalIdx = fold(perm, f, FOLDS);
            int[] calPool = foldComplement(perm, f, FOLDS);
            int[] calIdx = Arrays.copyOf(calPool, Math.min(calSize, calPool.length));
            Object[] r = selectConfig(pos, ei, calIdx, evalIdx, target, margin);
            int[] cfg = (int[]) r[0]; status = (String) r[1];
            double rec = meanRecallAt(pos, ei, evalIdx, cfg[1]);
            selP.add(cfg[0]); selD.add(cfg[1]); held.add(rec); costs.add(CA[ei]*cfg[0] + CE*cfg[1]);
            if (rec < target) nMiss++;
        }
        double ceiling = meanRecallAt(pos, ei, all(NQ), POOLS[POOLS.length-1]);
        int bestFixed = bestFixedHindsight(pos, ei, all(NQ), target);
        double oracle = oracleMeanDepth(pos, ei, all(NQ));
        double meanCost = meanD(costs), fullPoolCost = CA[ei]*1000 + CE*1000;
        double bestFixedCost = CA[ei]*bestFixed + CE*bestFixed;
        String row = String.format(java.util.Locale.ROOT,"%s,%s,%s,%.2f,%.3f,%d,%s,%.0f,%.0f,%.0f,%.0f,%.4f,%.4f,%.2f,%.0f,%.0f,%.0f,%.4f,%d,%.0f,%.3f,%.3f,%s",
            sweep,dt,e,target,margin,calSize,mode, mean(selP),std(selP),mean(selD),std(selD),
            meanD(held),min(held),nMiss/(double)FOLDS,meanCost,mean(selP),mean(selD),ceiling,bestFixed,oracle,
            1.0-meanCost/Math.max(1,bestFixedCost), 1.0-meanCost/fullPoolCost, status);
        System.out.printf(java.util.Locale.ROOT,"%-7s %-10s %-9s t=%.2f m=%.3f cal=%3d | pool=%.0f depth=%.0f held=%.3f(worst %.3f) slaMiss=%.2f cost=%.0f ceil=%.3f %s%n",
            sweep,dt,e,target,margin,calSize,mean(selP),mean(selD),meanD(held),min(held),nMiss/(double)FOLDS,meanCost,ceiling,status);
        return row;
    }

    // choose cheapest (pool,depth) meeting conservative recall + ceiling; return {cfg[pool,depth], status}
    private Object[] selectConfig(int[][][] pos, int ei, int[] calIdx, int[] evalIdx, double target, double margin) {
        Random rng = new Random(SEED + 7);
        double bestCost = Double.POSITIVE_INFINITY; int bp = POOLS[POOLS.length-1], bd = DEPTHS[DEPTHS.length-1]; boolean feasible = false;
        for (int P : POOLS) {
            double ceil = meanRecallAt(pos, ei, calIdx, P);
            if (ceil < target) continue;                              // pool ceiling too low -> skip
            for (int d : DEPTHS) { if (d > P) break;
                double lb = bootstrapLower(pos, ei, calIdx, d, rng);
                if (lb >= target + margin) { double c = CA[ei]*P + CE*d; if (c < bestCost) { bestCost=c; bp=P; bd=d; feasible=true; } break; }
            }
        }
        String status;
        if (feasible) status = "CALIBRATED";
        else { double ceilMax = meanRecallAt(pos, ei, calIdx, POOLS[POOLS.length-1]);
            status = ceilMax < target ? "SLA_UNACHIEVABLE_AT_MAX_POOL" : "INCREASE_CANDIDATE_POOL"; }
        return new Object[]{ new int[]{bp,bd}, status };
    }
    private int[] selectGlobalAcross(java.util.Map<String,int[][][]> POS, int ei, double target, double margin) {
        Random rng = new Random(SEED + 9);
        for (int P : POOLS) for (int d : DEPTHS) { if (d > P) break;
            // global must satisfy every segment's calibration (min over segments of lower bound)
            double minLb = 1.0, minCeil = 1.0;
            for (int[][][] pos : POS.values()) { minLb = Math.min(minLb, bootstrapLower(pos, ei, all(NQ), d, rng)); minCeil = Math.min(minCeil, meanRecallAt(pos, ei, all(NQ), P)); }
            if (minCeil >= target && minLb >= target + margin) return new int[]{P,d};
        }
        return new int[]{POOLS[POOLS.length-1], DEPTHS[DEPTHS.length-1]};
    }

    private double meanRecallAt(int[][][] pos, int ei, int[] qIdx, int depth) { double s=0; for(int q:qIdx){ int hit=0; for(int r:pos[ei][q]) if(r<depth) hit++; s+=hit/(double)K; } return s/qIdx.length; }
    private double bootstrapLower(int[][][] pos, int ei, int[] qIdx, int depth, Random rng) { double[] m=new double[BOOT]; for(int b=0;b<BOOT;b++){ double s=0; for(int i=0;i<qIdx.length;i++){ int q=qIdx[rng.nextInt(qIdx.length)]; int hit=0; for(int r:pos[ei][q]) if(r<depth) hit++; s+=hit/(double)K; } m[b]=s/qIdx.length; } Arrays.sort(m); return m[(int)(0.05*BOOT)]; }
    private int bestFixedHindsight(int[][][] pos,int ei,int[] q,double target){ for(int d:DEPTHS) if(meanRecallAt(pos,ei,q,d)>=target) return d; return DEPTHS[DEPTHS.length-1]; }
    private double oracleMeanDepth(int[][][] pos,int ei,int[] q){ double s=0; for(int qi:q){ int mx=K; for(int r:pos[ei][qi]) if(r<1000 && r+1>mx) mx=r+1; s+=mx; } return s/q.length; }

    // ---- positions of true-top-k under each ranker (reused approach) ----
    private int[][][] computePositions(String dt, int d) throws IOException {
        float[][] X, Q; int dim;
        if (dt.equals("sift_real")) { Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
            X = readFvecs(repo.resolve(BASE).toString(), N); Q = readFvecs(repo.resolve(QUERY).toString(), NQ); dim = X[0].length;
        } else { Random r = new Random(SEED); X = gen(dt, d, N, r); Q = gen(dt, d, NQ, new Random(SEED+5)); dim = d; }
        OneBitScalarQuantizer quant = new OneBitScalarQuantizer(false); final float[][] Xf = X;
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(N, false){ public float[] getVectorAtThePosition(int p){return Xf[p];} public void resetVectorValues(){} };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) quant.train(req);
        float[] thr=st.getMeanThresholds(), above=st.getAboveThresholdMeans(), below=st.getBelowThresholdMeans();
        byte[][] bits=new byte[N][]; float[][] xhat=new float[N][dim]; double[] normSq=new double[N];
        for(int j=0;j<N;j++){ BinaryQuantizationOutput o=new BinaryQuantizationOutput(1); quant.quantize(X[j].clone(),st,o); bits[j]=o.getQuantizedVector().clone(); double ns=0; for(int i=0;i<dim;i++){ float xh=X[j][i]>thr[i]?above[i]:below[i]; xhat[j][i]=xh; ns+=(double)X[j][i]*X[j][i]; } normSq[j]=ns; }
        int[][][] positions=new int[Est.values().length][NQ][K];
        for(int qi=0;qi<NQ;qi++){ float[] ex=new float[N]; for(int j=0;j<N;j++) ex[j]=l2sq(Q[qi],X[j]); int[] tk=topIndices(ex,K);
            float[] adc=new float[N],ham=new float[N],cor=new float[N]; float[] qadc=Q[qi].clone(); quant.transformWithADC(qadc,st,SpaceType.L2); byte[] qb=quantizeQ(quant,Q[qi],st); double qn2=dot(Q[qi],Q[qi]);
            for(int j=0;j<N;j++){ adc[j]=KNNScoringUtil.l2SquaredADC(qadc,bits[j]); ham[j]=hamming(qb,bits[j]); cor[j]=(float)(qn2+normSq[j]-2.0*dot(Q[qi],xhat[j])); }
            positions[0][qi]=ranksOf(adc,tk); positions[1][qi]=ranksOf(ham,tk); positions[2][qi]=ranksOf(cor,tk); }
        return positions;
    }
    private static int[] ranksOf(float[] s,int[] tg){ int[] rk=new int[tg.length]; for(int t=0;t<tg.length;t++){ float sv=s[tg[t]]; int less=0; for(int j=0;j<s.length;j++) if(s[j]<sv||(s[j]==sv&&j<tg[t])) less++; rk[t]=less; } return rk; }

    private static int[] shuffle(int n,Random r){ int[] a=new int[n]; for(int i=0;i<n;i++)a[i]=i; for(int i=n-1;i>0;i--){int j=r.nextInt(i+1);int t=a[i];a[i]=a[j];a[j]=t;} return a; }
    private static int[] fold(int[] p,int f,int folds){ List<Integer> l=new ArrayList<>(); for(int i=0;i<p.length;i++) if(i%folds==f) l.add(p[i]); return l.stream().mapToInt(Integer::intValue).toArray(); }
    private static int[] foldComplement(int[] p,int f,int folds){ List<Integer> l=new ArrayList<>(); for(int i=0;i<p.length;i++) if(i%folds!=f) l.add(p[i]); return l.stream().mapToInt(Integer::intValue).toArray(); }
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
