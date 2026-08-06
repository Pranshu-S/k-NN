/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.research;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
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
 * Track 2, step 6 — NATIVE (real Lucene HNSW graph) two-knob benchmark: graph-search effort
 * (ef_search) x exact rerank depth, per segment. NOT a flat scan — a genuine HNSW graph with real
 * traversal, so ef_search and rerank_depth are finally SEPARATE knobs (candidate quality/ceiling
 * depends on ef). Graph is built once per segment on 1-bit codes (Hamming, the Faiss binary-graph
 * analogue); the three rankers SEARCH the same graph with different query scorers (ADC / Hamming /
 * Corrected-8-byte), exactly like Faiss ADC-on-binary-graph — same graph settings for a fair
 * ranker comparison. Work metrics (nodes_visited, distance_comps, exact_reranks) are noise-free;
 * latency is WARM/indicative only (this sandbox cannot pin CPUs or evict page cache -> cold-cache
 * and production p99 are deferred to real hardware, per the task's noise-control requirements).
 *
 * Reuses the real OneBitScalarQuantizer / transformWithADC / KNNScoringUtil and the step-3 8-byte
 * corrected metadata (||x||^2, ||x-xhat||). Conservative per-segment calibration of (ranker, ef,
 * rerank_depth) vs a global config and OpenSearch-style fixed baselines.
 */
public class Track2NativeHnswTests extends KNNTestCase {

    private static final int N = 10000, NQ = 150, K = 10, M = 16, BEAM = 100;
    private static final int[] EFS = { 20, 50, 100, 200, 500 };
    private static final int[] DEPTHS = { 10, 20, 30, 50, 100, 150, 200, 300, 500 };
    private static final double TARGET = 0.95, MARGIN = 0.005;
    private static final double C_GRAPH = 1.0, C_EXACT = 10.0;   // work-cost weights (approx hop vs fp32 rerank)
    private static final long SEED = 51L;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";
    enum Est { ADC, HAMMING, CORRECTED }

    public void testNativeHnsw() throws IOException {
        List<String> cfg = new ArrayList<>();
        cfg.add("segment,ranker,ef_search,cand_ceiling,mean_nodes_visited,mean_dist_comps,recall@50rerank,"
            + "sel_rerank_depth,heldout_recall,warm_p50_us,warm_p95_us,warm_p99_us,status");
        List<String> sel = new ArrayList<>();
        sel.add("policy,segment,sel_ranker,sel_ef,sel_rerank_depth,heldout_recall,work_cost,warm_p50_us,warm_p95_us,warm_p99_us,savings_vs_osdefault,gap_to_hindsight,status");

        String[][] segments = { {"sift_real","128"}, {"isotropic","768"}, {"clustered","128"} };
        // per segment: build graph once, then evaluate rankers x ef
        java.util.Map<String,Seg> S = new java.util.LinkedHashMap<>();
        for (String[] sg : segments) S.put(sg[0], buildSegment(sg[0], Integer.parseInt(sg[1])));

        // ---- per (segment, ranker, ef): ceiling, work, recall curve, calibrate rerank depth ----
        for (String[] sg : segments) { String seg = sg[0]; Seg s = S.get(seg);
            for (Est e : Est.values()) for (int ef : EFS) {
                Res r = evalRankerEf(s, e, ef);
                // conservative rerank-depth calibration on CAL split
                int selDepth = calibrateDepth(r, s.nCal, TARGET, MARGIN);
                String status = r.ceiling < TARGET ? "SLA_UNACHIEVABLE_AT_MAX_CONFIG" : (selDepth < 0 ? "INCREASE_EF_SEARCH" : "CALIBRATED");
                int d = selDepth < 0 ? Math.min(ef, DEPTHS[DEPTHS.length-1]) : selDepth;
                double heldRec = recallAt(r, s.nCal, s.nCal + s.nEval, d);
                double[] lat = warmLatency(s, e, ef, d);
                cfg.add(String.format(java.util.Locale.ROOT, "%s,%s,%d,%.4f,%.1f,%.1f,%.4f,%d,%.4f,%.1f,%.1f,%.1f,%s",
                    seg, e, ef, r.ceiling, r.meanNodes, r.meanComps, recallAt(r, 0, s.nCal + s.nEval, 50), d, heldRec, lat[0], lat[1], lat[2], status));
            }
        }
        // ---- per-segment cheapest feasible config, vs global, vs OpenSearch-style fixed baseline ----
        for (String[] sg : segments) { String seg = sg[0]; Seg s = S.get(seg);
            Object[] best = pickCheapest(s, TARGET, MARGIN);   // {Est,ef,depth,workCost,status}
            // OS-ish default baseline: ADC ranker, ef=100, rerank=min-100
            Res base = evalRankerEf(s, Est.ADC, 100);
            int baseDepth = 100; double baseRec = recallAt(base, s.nCal, s.nCal + s.nEval, baseDepth);
            double baseCost = C_GRAPH*base.meanNodes + C_EXACT*baseDepth;
            double[] baseLat = warmLatency(s, Est.ADC, 100, baseDepth);
            if (best != null) { Est be=(Est)best[0]; int bef=(int)best[1], bd=(int)best[2]; double bcost=(double)best[3];
                Res br = evalRankerEf(s, be, bef); double bRec = recallAt(br, s.nCal, s.nCal + s.nEval, bd); double[] bLat = warmLatency(s, be, bef, bd);
                sel.add(String.format(java.util.Locale.ROOT,"per_segment,%s,%s,%d,%d,%.4f,%.0f,%.1f,%.1f,%.1f,%.3f,%s,%s",
                    seg, be, bef, bd, bRec, bcost, bLat[0], bLat[1], bLat[2], 1.0-bcost/Math.max(1,baseCost), "n/a", best[4]));
            } else sel.add(String.format(java.util.Locale.ROOT,"per_segment,%s,NONE,0,0,%.4f,0,0,0,0,0,%s", seg, base.ceiling, "SLA_UNACHIEVABLE"));
            sel.add(String.format(java.util.Locale.ROOT,"os_default(ADC ef100 rr100),%s,ADC,100,100,%.4f,%.0f,%.1f,%.1f,%.1f,0,n/a,%s",
                seg, baseRec, baseCost, baseLat[0], baseLat[1], baseLat[2], baseRec>=TARGET?"MEETS":"MISSES_SLA"));
        }
        write("track2_native_configs.csv", cfg);
        write("track2_native_selected.csv", sel);
    }

    // cheapest (ranker, ef, depth) meeting conservative SLA, min work-cost
    private Object[] pickCheapest(Seg s, double target, double margin) {
        Object[] best = null; double bestCost = Double.POSITIVE_INFINITY;
        for (Est e : Est.values()) for (int ef : EFS) {
            Res r = evalRankerEf(s, e, ef); if (r.ceiling < target) continue;
            int d = calibrateDepth(r, s.nCal, target, margin); if (d < 0) continue;
            double cost = C_GRAPH*r.meanNodes + C_EXACT*d;
            if (cost < bestCost) { bestCost = cost; best = new Object[]{e, ef, d, cost, "CALIBRATED"}; }
        }
        return best;
    }
    private int calibrateDepth(Res r, int nCal, double target, double margin) {
        for (int d : DEPTHS) { if (d > r.candLen) break; double lb = bootLower(r, 0, nCal, d); if (lb >= target + margin) return d; }
        return -1;
    }
    private double recallAt(Res r, int from, int to, int depth) { double s=0; int c=0; for (int q=from;q<to;q++){ int hit=0; for (int p: r.pos[q]) if (p>=0 && p<depth) hit++; s+=hit/(double)K; c++; } return c==0?0:s/c; }
    private double bootLower(Res r, int from, int to, int depth) { Random rng=new Random(SEED+depth); int n=to-from; double[] m=new double[200];
        for (int b=0;b<200;b++){ double s=0; for (int i=0;i<n;i++){ int q=from+rng.nextInt(n); int hit=0; for (int p:r.pos[q]) if(p>=0&&p<depth) hit++; s+=hit/(double)K; } m[b]=s/n; } Arrays.sort(m); return m[10]; }

    // ---- evaluate one ranker at one ef over all queries: candidate positions of true-topk + work ----
    static final class Res { int[][] pos; double ceiling, meanNodes, meanComps; int candLen; }
    private Res evalRankerEf(Seg s, Est e, int ef) {
        Res r = new Res(); r.pos = new int[s.nCal + s.nEval][K]; double nodes=0, comps=0; int totalQ = s.nCal + s.nEval; r.candLen = ef;
        for (int qi = 0; qi < totalQ; qi++) {
            int[] cand; long[] work = new long[2];
            try { cand = search(s, e, qi, ef, work); } catch (IOException ex) { throw new RuntimeException(ex); }
            nodes += work[0]; comps += work[1];
            // positions of the k true neighbours within the candidate list (approx order)
            java.util.HashMap<Integer,Integer> rankOf = new java.util.HashMap<>();
            for (int i = 0; i < cand.length; i++) rankOf.put(cand[i], i);
            for (int t = 0; t < K; t++) r.pos[qi][t] = rankOf.getOrDefault(s.trueTopK[qi][t], -1);
        }
        r.meanNodes = nodes/totalQ; r.meanComps = comps/totalQ;
        r.ceiling = recallAt(r, 0, totalQ, ef);
        return r;
    }

    // real Lucene HNSW search with the ranker's query scorer; returns candidate ordinals (approx order)
    private int[] search(Seg s, Est e, int qi, int ef, long[] work) throws IOException {
        final int[] comps = {0};
        RandomVectorScorer qs = queryScorer(s, e, qi, comps);
        TopKnnCollector col = new TopKnnCollector(ef, Integer.MAX_VALUE);
        HnswGraphSearcher.search(qs, col, s.graph, null);
        work[1] = comps[0];
        work[0] = col.visitedCount();
        TopDocs td = col.topDocs(); ScoreDoc[] sd = td.scoreDocs;
        int[] out = new int[sd.length]; for (int i = 0; i < sd.length; i++) out[i] = sd[i].doc; return out;
    }
    private RandomVectorScorer queryScorer(Seg s, Est e, int qi, int[] comps) {
        return new RandomVectorScorer() {
            public int maxOrd() { return s.N; }
            public float score(int ord) { comps[0]++;
                if (e == Est.HAMMING) return -ham(s.qbits[qi], s.codes[ord]);
                if (e == Est.ADC) return -KNNScoringUtil.l2SquaredADC(s.qadc[qi], s.codes[ord]);
                double raw = s.qn2[qi] + s.normSq[ord] - 2.0*dot(s.Q[qi], s.xhat[ord]); return (float) -raw; // corrected
            }
        };
    }

    // warm latency (us) p50/p95/p99 of search(ef)+rerank(depth) over eval queries, best of 3 reps
    private double[] warmLatency(Seg s, Est e, int ef, int depth) {
        int from = s.nCal, to = s.nCal + s.nEval; double[] best = null;
        try {
            for (int rep = 0; rep < 4; rep++) { double[] samp = new double[to-from]; int idx=0;
                for (int q = from; q < to; q++) { long t0 = System.nanoTime();
                    long[] w = new long[2]; int[] cand = search(s, e, q, ef, w);
                    int dd = Math.min(depth, cand.length); float[] ex = new float[dd];
                    for (int i = 0; i < dd; i++) ex[i] = KNNScoringUtil.l2Squared(s.Q[q], s.X[cand[i]]);   // real fp32 rerank
                    Integer[] o = new Integer[dd]; for (int i=0;i<dd;i++) o[i]=i; Arrays.sort(o,(a,b)->Float.compare(ex[a],ex[b]));
                    long t1 = System.nanoTime(); samp[idx++] = (t1-t0)/1000.0;
                }
                Arrays.sort(samp); double[] cur = { samp[samp.length/2], samp[(int)(0.95*samp.length)], samp[(int)(0.99*samp.length)] };
                if (best == null || cur[0] < best[0]) best = cur;
            }
        } catch (IOException ex) { throw new RuntimeException(ex); }
        return best;
    }

    // ---- build one segment: quantize, metadata, and a real Lucene HNSW graph on 1-bit codes (Hamming) ----
    static final class Seg { int N, dim, nCal, nEval; float[][] X, Q; byte[][] codes, qbits; float[][] qadc, xhat; double[] normSq, qn2; int[][] trueTopK; OnHeapHnswGraph graph; }
    private Seg buildSegment(String dt, int d) throws IOException {
        Seg s = new Seg(); s.nCal = 50; s.nEval = 100;
        if (dt.equals("sift_real")) { Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
            s.X = readFvecs(repo.resolve(BASE).toString(), N); s.Q = readFvecs(repo.resolve(QUERY).toString(), s.nCal+s.nEval); s.dim = s.X[0].length;
        } else { Random r=new Random(SEED); s.X = gen(dt,d,N,r); s.Q = gen(dt,d,s.nCal+s.nEval,new Random(SEED+5)); s.dim = d; }
        s.N = s.X.length; int dim = s.dim;
        OneBitScalarQuantizer quant = new OneBitScalarQuantizer(false); final float[][] Xf = s.X;
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(s.N, false){ public float[] getVectorAtThePosition(int p){return Xf[p];} public void resetVectorValues(){} };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) quant.train(req);
        float[] thr=st.getMeanThresholds(), ab=st.getAboveThresholdMeans(), be=st.getBelowThresholdMeans();
        s.codes = new byte[s.N][]; s.xhat = new float[s.N][dim]; s.normSq = new double[s.N];
        for (int j=0;j<s.N;j++){ BinaryQuantizationOutput o=new BinaryQuantizationOutput(1); quant.quantize(s.X[j].clone(),st,o); s.codes[j]=o.getQuantizedVector().clone();
            double ns=0; for(int i=0;i<dim;i++){ float xh=s.X[j][i]>thr[i]?ab[i]:be[i]; s.xhat[j][i]=xh; ns+=(double)s.X[j][i]*s.X[j][i]; } s.normSq[j]=ns; }
        int nq = s.nCal+s.nEval; s.qbits=new byte[nq][]; s.qadc=new float[nq][]; s.qn2=new double[nq]; s.trueTopK=new int[nq][K];
        for (int qi=0;qi<nq;qi++){ BinaryQuantizationOutput o=new BinaryQuantizationOutput(1); quant.quantize(s.Q[qi].clone(),st,o); s.qbits[qi]=o.getQuantizedVector().clone();
            float[] qa=s.Q[qi].clone(); quant.transformWithADC(qa,st,SpaceType.L2); s.qadc[qi]=qa; s.qn2[qi]=dot(s.Q[qi],s.Q[qi]);
            float[] ex=new float[s.N]; for(int j=0;j<s.N;j++) ex[j]=l2sq(s.Q[qi],s.X[j]); s.trueTopK[qi]=topIndices(ex,K); }
        // build the real HNSW graph on binary codes with Hamming similarity
        final byte[][] codes = s.codes; final int NN = s.N;
        RandomVectorScorerSupplier supplier = new RandomVectorScorerSupplier() {
            public UpdateableRandomVectorScorer scorer() { return new UpdateableRandomVectorScorer() {
                int cur = 0; public int maxOrd(){ return NN; } public void setScoringOrdinal(int o){ cur=o; }
                public float score(int j){ return -ham(codes[cur], codes[j]); } }; }
            public RandomVectorScorerSupplier copy() { return this; }
        };
        HnswGraphBuilder builder = HnswGraphBuilder.create(supplier, M, BEAM, SEED);
        s.graph = builder.build(s.N);
        System.out.printf(java.util.Locale.ROOT, "[native] segment %s built: N=%d dim=%d graph(M=%d,beam=%d)%n", dt, s.N, dim, M, BEAM);
        return s;
    }

    private static float ham(byte[] a, byte[] b){ int s=0; for(int i=0;i<a.length;i++) s+=Integer.bitCount((a[i]^b[i])&0xFF); return s; }
    private static float l2sq(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static double dot(float[] a,float[] b){ double s=0; for(int i=0;i<a.length;i++) s+=(double)a[i]*b[i]; return s; }
    private static int[] topIndices(float[] sc,int m){ Integer[] idx=new Integer[sc.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(sc[x],sc[y])); int[] o=new int[m]; for(int i=0;i<m;i++)o[i]=idx[i]; return o; }
    private void write(String name,List<String> rows) throws IOException { Path out=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(out); try(Writer w=Files.newBufferedWriter(out.resolve(name))){ for(String r:rows){ w.write(r); w.write("\n"); } } System.out.println("WROTE "+rows.size()+" -> "+out.resolve(name)); }
    private static float[][] gen(String kind,int d,int n,Random rng){ float[][] X=new float[n][d];
        if(kind.equals("isotropic")){ for(int i=0;i<n;i++) for(int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); }
        else if(kind.equals("clustered")){ int ncl=32; float[][] c=new float[ncl][d]; for(int a=0;a<ncl;a++) for(int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f; for(int i=0;i<n;i++){ int a=rng.nextInt(ncl); for(int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } }
        else throw new IllegalArgumentException(kind); return X; }
    private static float[][] readFvecs(String path,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++) v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
