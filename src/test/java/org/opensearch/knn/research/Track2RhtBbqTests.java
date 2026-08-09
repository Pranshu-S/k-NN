/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.research;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.hnsw.HnswGraphBuilder;
import org.apache.lucene.util.hnsw.HnswGraphSearcher;
import org.apache.lucene.util.hnsw.NeighborArray;
import org.apache.lucene.util.hnsw.OnHeapHnswGraph;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.UpdateableRandomVectorScorer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer.QuantizationResult;
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
 * Track 2, step 14 — Does random-Hadamard (RHT) preconditioning improve Lucene's REAL production 1-bit
 * OSQ/BBQ candidate recall and cut the HNSW ef needed for a target?
 *
 * FAITHFUL BBQ (not dequantize-L2): Lucene {@link OptimizedScalarQuantizer}.scalarQuantize (1-bit doc +
 * 4-bit query) -> packAsBinary / transposeHalfByte -> {@link VectorUtil#int4BitDotProduct} -> the exact
 * Lucene104 corrective-term Euclidean formula. Build uses Lucene's asymmetric geometry (4-bit-query-of-A
 * vs 1-bit-doc-of-B); traversal uses 4-bit-query-of-query vs 1-bit-doc. RHT+BBQ = identical pipeline on
 * R(x)=H(Dx)/sqrt(block), same seed for index+query; ONLY the transform differs. Candidate recall@10
 * BEFORE rerank is primary. Exact FP32 ground truth. Datasets: fashion-mnist-784 (real), iid-784 control.
 */
public class Track2RhtBbqTests extends KNNTestCase {

    private static final int K = 10, M = 16, BEAM = 100;
    private static final int NQ = 300, NEVAL = 200;          // last NEVAL are held-out eval queries
    private static final int[] EFS = { 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000 };
    private static final long SEED = 71L, RHT_SEED = 1234567L;
    private static final String REAL = "research/acorn/data/real/";

    // ==================== correctness assertions (task 16) ====================
    public void testInvariants() {
        Random r = new Random(SEED);
        // orthogonality: ||Rx-Ry|| == ||x-y||
        for (int dim : new int[]{ 128, 256, 784, 960 }) {
            int[] blk = blockSplit(dim); float[] sign = signs(sum(blk), RHT_SEED);
            double maxErr = 0;
            for (int t = 0; t < 40; t++) { float[] x = randVec(dim, r), y = randVec(dim, r);
                float[] rx = rht(x, blk, sign), ry = rht(y, blk, sign);
                maxErr = Math.max(maxErr, Math.abs(Math.sqrt(l2(x, y)) - Math.sqrt(l2(rx, ry))));
            }
            assertTrue("RHT orthogonality dim=" + dim + " err=" + maxErr, maxErr < 1e-3);
        }
        // determinism: same seed -> identical transform
        float[] x = randVec(784, r); int[] blk = blockSplit(784);
        assertTrue("RHT determinism", Arrays.equals(rht(x, blk, signs(sum(blk), RHT_SEED)), rht(x, blk, signs(sum(blk), RHT_SEED))));
        // BBQ formula parity: two independent evaluations of the same (query,doc) match exactly
        float[][] V = randData(784, 500, new Random(SEED + 1)); float[] cen = mean(V);
        Bbq b = new Bbq(V, cen, false); float[] q = randVec(784, r);
        assertEquals("BBQ deterministic score", b.asymDist(q, 3), b.asymDist(q, 3), 0f);
        System.out.println("[step14] invariants PASS (orthogonality, determinism, BBQ parity)");
    }

    // ==================== Phase 1+2: flat fidelity + candidate recall, BBQ vs RHT+BBQ ====================
    public void testRhtBbq() throws IOException {
        Path dir = outDir();
        int start = countDone(dir);
        String h = "dataset,dim,N,rep,graph_mode,eff_bits,code_bytes,sidecar_bytes,total_bytes,rotation_global_bytes,"
            + "flat_top10_fidelity,build_ms,rotation_ms,"
            + String.join(",", ef("recall_ef", EFS)) + ",ef90,ef95,ef97,ef99,mae,rmse,p95_abs_err";
        if (start == 0) writeFresh(dir, "step14_rht_bbq.csv", h);
        Object[][] sets = { {"fmnist784", 784, 20000, "fmnist"}, {"iid784", 784, 15000, "iid"} };
        System.out.printf(java.util.Locale.ROOT, "[step14] start=%d%n", start);
        for (int si = start; si < sets.length; si++) {
            Object[] s = sets[si]; String name = (String) s[0]; int dim = (int) s[1], N = (int) s[2]; String src = (String) s[3];
            float[][] X = load(src, dim, N, false), Q = load(src, dim, NQ, true);
            int[][] gt = groundTruth(X, Q);
            List<String> rows = new ArrayList<>();
            // representations (each on native representation graph). fp32 reference first.
            rows.add(eval(name, dim, N, "fp32", "native", new Fp32(X), X, Q, gt, 0, 0, 0));
            // A: production BBQ
            long r0 = System.nanoTime(); float[] cen = mean(X); Bbq bbq = new Bbq(X, cen, false); long bbqEnc = (System.nanoTime() - r0) / 1_000_000;
            rows.add(eval(name, dim, N, "bbq_1bit", "native", bbq, X, Q, gt, bbq.codeBytes, bbq.sidecar, 0));
            // B: RHT -> BBQ (rotate X and queries with same transform)
            int[] blk = blockSplit(dim); float[] sign = signs(sum(blk), RHT_SEED);
            long rt0 = System.nanoTime(); float[][] Xr = rhtAll(X, blk, sign); float[][] Qr = rhtAll(Q, blk, sign); long rhtMs = (System.nanoTime() - rt0) / 1_000_000;
            float[] cenR = mean(Xr); Bbq rbbq = new Bbq(Xr, cenR, false);
            int[][] gtR = gt;   // GT is rotation-invariant (orthogonal transform preserves L2 order) -> reuse
            rows.add(eval(name, dim, N, "rht_bbq_1bit", "native", rbbq, Xr, Qr, gtR, rbbq.codeBytes, rbbq.sidecar, blk.length * 4 + 8, rhtMs));
            // for context: old dequantize-L2 OSQ (to quantify how much BBQ was understated) + RaBitQ-inspired
            rows.add(eval(name, dim, N, "osq_dequant_l2", "native", new OsqDequant(X, cen, 1), X, Q, gt, (dim + 7) / 8, 8, 0));
            rows.add(eval(name, dim, N, "rabitq_insp", "native", new Rabitq(X, RHT_SEED), X, Q, gt, (dim + 7) / 8, 8, 0));
            append(dir, "step14_rht_bbq.csv", rows);
            System.out.printf(java.util.Locale.ROOT, "[step14] %s done%n", name);
        }
        System.out.println("[step14] DONE");
    }

    // ==================== Experiment 1: COMMON fp32 graph, searched by each scorer (isolates traversal) ====================
    public void testCommonGraph() throws IOException {
        Path dir = outDir(); writeFresh(dir, "step14_common_graph.csv",
            "dataset,rep,graph_mode," + String.join(",", ef("recall_ef", EFS)) + ",ef90,ef95,ef97");
        Object[][] sets = { {"fmnist784", 784, 20000, "fmnist"}, {"iid784", 784, 15000, "iid"} };
        for (Object[] s : sets) {
            String name = (String) s[0]; int dim = (int) s[1], N = (int) s[2]; String src = (String) s[3];
            float[][] X = load(src, dim, N, false), Q = load(src, dim, NQ, true); int[][] gt = groundTruth(X, Q);
            OnHeapHnswGraph g = build(new Fp32(X), N);   // ONE fp32 graph topology for all scorers
            int[] blk = blockSplit(dim); float[] sign = signs(sum(blk), RHT_SEED);
            float[][] Xr = rhtAll(X, blk, sign), Qr = rhtAll(Q, blk, sign); float[] cen = mean(X), cenR = mean(Xr);
            java.util.LinkedHashMap<String, Object[]> reps = new java.util.LinkedHashMap<>();  // scorer, its queries, its gt
            reps.put("fp32", new Object[]{ new Fp32(X), Q, gt });
            reps.put("bbq_1bit", new Object[]{ new Bbq(X, cen, false), Q, gt });
            reps.put("rht_bbq_1bit", new Object[]{ new Bbq(Xr, cenR, false), Qr, gt });   // same fp32 graph, rotated scorer
            reps.put("rabitq_insp", new Object[]{ new Rabitq(X, RHT_SEED), Q, gt });
            List<String> rows = new ArrayList<>();
            for (var e : reps.entrySet()) { Rep r = (Rep) e.getValue()[0]; float[][] qq = (float[][]) e.getValue()[1]; int[][] gg = (int[][]) e.getValue()[2];
                double[] rec = new double[EFS.length]; int[] efT = { -1, -1, -1 }; double[] tg = { 0.90, 0.95, 0.97 };
                for (int i = 0; i < EFS.length; i++) { rec[i] = candRecall(g, r, qq, gg, EFS[i]); for (int t = 0; t < 3; t++) if (efT[t] < 0 && rec[i] >= tg[t]) efT[t] = EFS[i]; }
                StringBuilder sb = new StringBuilder(name + "," + e.getKey() + ",common_fp32_graph");
                for (double v : rec) sb.append(String.format(java.util.Locale.ROOT, ",%.4f", v));
                sb.append(","+efT[0]+","+efT[1]+","+efT[2]); rows.add(sb.toString());
                System.out.printf(java.util.Locale.ROOT, "[step14-common] %s %-14s ef90=%d ef95=%d ef97=%d%n", name, e.getKey(), efT[0], efT[1], efT[2]);
            }
            append(dir, "step14_common_graph.csv", rows);
        }
    }

    private String eval(String ds, int dim, int N, String rep, String gmode, Rep r, float[][] X, float[][] Q, int[][] gt,
                        int codeB, int sideB, int rotGlobal) throws IOException { return eval(ds, dim, N, rep, gmode, r, X, Q, gt, codeB, sideB, rotGlobal, 0); }
    private String eval(String ds, int dim, int N, String rep, String gmode, Rep r, float[][] X, float[][] Q, int[][] gt,
                        int codeB, int sideB, int rotGlobal, long rhtMs) throws IOException {
        double flat = flatFidelity(r, X.length, Q, gt);            // Phase-2 flat top-10 fidelity (no graph)
        double[] err = scoreError(r, X, Q, gt);                     // MAE/RMSE/p95|err| on sampled pairs
        long b0 = System.nanoTime(); OnHeapHnswGraph g = build(r, N); long buildMs = (System.nanoTime() - b0) / 1_000_000;
        double[] rec = new double[EFS.length]; int[] efT = { -1, -1, -1, -1 }; double[] tgt = { 0.90, 0.95, 0.97, 0.99 };
        for (int e = 0; e < EFS.length; e++) { rec[e] = candRecall(g, r, Q, gt, EFS[e]);
            for (int t = 0; t < 4; t++) if (efT[t] < 0 && rec[e] >= tgt[t]) efT[t] = EFS[e]; }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(java.util.Locale.ROOT, "%s,%d,%d,%s,%s,%.3f,%d,%d,%d,%d,%.4f,%d,%d",
            ds, dim, N, rep, gmode, r.effBits(), codeB, sideB, codeB + sideB, rotGlobal, flat, buildMs, rhtMs));
        for (double v : rec) sb.append(String.format(java.util.Locale.ROOT, ",%.4f", v));
        sb.append(String.format(java.util.Locale.ROOT, ",%d,%d,%d,%d,%.4f,%.4f,%.4f", efT[0], efT[1], efT[2], efT[3], err[0], err[1], err[2]));
        System.out.printf(java.util.Locale.ROOT, "[step14]   %-16s flat=%.3f ef90=%d ef95=%d ef97=%d build=%dms%n", rep, flat, efT[0], efT[1], efT[2], buildMs);
        return sb.toString();
    }

    // ==================== representations ====================
    interface Rep { float symDist(int a, int b); float asymDist(float[] q, int b); double effBits(); int n(); }

    static final class Fp32 implements Rep { final float[][] X; Fp32(float[][] X){this.X=X;} public float symDist(int a,int b){return l2(X[a],X[b]);} public float asymDist(float[] q,int b){return l2(q,X[b]);} public double effBits(){return 32;} public int n(){return X.length;} }

    /** FAITHFUL Lucene production BBQ (1-bit doc + 4-bit query + corrective terms + int4BitDotProduct). */
    static final class Bbq implements Rep {
        final int n, dim, disc, packedLen; final float[] centroid;
        final byte[][] docPacked; final float[] dLo, dHi, dAdd; final int[] dSum;      // 1-bit doc side
        final byte[][] qTrans; final float[] qLo, qHi, qAdd; final int[] qSum;         // 4-bit query-of-doc (build)
        final OptimizedScalarQuantizer osq; int codeBytes, sidecar;
        Bbq(float[][] X, float[] centroid, boolean unusedRotate) {
            this.n = X.length; this.dim = X[0].length; this.centroid = centroid;
            this.disc = OptimizedScalarQuantizer.discretize(dim, 64); this.packedLen = disc / 8;
            this.osq = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
            docPacked = new byte[n][packedLen]; dLo = new float[n]; dHi = new float[n]; dAdd = new float[n]; dSum = new int[n];
            qTrans = new byte[n][4 * packedLen]; qLo = new float[n]; qHi = new float[n]; qAdd = new float[n]; qSum = new int[n];
            byte[] destD = new byte[disc], destQ = new byte[disc];
            for (int i = 0; i < n; i++) {
                QuantizationResult rd = osq.scalarQuantize(X[i].clone(), destD, (byte) 1, centroid);
                OptimizedScalarQuantizer.packAsBinary(destD, docPacked[i]); dLo[i]=rd.lowerInterval(); dHi[i]=rd.upperInterval(); dAdd[i]=rd.additionalCorrection(); dSum[i]=rd.quantizedComponentSum();
                QuantizationResult rq = osq.scalarQuantize(X[i].clone(), destQ, (byte) 4, centroid);
                OptimizedScalarQuantizer.transposeHalfByte(destQ, qTrans[i]); qLo[i]=rq.lowerInterval(); qHi[i]=rq.upperInterval(); qAdd[i]=rq.additionalCorrection(); qSum[i]=rq.quantizedComponentSum();
            }
            // persistent storage = 1-bit packed code + corrective terms (lo,hi,add : 3 floats + sum : 1 int) = 16B
            codeBytes = packedLen; sidecar = 16;
        }
        // core Lucene104 quantizedScore (EUCLIDEAN) -> squared L2 distance (>=0)
        private float score(byte[] qT, float qlo, float qhi, float qadd, int qsm, int b) {
            long qc = VectorUtil.int4BitDotProduct(qT, docPacked[b]);
            float scale = 1f, queryScale = 1f / 15f;   // doc 1-bit, query 4-bit
            float x1 = dSum[b], ax = dLo[b], lx = (dHi[b] - ax) * scale;
            float ay = qlo, ly = (qhi - ay) * queryScale, y1 = qsm;
            float dot = ax * ay * dim + ay * lx * x1 + ax * ly * y1 + lx * ly * (float) qc;
            float dist2 = qadd + dAdd[b] - 2f * dot; return Math.max(dist2, 0f);
        }
        public float symDist(int a, int b) { return score(qTrans[a], qLo[a], qHi[a], qAdd[a], qSum[a], b); }   // build: 4bit-query-of-a vs 1bit-doc-of-b
        public float asymDist(float[] q, int b) { byte[] dq = new byte[disc]; QuantizationResult rq = osq.scalarQuantize(q.clone(), dq, (byte) 4, centroid);
            byte[] t = new byte[4 * packedLen]; OptimizedScalarQuantizer.transposeHalfByte(dq, t); return score(t, rq.lowerInterval(), rq.upperInterval(), rq.additionalCorrection(), rq.quantizedComponentSum(), b); }
        public double effBits() { return (double) (codeBytes + sidecar) * 8 / dim; } public int n() { return n; }
    }

    /** old approach for contrast: OSQ interval optimization but dequantize -> plain L2 (fp32 query). */
    static final class OsqDequant implements Rep {
        final float[][] rec; final int dim, bits; OsqDequant(float[][] X, float[] cen, int bits){ this.dim=X[0].length; this.bits=bits;
            OptimizedScalarQuantizer q=new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN); rec=new float[X.length][]; byte[] dest=new byte[dim];
            for(int i=0;i<X.length;i++){ QuantizationResult r=q.scalarQuantize(X[i].clone(),dest,(byte)bits,cen); float[] o=new float[dim]; OptimizedScalarQuantizer.deQuantize(dest,o,(byte)bits,r.lowerInterval(),r.upperInterval(),cen); rec[i]=o; } }
        public float symDist(int a,int b){return l2(rec[a],rec[b]);} public float asymDist(float[] q,int b){return l2(q,rec[b]);} public double effBits(){return bits;} public int n(){return rec.length;} }

    /** RaBitQ-inspired: centroid-residual unit-normalize + structured rotation + sign code + norm/correction. */
    static final class Rabitq implements Rep {
        final int n, dim, Dt; final byte[][] bit; final float[] nx, invDot; final float[] cen; final int[] blk; final float[] sign; final float[][] rawA;
        Rabitq(float[][] X, long sd){ n=X.length; dim=X[0].length; blk=blockSplit(dim); Dt=sum(blk); sign=signs(Dt,sd+5); cen=mean(X); rawA=X;
            bit=new byte[n][Dt]; nx=new float[n]; invDot=new float[n]; float inv=1f/(float)Math.sqrt(Dt);
            for(int i=0;i<n;i++){ float[] r=new float[dim]; float nn=0; for(int j=0;j<dim;j++){r[j]=X[i][j]-cen[j];nn+=r[j]*r[j];} nn=(float)Math.sqrt(nn); nx[i]=nn;
                float[] u=nn>0?scale(r,1f/nn):r; float[] y=rht(u,blk,sign); float dot=0; for(int j=0;j<Dt;j++){ bit[i][j]=(byte)(y[j]>=0?1:0); dot+=Math.abs(y[j]); } dot*=inv; invDot[i]=dot>1e-6f?1f/dot:0f; } }
        private float[] qunit(float[] q){ float[] r=new float[dim]; float nn=0; for(int j=0;j<dim;j++){r[j]=q[j]-cen[j];nn+=r[j]*r[j];} nn=(float)Math.sqrt(nn); float[] u=nn>0?scale(r,1f/nn):r; float[] y=rht(u,blk,sign); float[] out=new float[Dt+1]; System.arraycopy(y,0,out,0,Dt); out[Dt]=nn; return out; }
        private float est(float[] yq,int b){ float dot=0; for(int j=0;j<Dt;j++) dot+= (bit[b][j]==1?yq[j]:-yq[j]); dot*=1f/(float)Math.sqrt(Dt); return dot*invDot[b]; }
        public float asymDist(float[] q,int b){ float[] qq=qunit(q); float nq=qq[Dt]; float e=est(qq,b); float dd=nq*nq+nx[b]*nx[b]-2*nq*nx[b]*e; return dd<0?0:dd; }
        public float symDist(int a,int b){ return asymDist(rawA[a], b); }
        public double effBits(){return 1.0;} public int n(){return n;} }

    // ==================== graph + metrics ====================
    private OnHeapHnswGraph build(Rep r, int N) throws IOException {

        RandomVectorScorerSupplier sup = new RandomVectorScorerSupplier() {
            public UpdateableRandomVectorScorer scorer() { return new UpdateableRandomVectorScorer() { int cur = 0; public int maxOrd() { return N; } public void setScoringOrdinal(int o) { cur = o; } public float score(int j) { return -r.symDist(cur, j); } }; }
            public RandomVectorScorerSupplier copy() { return this; } };
        return HnswGraphBuilder.create(sup, M, BEAM, SEED).build(N);
    }
    static final class GV extends HnswGraph { final OnHeapHnswGraph g; int[] cur=new int[0]; int sz=0,up=0; GV(OnHeapHnswGraph g){this.g=g;} public void seek(int l,int nd){NeighborArray na=g.getNeighbors(l,nd);cur=na.nodes();sz=na.size();up=0;} public int size(){return g.size();} public int nextNeighbor(){return up<sz?cur[up++]:DocIdSetIterator.NO_MORE_DOCS;} public int numLevels()throws IOException{return g.numLevels();} public int maxConn(){return g.maxConn();} public int entryNode()throws IOException{return g.entryNode();} public int neighborCount(){return sz;} public HnswGraph.NodesIterator getNodesOnLevel(int l)throws IOException{return g.getNodesOnLevel(l);} }
    private double candRecall(OnHeapHnswGraph g, Rep r, float[][] Q, int[][] gt, int ef) throws IOException {
        double s=0; int c=0; for(int qi=NQ-NEVAL; qi<NQ; qi++){ final float[] qv=Q[qi]; RandomVectorScorer qs=new RandomVectorScorer(){ public int maxOrd(){return r.n();} public float score(int ord){ return -r.asymDist(qv,ord); } };
            TopKnnCollector col=new TopKnnCollector(ef,Integer.MAX_VALUE); HnswGraphSearcher.search(qs,col,new GV(g),null); ScoreDoc[] sd=col.topDocs().scoreDocs;
            java.util.HashSet<Integer> cand=new java.util.HashSet<>(); for(ScoreDoc d:sd) cand.add(d.doc); int h=0; for(int t=0;t<K;t++) if(cand.contains(gt[qi][t])) h++; s+=h/(double)K; c++; } return c==0?0:s/c;
    }
    private double flatFidelity(Rep r, int N, float[][] Q, int[][] gt){ double s=0; int c=0; int probe=Math.min(NEVAL, 80);
        for(int qi=NQ-NEVAL; qi<NQ-NEVAL+probe; qi++){ float[] sc=new float[N]; for(int j=0;j<N;j++) sc[j]=r.asymDist(Q[qi],j); int[] top=topIdx(sc,K);
            java.util.HashSet<Integer> t=new java.util.HashSet<>(); for(int x:top)t.add(x); int h=0; for(int x:gt[qi]) if(t.contains(x))h++; s+=h/(double)K; c++; } return c==0?0:s/c; }
    private double[] scoreError(Rep r, float[][] X, float[][] Q, int[][] gt){ Random rng=new Random(SEED+9); List<Double> ae=new ArrayList<>(); double se=0; int cnt=0;
        for(int t=0;t<40;t++){ int qi=NQ-NEVAL+rng.nextInt(NEVAL); float[] q=Q[qi]; for(int p=0;p<25;p++){ int d=rng.nextInt(X.length); double approx=Math.sqrt(r.asymDist(q,d)); double exact=Math.sqrt(l2(q,X[d])); double e=approx-exact; ae.add(Math.abs(e)); se+=e*e; cnt++; } }
        java.util.Collections.sort(ae); double mae=0; for(double v:ae)mae+=v; mae/=ae.size(); double rmse=Math.sqrt(se/cnt); double p95=ae.get((int)(0.95*ae.size())); return new double[]{mae,rmse,p95}; }

    // ==================== RHT (block-Hadamard, dimension-preserving) ====================
    private static int[] blockSplit(int d){ List<Integer> b=new ArrayList<>(); int rem=d; while(rem>0){ int p=1; while(p*2<=rem)p<<=1; b.add(p); rem-=p; } int[] o=new int[b.size()]; for(int i=0;i<o.length;i++)o[i]=b.get(i); return o; }
    private static int sum(int[] a){ int s=0; for(int x:a)s+=x; return s; }
    private static float[] signs(int D,long sd){ Random r=new Random(sd); float[] s=new float[D]; for(int j=0;j<D;j++)s[j]=r.nextBoolean()?1f:-1f; return s; }
    /** R(x)=H(Dx)/sqrt(block) per block, dimension-preserving (blocks exactly cover d). Orthogonal -> L2-preserving. */
    private static float[] rht(float[] x, int[] blocks, float[] sign){ int Dt=sum(blocks); float[] y=new float[Dt]; int d=x.length; for(int j=0;j<Dt;j++) y[j]=(j<d?x[j]:0f)*sign[j];
        int off=0; for(int bl:blocks){ fwht(y,off,bl); float inv=1f/(float)Math.sqrt(bl); for(int j=off;j<off+bl;j++)y[j]*=inv; off+=bl; } return y; }
    private static float[][] rhtAll(float[][] X,int[] blk,float[] sign){ float[][] Y=new float[X.length][]; for(int i=0;i<X.length;i++)Y[i]=rht(X[i],blk,sign); return Y; }
    private static void fwht(float[] a,int off,int n){ for(int len=1;len<n;len<<=1){ for(int i=off;i<off+n;i+=len<<1){ for(int j=i;j<i+len;j++){ float u=a[j],v=a[j+len]; a[j]=u+v; a[j+len]=u-v; } } } }

    // ==================== data + io ====================
    private int[][] groundTruth(float[][] X, float[][] Q){ int N=X.length; int[][] gt=new int[Q.length][K]; for(int qi=0;qi<Q.length;qi++){ float[] ex=new float[N]; for(int j=0;j<N;j++) ex[j]=l2(Q[qi],X[j]); gt[qi]=topIdx(ex,K); } return gt; }
    private float[][] load(String src,int dim,int n,boolean query) throws IOException {
        if(src.equals("iid")){ Random r=new Random(SEED+(query?999:1)+dim); float[][] X=new float[n][dim]; for(int i=0;i<n;i++)for(int j=0;j<dim;j++)X[i][j]=(float)r.nextGaussian(); return X; }
        Path repo=Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
        return readFvecs(repo.resolve(REAL+src+(query?"_query.fvecs":"_base.fvecs")).toString(),0,n);
    }
    private static float[] mean(float[][] X){ int d=X[0].length; float[] c=new float[d]; for(float[] v:X)for(int j=0;j<d;j++)c[j]+=v[j]; for(int j=0;j<d;j++)c[j]/=X.length; return c; }
    private static float[] scale(float[] x,float s){ float[] o=new float[x.length]; for(int i=0;i<x.length;i++)o[i]=x[i]*s; return o; }
    private static float[] randVec(int d,Random r){ float[] v=new float[d]; for(int j=0;j<d;j++)v[j]=(float)r.nextGaussian(); return v; }
    private static float[][] randData(int d,int n,Random r){ float[][] X=new float[n][d]; for(int i=0;i<n;i++)X[i]=randVec(d,r); return X; }
    private static float l2(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static int[] topIdx(float[] sc,int m){ Integer[] idx=new Integer[sc.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(sc[x],sc[y])); int[] o=new int[m]; for(int i=0;i<m;i++)o[i]=idx[i]; return o; }
    private static List<String> ef(String pre,int[] efs){ List<String> l=new ArrayList<>(); for(int e:efs)l.add(pre+e); return l; }
    private int countDone(Path dir) throws IOException { Path p=dir.resolve("step14_rht_bbq.csv"); if(!Files.exists(p))return 0; java.util.Set<String> s=new java.util.HashSet<>(); for(String ln:(Iterable<String>)Files.lines(p)::iterator){ if(ln.startsWith("dataset,"))continue; int c=ln.indexOf(','); if(c>0)s.add(ln.substring(0,c)); } int d=0; for(String ds:new String[]{"fmnist784","iid784"}) if(s.contains(ds))d++; return d; }
    private Path outDir() throws IOException { Path o=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(o); return o; }
    private void writeFresh(Path dir,String name,String hdr) throws IOException { try(Writer w=Files.newBufferedWriter(dir.resolve(name))){ w.write(hdr); w.write("\n"); } }
    private void append(Path dir,String name,List<String> rows) throws IOException { try(Writer w=Files.newBufferedWriter(dir.resolve(name),StandardOpenOption.CREATE,StandardOpenOption.APPEND)){ for(String r:rows){ w.write(r); w.write("\n"); } } }
    private static float[][] readFvecs(String path,int skip,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; int seen=0; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; if(seen++<skip)continue; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++)v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
