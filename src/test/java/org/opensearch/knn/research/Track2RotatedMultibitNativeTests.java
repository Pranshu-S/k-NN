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
 * Track 2, step 7 — rotation-based multi-bit representation used CONSISTENTLY for graph construction
 * AND query navigation (fixing the step-6 build/navigate geometry mismatch). Real Lucene HNSW graph.
 *
 * Representation (OSQ-style rotated scalar quantization — NOT literal RaBitQ; labelled honestly):
 *   optional random orthogonal rotation R (Gaussian + Gram-Schmidt, fixed seed, identical for
 *   base+query, L2-preserving); per-dim uniform scalar quantization to b in {1,2,4} bits over the
 *   (rotated) data's per-dim [min,max]; the code DEQUANTIZES to a reconstruction xhat.
 *   GEOMETRY = L2 over xhat, used for BOTH: graph build (symmetric L2(xhat_a,xhat_b)) and query
 *   navigation (asymmetric L2(rotated_fp32_query, xhat_x)). One geometry, one graph per representation.
 *   code bytes/vec = ceil(dim*b/8); xhat kept in-heap only to drive the graph (accounted separately).
 *
 * Ground truth = exact fp32 top-k over ORIGINAL vectors (rotation-invariant). Rerank uses fp32.
 * Phases: A flat codec quality, C native candidate-ceiling vs ef_search, D rerank recall+warm latency,
 * E per-segment cheapest feasible representation. Warm latency only (sandbox); cold/p99 -> real hw.
 */
public class Track2RotatedMultibitNativeTests extends KNNTestCase {

    private static final int N = 10000, NCAL = 50, NEVAL = 100, K = 10, M = 16, BEAM = 100;
    private static final int[] EFS = { 20, 50, 100, 200, 500 };
    private static final int[] DEPTHS = { 10, 20, 30, 50, 100, 200, 300, 500 };
    private static final double TARGET = 0.95, MARGIN = 0.005, C_GRAPH = 1.0, C_EXACT = 10.0;
    private static final long SEED = 61L;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";

    static final class Rep { final String name; final int bits; final boolean rot; Rep(String n,int b,boolean r){name=n;bits=b;rot=r;} }
    private static final Rep[] REPS = {
        new Rep("fp32", 32, false),
        new Rep("1bit", 1, false), new Rep("1bit_rot", 1, true),
        new Rep("2bit", 2, false), new Rep("2bit_rot", 2, true),
        new Rep("4bit", 4, false), new Rep("4bit_rot", 4, true),
    };

    public void testRotatedMultibit() throws IOException {
        List<String> flat = new ArrayList<>();
        flat.add("segment,dim,representation,bits,rotation,code_bytes_per_vec,flat_topk_recall,rmse_vs_fp32");
        List<String> cfg = new ArrayList<>();
        cfg.add("segment,representation,bits,rotation,ef_search,candidate_ceiling,mean_nodes_visited,recall@100rerank,warm_p50_us,warm_p99_us");
        List<String> sel = new ArrayList<>();
        sel.add("segment,sel_representation,sel_bits,sel_rotation,sel_ef,sel_rerank_depth,heldout_recall,work_cost,code_bytes_per_vec,warm_p50_us,status,ef_vs_1bitADC");

        String[][] segs = { {"sift_real","128"}, {"isotropic","768"}, {"clustered","128"} };
        for (String[] sg : segs) {
            String seg = sg[0]; int d = Integer.parseInt(sg[1]);
            Data data = load(seg, d);
            int dim = data.X[0].length;
            float[][] R = rotationMatrix(dim, SEED);          // one rotation per segment, reused across bit widths
            float[][] rotX = applyRot(data.X, R), rotQ = applyRot(data.Q, R);
            // ground truth (fp32, original space)
            int[][] gt = new int[data.Q.length][K];
            for (int q = 0; q < data.Q.length; q++) { float[] ex = new float[N]; for (int j=0;j<N;j++) ex[j]=l2(data.Q[q],data.X[j]); gt[q]=topIdx(ex,K); }

            List<Object[]> perRepEval = new ArrayList<>();    // {Rep, ceilingByEf[], posByEf[][][], codeBytes}
            for (Rep rep : REPS) {
                float[][] docSpace = rep.rot ? rotX : data.X, qSpace = rep.rot ? rotQ : data.Q;
                float[][] xhat; int codeBytes;
                if (rep.bits == 32) { xhat = docSpace; codeBytes = dim*4; }
                else { xhat = quantRecon(docSpace, rep.bits); codeBytes = (dim*rep.bits + 7)/8; }
                // Phase A: flat top-k recall of exhaustive L2 over xhat (asymmetric query = qSpace fp32)
                double flatRec = flatTopKRecall(qSpace, xhat, gt);
                double rmse = rmseVsFp32(data.Q, data.X, qSpace, xhat, 30);
                flat.add(String.format(java.util.Locale.ROOT,"%s,%d,%s,%d,%b,%d,%.4f,%.4g", seg, dim, rep.name, rep.bits, rep.rot, codeBytes, flatRec, rmse));
                // Phase C: build a REAL Lucene HNSW over xhat (L2), navigate with qSpace fp32
                OnHeapHnswGraph graph = buildGraph(xhat);
                double[] ceilByEf = new double[EFS.length]; int[][][] posByEf = new int[EFS.length][][]; double[][] latByEf = new double[EFS.length][];
                for (int ei = 0; ei < EFS.length; ei++) { int ef = EFS[ei];
                    int[][] pos = new int[data.Q.length][K]; double nodes = 0;
                    for (int q = 0; q < data.Q.length; q++) { long[] w=new long[1]; int[] cand = search(graph, xhat, qSpace[q], ef, w); nodes += w[0];
                        java.util.HashMap<Integer,Integer> rk=new java.util.HashMap<>(); for(int i=0;i<cand.length;i++) rk.put(cand[i],i);
                        for (int t=0;t<K;t++) pos[q][t]=rk.getOrDefault(gt[q][t],-1); }
                    ceilByEf[ei] = recallAt(pos, 0, data.Q.length, ef); posByEf[ei] = pos;
                    double[] lat = warmLatency(graph, xhat, qSpace, data.Q, data.X, ef, 100);
                    latByEf[ei] = lat;
                    cfg.add(String.format(java.util.Locale.ROOT,"%s,%s,%d,%b,%d,%.4f,%.1f,%.4f,%.1f,%.1f", seg, rep.name, rep.bits, rep.rot, ef,
                        ceilByEf[ei], nodes/data.Q.length, recallAt(pos,NCAL,data.Q.length,Math.min(100,ef)), lat[0], lat[1]));
                }
                perRepEval.add(new Object[]{ rep, ceilByEf, posByEf, codeBytes, graph, xhat, qSpace });
            }
            // reference: 1bit ADC-ish ef needed to hit SLA (use the "1bit" no-rot rep as the step-6 analogue)
            int ef1bit = efToMeetSLA(perRepEval, "1bit");
            // Phase E: cheapest feasible (representation, ef, rerank_depth) per segment
            Object[] best = null; double bestCost = Double.POSITIVE_INFINITY;
            for (Object[] pe : perRepEval) { Rep rep=(Rep)pe[0]; double[] ceil=(double[])pe[1]; int[][][] posByEf=(int[][][])pe[2]; int cb=(int)pe[3];
                for (int ei=0; ei<EFS.length; ei++) { if (ceil[ei] < TARGET) continue;
                    int d2 = calibrateDepth(posByEf[ei], TARGET, MARGIN); if (d2 < 0) continue;
                    double nodes = 0; // approx: use ef as proxy already in ceil; recompute mean nodes cheaply omitted -> use ef
                    double cost = C_GRAPH*EFS[ei] + C_EXACT*d2;   // work proxy (nodes ~ scales with ef)
                    if (cost < bestCost) { bestCost=cost; best=new Object[]{rep, EFS[ei], d2, cost, cb, pe}; }
                }
            }
            if (best != null) { Rep rep=(Rep)best[0]; int ef=(int)best[1], dep=(int)best[2]; double cost=(double)best[3]; int cb=(int)best[4]; Object[] pe=(Object[])best[5];
                OnHeapHnswGraph g=(OnHeapHnswGraph)pe[4]; float[][] xh=(float[][])pe[5]; float[][] qs=(float[][])pe[6];
                int efi=idx(EFS,ef); double heldRec = recallAt(((int[][][])pe[2])[efi], NCAL, data.Q.length, dep);
                double[] lat = warmLatency(g, xh, qs, data.Q, data.X, ef, dep);
                String efcmp = ef1bit<0 ? "1bitADC infeasible" : (ef < ef1bit ? ("lower ef ("+ef+" vs "+ef1bit+")") : ("ef "+ef+" vs 1bit "+ef1bit));
                sel.add(String.format(java.util.Locale.ROOT,"%s,%s,%d,%b,%d,%d,%.4f,%.0f,%d,%.1f,CALIBRATED,%s", seg, rep.name, rep.bits, rep.rot, ef, dep, heldRec, cost, cb, lat[0], efcmp));
            } else sel.add(String.format(java.util.Locale.ROOT,"%s,NONE,0,false,0,0,0,0,0,0,SLA_UNACHIEVABLE_AT_MAX_CONFIG,n/a", seg));
            System.out.printf(java.util.Locale.ROOT,"[step7] segment %s done (1bit ef->SLA=%s)%n", seg, ef1bit<0?"infeasible":(""+ef1bit));
        }
        write("track2_rotmb_flat.csv", flat);
        write("track2_rotmb_configs.csv", cfg);
        write("track2_rotmb_selected.csv", sel);
    }

    private int efToMeetSLA(List<Object[]> perRep, String name){ for(Object[] pe:perRep){ Rep r=(Rep)pe[0]; if(!r.name.equals(name)) continue; double[] c=(double[])pe[1]; for(int i=0;i<EFS.length;i++) if(c[i]>=TARGET) return EFS[i]; } return -1; }
    private static int idx(int[] a,int v){ for(int i=0;i<a.length;i++) if(a[i]==v) return i; return -1; }
    private int calibrateDepth(int[][] pos, double target, double margin){ Random rng=new Random(SEED+3); for(int d:DEPTHS){ double lb=bootLower(pos,0,NCAL,d,rng); if(lb>=target+margin) return d; } return -1; }
    private double recallAt(int[][] pos,int from,int to,int depth){ double s=0;int c=0; for(int q=from;q<to;q++){ int h=0; for(int p:pos[q]) if(p>=0&&p<depth) h++; s+=h/(double)K; c++; } return c==0?0:s/c; }
    private double bootLower(int[][] pos,int from,int to,int depth,Random rng){ int n=to-from; double[] m=new double[200]; for(int b=0;b<200;b++){ double s=0; for(int i=0;i<n;i++){ int q=from+rng.nextInt(n); int h=0; for(int p:pos[q]) if(p>=0&&p<depth) h++; s+=h/(double)K; } m[b]=s/n; } Arrays.sort(m); return m[10]; }

    // ---- codec ----
    private static float[][] quantRecon(float[][] V, int bits) {
        int n=V.length, d=V[0].length, L=1<<bits; float[] mn=new float[d], mx=new float[d];
        Arrays.fill(mn, Float.POSITIVE_INFINITY); Arrays.fill(mx, Float.NEGATIVE_INFINITY);
        for (float[] v : V) for (int j=0;j<d;j++){ if(v[j]<mn[j])mn[j]=v[j]; if(v[j]>mx[j])mx[j]=v[j]; }
        float[][] xhat=new float[n][d];
        for (int i=0;i<n;i++) for (int j=0;j<d;j++){ float rng=mx[j]-mn[j]; if(rng<=0){ xhat[i][j]=mn[j]; continue; }
            int lvl=Math.round((V[i][j]-mn[j])/rng*(L-1)); if(lvl<0)lvl=0; if(lvl>L-1)lvl=L-1; xhat[i][j]=mn[j]+(float)lvl/(L-1)*rng; }
        return xhat;
    }
    // dense random orthogonal rotation (Gaussian + Gram-Schmidt), fixed seed; L2-preserving
    private static float[][] rotationMatrix(int d, long seed) {
        Random r=new Random(seed); double[][] a=new double[d][d];
        for(int i=0;i<d;i++) for(int j=0;j<d;j++) a[i][j]=r.nextGaussian();
        for(int i=0;i<d;i++){ for(int k=0;k<i;k++){ double dot=0; for(int j=0;j<d;j++) dot+=a[i][j]*a[k][j]; for(int j=0;j<d;j++) a[i][j]-=dot*a[k][j]; }
            double nrm=0; for(int j=0;j<d;j++) nrm+=a[i][j]*a[i][j]; nrm=Math.sqrt(nrm); for(int j=0;j<d;j++) a[i][j]/=nrm; }
        float[][] R=new float[d][d]; for(int i=0;i<d;i++) for(int j=0;j<d;j++) R[i][j]=(float)a[i][j]; return R;
    }
    private static float[][] applyRot(float[][] V, float[][] R){ int n=V.length,d=R.length; float[][] O=new float[n][d];
        for(int i=0;i<n;i++) for(int a=0;a<d;a++){ double s=0; float[] Ra=R[a]; float[] Vi=V[i]; for(int j=0;j<d;j++) s+=Ra[j]*Vi[j]; O[i][a]=(float)s; } return O; }

    // ---- flat quality (Phase A) ----
    private double flatTopKRecall(float[][] Q, float[][] xhat, int[][] gt){ double s=0; for(int q=0;q<Q.length;q++){ float[] sc=new float[xhat.length]; for(int j=0;j<xhat.length;j++) sc[j]=l2(Q[q],xhat[j]); int[] top=topIdx(sc,K); java.util.HashSet<Integer> t=new java.util.HashSet<>(); for(int x:gt[q])t.add(x); int h=0; for(int x:top) if(t.contains(x))h++; s+=h/(double)K; } return s/Q.length; }
    private double rmseVsFp32(float[][] Qorig, float[][] Xorig, float[][] Qspace, float[][] xhat, int sample){ Random rng=new Random(SEED+9); double se=0;int c=0; for(int q=0;q<Math.min(sample,Qspace.length);q++) for(int s=0;s<100;s++){ int j=rng.nextInt(xhat.length); double approx=l2(Qspace[q],xhat[j]); double exact=l2(Qorig[q],Xorig[j]); se+=(approx-exact)*(approx-exact); c++; } return Math.sqrt(se/c); }

    // ---- native graph (Phase C/D) ----
    private OnHeapHnswGraph buildGraph(float[][] xhat) throws IOException {
        final int NN=xhat.length;
        RandomVectorScorerSupplier sup = new RandomVectorScorerSupplier() {
            public UpdateableRandomVectorScorer scorer(){ return new UpdateableRandomVectorScorer(){ int cur=0; public int maxOrd(){return NN;} public void setScoringOrdinal(int o){cur=o;} public float score(int j){ return -l2(xhat[cur],xhat[j]); } }; }
            public RandomVectorScorerSupplier copy(){ return this; } };
        return HnswGraphBuilder.create(sup, M, BEAM, SEED).build(NN);
    }
    private int[] search(OnHeapHnswGraph g, float[][] xhat, float[] q, int ef, long[] work) throws IOException {
        RandomVectorScorer qs = new RandomVectorScorer(){ public int maxOrd(){return xhat.length;} public float score(int ord){ return -l2(q, xhat[ord]); } };
        TopKnnCollector col = new TopKnnCollector(ef, Integer.MAX_VALUE);
        HnswGraphSearcher.search(qs, col, g, null); work[0]=col.visitedCount();
        ScoreDoc[] sd = col.topDocs().scoreDocs; int[] o=new int[sd.length]; for(int i=0;i<sd.length;i++) o[i]=sd[i].doc; return o;
    }
    private double[] warmLatency(OnHeapHnswGraph g, float[][] xhat, float[][] qSpace, float[][] Qorig, float[][] Xorig, int ef, int depth){
        double[] best=null; try { for(int rep=0;rep<3;rep++){ double[] samp=new double[NEVAL]; int idx=0;
            for(int q=NCAL;q<NCAL+NEVAL;q++){ long t0=System.nanoTime(); long[] w=new long[1]; int[] cand=search(g,xhat,qSpace[q],ef,w);
                int dd=Math.min(depth,cand.length); float[] ex=new float[dd]; for(int i=0;i<dd;i++) ex[i]=l2(Qorig[q],Xorig[cand[i]]);
                Integer[] o=new Integer[dd]; for(int i=0;i<dd;i++)o[i]=i; Arrays.sort(o,(a,b)->Float.compare(ex[a],ex[b])); long t1=System.nanoTime(); samp[idx++]=(t1-t0)/1000.0; }
            Arrays.sort(samp); double[] cur={samp[samp.length/2], samp[(int)(0.99*samp.length)]}; if(best==null||cur[0]<best[0]) best=cur; } } catch(IOException e){ throw new RuntimeException(e);} return best;
    }

    private static float l2(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static int[] topIdx(float[] sc,int m){ Integer[] idx=new Integer[sc.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(sc[x],sc[y])); int[] o=new int[m]; for(int i=0;i<m;i++)o[i]=idx[i]; return o; }
    private void write(String name,List<String> rows) throws IOException { Path out=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(out); try(Writer w=Files.newBufferedWriter(out.resolve(name))){ for(String r:rows){ w.write(r); w.write("\n"); } } System.out.println("WROTE "+rows.size()+" -> "+out.resolve(name)); }
    static final class Data { float[][] X, Q; }
    private Data load(String seg,int d) throws IOException { Data dt=new Data();
        if (seg.equals("sift_real")){ Path repo=Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent(); dt.X=readFvecs(repo.resolve(BASE).toString(),N); dt.Q=readFvecs(repo.resolve(QUERY).toString(),NCAL+NEVAL); }
        else { Random r=new Random(SEED); dt.X=gen(seg,d,N,r); dt.Q=gen(seg,d,NCAL+NEVAL,new Random(SEED+5)); } return dt; }
    private static float[][] gen(String kind,int d,int n,Random rng){ float[][] X=new float[n][d];
        if(kind.equals("isotropic")){ for(int i=0;i<n;i++) for(int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); }
        else if(kind.equals("clustered")){ int ncl=32; float[][] c=new float[ncl][d]; for(int a=0;a<ncl;a++) for(int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f; for(int i=0;i<n;i++){ int a=rng.nextInt(ncl); for(int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } }
        else throw new IllegalArgumentException(kind); return X; }
    private static float[][] readFvecs(String path,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++) v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
