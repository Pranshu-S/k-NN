/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.research;

import org.apache.lucene.search.ScoreDoc;
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
 * Track 2, step 8 — PRODUCTION-shaped consistent multi-bit HNSW: does the native recall win of
 * 2/4-bit translate into real END-TO-END warm latency + bytes-read at EQUAL recall vs 1-bit and fp32?
 *
 * Uses REAL PACKED codes (1/2/4 bits/dim, 8/4/2 dims per byte) so scoring reads the actual code
 * bytes. GEOMETRY = L2 over the per-dim-uniform reconstruction, used consistently for graph build
 * (symmetric, unpack both codes) AND navigation (asymmetric, fp32 query vs unpacked doc). One graph
 * per bit width per segment (never a 1-bit graph reused). Bytes-read/query = nodes_visited * code
 * bytes -> the decisive memory-bandwidth crux (4-bit reads 4x/node but visits far fewer nodes).
 *
 * Measures: packed-scorer microbench (ns/score, bytes/score) at 128/768/1536; per (segment,bits)
 * build time, ceiling vs ef, equal-recall cheapest config, warm p50/p99, mean nodes, bytes-read,
 * storage; per-segment bit-width selection with explicit statuses.
 * WARM latency only (sandbox); cold-cache / threaded-throughput / p99-under-load deferred to hardware.
 * Rotation OFF (step-7 showed generic rotation is unreliable; true RaBitQ is a separate branch).
 */
public class Track2ProductionMultibitTests extends KNNTestCase {

    private static final int N = 6000, NCAL = 50, NEVAL = 100, K = 10, M = 16, BEAM = 100;
    private static final int[] EFS = { 20, 50, 100, 200, 500 };
    private static final int[] DEPTHS = { 10, 20, 30, 50, 100, 200, 300, 500 };
    private static final double TARGET = 0.95, MARGIN = 0.005;
    private static final long SEED = 71L;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";
    private static final int[] BITWIDTHS = { 1, 2, 4, 32 };   // 32 = fp32 reference

    public void testProductionMultibit() throws IOException {
        List<String> micro = new ArrayList<>();
        micro.add("dim,bits,code_bytes,ns_per_score,bytes_per_score,scores_per_sec_M");
        List<String> cfg = new ArrayList<>();
        cfg.add("segment,dim,bits,ef_search,candidate_ceiling,mean_nodes_visited,code_bytes_per_vec,bytes_read_per_query,warm_p50_us,warm_p99_us,build_ms,status");
        List<String> sel = new ArrayList<>();
        sel.add("segment,sel_bits,sel_ef,sel_rerank_depth,heldout_recall,mean_nodes,bytes_read_per_query,code_bytes_per_vec,warm_p50_us,warm_p99_us,vs_1bit_bytes,vs_1bit_p50,status,reason");

        // ---- Phase: scoring microbenchmark ----
        for (int dim : new int[]{128, 768, 1536}) {
            float[][] V = randData(dim, 4096, new Random(SEED));
            float[] q = V[0];
            for (int bits : BITWIDTHS) { Packed p = encode(V, bits); double[] r = microbench(q, p, bits);
                micro.add(String.format(java.util.Locale.ROOT,"%d,%d,%d,%.2f,%d,%.1f", dim, bits, p.codeBytes, r[0], (int)r[1], 1000.0/r[0])); }
        }

        // ---- per segment: build packed graph per bit width, ceiling sweep, equal-recall config ----
        String[][] segs = { {"sift_real","128"}, {"isotropic","768"}, {"clustered","768"} };
        for (String[] sg : segs) {
            String seg = sg[0]; int d = Integer.parseInt(sg[1]);
            Data data = load(seg, d); int dim = data.X[0].length;
            int[][] gt = new int[data.Q.length][K];
            for (int qi=0; qi<data.Q.length; qi++){ float[] ex=new float[N]; for(int j=0;j<N;j++) ex[j]=l2(data.Q[qi],data.X[j]); gt[qi]=topIdx(ex,K); }
            // reference 1-bit config (bytes/p50 baseline for equal-recall comparison)
            double[] oneBitRef = null; Object[] best = null; double bestP50 = Double.POSITIVE_INFINITY;
            for (int bits : BITWIDTHS) {
                Packed p = encode(data.X, bits);
                long t0 = System.nanoTime(); OnHeapHnswGraph g = buildGraph(p, data.X, bits); long buildMs = (System.nanoTime()-t0)/1_000_000;
                int feasibleEf = -1; int selDepth = -1; double[] selCeilNodes = null;
                for (int ef : EFS) {
                    int[][] pos = new int[data.Q.length][K]; double nodes = 0;
                    for (int qi=0; qi<data.Q.length; qi++){ long[] w=new long[1]; int[] cand=search(g, p, data.X, data.Q[qi], bits, ef, w); nodes+=w[0];
                        java.util.HashMap<Integer,Integer> rk=new java.util.HashMap<>(); for(int i=0;i<cand.length;i++) rk.put(cand[i],i);
                        for(int t=0;t<K;t++) pos[qi][t]=rk.getOrDefault(gt[qi][t],-1); }
                    double ceil = recallAt(pos,0,data.Q.length,ef); double meanNodes=nodes/data.Q.length;
                    long bytesRead = (long)(meanNodes * p.codeBytes);
                    double[] lat = warmLatency(g, p, data.X, data.Q, bits, ef, 100);
                    String status = ceil<TARGET ? "SLA_UNACHIEVABLE_AT_MAX_CONFIG" : "CALIBRATED";
                    cfg.add(String.format(java.util.Locale.ROOT,"%s,%d,%d,%d,%.4f,%.1f,%d,%d,%.1f,%.1f,%d,%s",
                        seg, dim, bits, ef, ceil, meanNodes, p.codeBytes, bytesRead, lat[0], lat[1], buildMs, status));
                    if (feasibleEf<0 && ceil>=TARGET) { int dep=calibrateDepth(pos,TARGET,MARGIN); if(dep>0){ feasibleEf=ef; selDepth=dep; selCeilNodes=new double[]{ceil,meanNodes,bytesRead}; } }
                }
                if (feasibleEf>0) { int efi=idx(EFS,feasibleEf);
                    double[] lat = warmLatency(g, p, data.X, data.Q, bits, feasibleEf, selDepth);
                    // recompute pos at selected ef for held-out recall
                    int[][] pos = posAt(g, p, data.X, data.Q, gt, bits, feasibleEf);
                    double heldRec = recallAt(pos, NCAL, data.Q.length, selDepth);
                    long bytesRead = (long)(selCeilNodes[1]*p.codeBytes);
                    Object[] c = new Object[]{bits, feasibleEf, selDepth, heldRec, selCeilNodes[1], bytesRead, p.codeBytes, lat[0], lat[1]};
                    if (bits==1) oneBitRef = new double[]{bytesRead, lat[0]};
                    if (lat[0] < bestP50 && bits!=32) { bestP50 = lat[0]; best = c; }   // pick cheapest quantized by warm p50
                }
            }
            // selected per-segment (cheapest quantized bit width meeting SLA)
            if (best != null) { int bits=(int)best[0], ef=(int)best[1], dep=(int)best[2]; double hr=(double)best[3], nodes=(double)best[4]; long br=(long)best[5]; int cb=(int)best[6]; double p50=(double)best[7], p99=(double)best[8];
                double vsBytes = oneBitRef!=null ? 1.0-br/oneBitRef[0] : 0, vsP50 = oneBitRef!=null ? 1.0-p50/oneBitRef[1] : 0;
                String status = bits==1?"CALIBRATED_1BIT":bits==2?"CALIBRATED_2BIT":"CALIBRATED_4BIT";
                sel.add(String.format(java.util.Locale.ROOT,"%s,%d,%d,%d,%.4f,%.1f,%d,%d,%.1f,%.1f,%.3f,%.3f,%s,cheapest quantized by warm p50 meeting SLA",
                    seg, bits, ef, dep, hr, nodes, br, cb, p50, p99, vsBytes, vsP50, status));
            } else sel.add(String.format(java.util.Locale.ROOT,"%s,0,0,0,0,0,0,0,0,0,0,0,SLA_UNACHIEVABLE,no quantized bit width feasible", seg));
            System.out.printf(java.util.Locale.ROOT,"[step8] %s done%n", seg);
        }
        write("track2_prod_micro.csv", micro);
        write("track2_prod_configs.csv", cfg);
        write("track2_prod_selected.csv", sel);
    }

    // ---- packed codec ----
    static final class Packed { byte[][] codes; float[][] floats; float[] mn, mx; int bits, dim, codeBytes; }
    private static Packed encode(float[][] V, int bits) {
        int n=V.length, d=V[0].length; Packed p=new Packed(); p.bits=bits; p.dim=d;
        if (bits==32) { p.codeBytes=d*4; p.codes=null; p.mn=null; p.mx=null; p.floats=V; return p; }
        p.mn=new float[d]; p.mx=new float[d]; Arrays.fill(p.mn,Float.POSITIVE_INFINITY); Arrays.fill(p.mx,Float.NEGATIVE_INFINITY);
        for (float[] v:V) for(int j=0;j<d;j++){ if(v[j]<p.mn[j])p.mn[j]=v[j]; if(v[j]>p.mx[j])p.mx[j]=v[j]; }
        int L=1<<bits; p.codeBytes=(d*bits+7)/8; p.codes=new byte[n][p.codeBytes];
        for (int i=0;i<n;i++) for(int j=0;j<d;j++){ float rng=p.mx[j]-p.mn[j]; int lvl = rng<=0?0:Math.round((V[i][j]-p.mn[j])/rng*(L-1)); if(lvl<0)lvl=0; if(lvl>L-1)lvl=L-1;
            int dimsPerByte=8/bits, byteIdx=j/dimsPerByte, shift=(j%dimsPerByte)*bits; p.codes[i][byteIdx]|=(lvl<<shift); }
        return p;
    }
    private static float decLevel(Packed p, byte[] code, int j) { int dimsPerByte=8/p.bits, byteIdx=j/dimsPerByte, shift=(j%dimsPerByte)*p.bits, mask=(1<<p.bits)-1; int lvl=(code[byteIdx]>>shift)&mask; int L=1<<p.bits; float rng=p.mx[j]-p.mn[j]; return rng<=0?p.mn[j]:p.mn[j]+(float)lvl/(L-1)*rng; }
    private static float symL2(Packed p, byte[] a, byte[] b) { float s=0; for(int j=0;j<p.dim;j++){ float x=decLevel(p,a,j)-decLevel(p,b,j); s+=x*x; } return s; }
    private static float asymL2(Packed p, float[] q, byte[] b) { float s=0; for(int j=0;j<p.dim;j++){ float x=q[j]-decLevel(p,b,j); s+=x*x; } return s; }

    private double[] microbench(float[] q, Packed p, int bits) {
        int n = p.bits==32 ? p.floats.length : p.codes.length; float sink=0;
        for (int w=0;w<3;w++) for(int j=0;j<n;j++) sink+= p.bits==32? l2(q,p.floats[j]) : asymL2(p,q,p.codes[j]);
        long t0=System.nanoTime(); int iters=40;
        for (int it=0;it<iters;it++) for(int j=0;j<n;j++) sink+= p.bits==32? l2(q,p.floats[j]) : asymL2(p,q,p.codes[j]);
        long t1=System.nanoTime(); if(sink==Float.NaN) throw new IllegalStateException();
        double ns=(double)(t1-t0)/((long)iters*n); int bytesPerScore = p.bits==32? p.dim*4 : p.codeBytes;
        return new double[]{ns, bytesPerScore};
    }

    // ---- native graph (packed, consistent geometry) ----
    private OnHeapHnswGraph buildGraph(Packed p, float[][] X, int bits) throws IOException {
        final int NN = bits==32? X.length : p.codes.length;
        RandomVectorScorerSupplier sup = new RandomVectorScorerSupplier() {
            public UpdateableRandomVectorScorer scorer(){ return new UpdateableRandomVectorScorer(){ int cur=0; public int maxOrd(){return NN;} public void setScoringOrdinal(int o){cur=o;}
                public float score(int j){ return bits==32? -l2(X[cur],X[j]) : -symL2(p,p.codes[cur],p.codes[j]); } }; }
            public RandomVectorScorerSupplier copy(){ return this; } };
        return HnswGraphBuilder.create(sup, M, BEAM, SEED).build(NN);
    }
    private int[] search(OnHeapHnswGraph g, Packed p, float[][] X, float[] q, int bits, int ef, long[] work) throws IOException {
        RandomVectorScorer qs = new RandomVectorScorer(){ public int maxOrd(){return bits==32?X.length:p.codes.length;} public float score(int ord){ return bits==32? -l2(q,X[ord]) : -asymL2(p,q,p.codes[ord]); } };
        TopKnnCollector col = new TopKnnCollector(ef, Integer.MAX_VALUE); HnswGraphSearcher.search(qs, col, g, null); work[0]=col.visitedCount();
        ScoreDoc[] sd = col.topDocs().scoreDocs; int[] o=new int[sd.length]; for(int i=0;i<sd.length;i++) o[i]=sd[i].doc; return o;
    }
    private int[][] posAt(OnHeapHnswGraph g, Packed p, float[][] X, float[][] Q, int[][] gt, int bits, int ef) throws IOException {
        int[][] pos=new int[Q.length][K]; for(int qi=0;qi<Q.length;qi++){ long[] w=new long[1]; int[] cand=search(g,p,X,Q[qi],bits,ef,w); java.util.HashMap<Integer,Integer> rk=new java.util.HashMap<>(); for(int i=0;i<cand.length;i++) rk.put(cand[i],i); for(int t=0;t<K;t++) pos[qi][t]=rk.getOrDefault(gt[qi][t],-1); } return pos;
    }
    private double[] warmLatency(OnHeapHnswGraph g, Packed p, float[][] X, float[][] Q, int bits, int ef, int depth) {
        double[] best=null; try { for(int rep=0;rep<3;rep++){ double[] samp=new double[NEVAL]; int idx=0;
            for(int qi=NCAL;qi<NCAL+NEVAL;qi++){ long t0=System.nanoTime(); long[] w=new long[1]; int[] cand=search(g,p,X,Q[qi],bits,ef,w);
                int dd=Math.min(depth,cand.length); float[] ex=new float[dd]; for(int i=0;i<dd;i++) ex[i]=l2(Q[qi],X[cand[i]]);   // fp32 rerank
                Integer[] o=new Integer[dd]; for(int i=0;i<dd;i++)o[i]=i; Arrays.sort(o,(a,b)->Float.compare(ex[a],ex[b])); long t1=System.nanoTime(); samp[idx++]=(t1-t0)/1000.0; }
            Arrays.sort(samp); double[] cur={samp[samp.length/2], samp[(int)(0.99*samp.length)]}; if(best==null||cur[0]<best[0]) best=cur; } } catch(IOException e){ throw new RuntimeException(e);} return best;
    }

    private int calibrateDepth(int[][] pos,double target,double margin){ Random rng=new Random(SEED+3); for(int d:DEPTHS){ double lb=bootLower(pos,0,NCAL,d,rng); if(lb>=target+margin) return d; } return -1; }
    private double recallAt(int[][] pos,int from,int to,int depth){ double s=0;int c=0; for(int q=from;q<to;q++){ int h=0; for(int pp:pos[q]) if(pp>=0&&pp<depth) h++; s+=h/(double)K; c++; } return c==0?0:s/c; }
    private double bootLower(int[][] pos,int from,int to,int depth,Random rng){ int n=to-from; double[] m=new double[200]; for(int b=0;b<200;b++){ double s=0; for(int i=0;i<n;i++){ int q=from+rng.nextInt(n); int h=0; for(int pp:pos[q]) if(pp>=0&&pp<depth) h++; s+=h/(double)K; } m[b]=s/n; } Arrays.sort(m); return m[10]; }
    private static int idx(int[] a,int v){ for(int i=0;i<a.length;i++) if(a[i]==v) return i; return -1; }
    private static float l2(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static int[] topIdx(float[] sc,int m){ Integer[] idx=new Integer[sc.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(sc[x],sc[y])); int[] o=new int[m]; for(int i=0;i<m;i++)o[i]=idx[i]; return o; }
    private void write(String name,List<String> rows) throws IOException { Path out=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(out); try(Writer w=Files.newBufferedWriter(out.resolve(name))){ for(String r:rows){ w.write(r); w.write("\n"); } } System.out.println("WROTE "+rows.size()+" -> "+out.resolve(name)); }
    static final class Data { float[][] X, Q; }
    private Data load(String seg,int d) throws IOException { Data dt=new Data();
        if(seg.equals("sift_real")){ Path repo=Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent(); dt.X=readFvecs(repo.resolve(BASE).toString(),N); dt.Q=readFvecs(repo.resolve(QUERY).toString(),NCAL+NEVAL); }
        else { Random r=new Random(SEED); dt.X=gen(seg,d,N,r); dt.Q=gen(seg,d,NCAL+NEVAL,new Random(SEED+5)); } return dt; }
    private static float[][] randData(int d,int n,Random r){ float[][] X=new float[n][d]; for(int i=0;i<n;i++) for(int j=0;j<d;j++) X[i][j]=(float)r.nextGaussian(); return X; }
    private static float[][] gen(String kind,int d,int n,Random rng){ float[][] X=new float[n][d];
        if(kind.equals("isotropic")){ for(int i=0;i<n;i++) for(int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); }
        else if(kind.equals("clustered")){ int ncl=32; float[][] c=new float[ncl][d]; for(int a=0;a<ncl;a++) for(int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f; for(int i=0;i<n;i++){ int a=rng.nextInt(ncl); for(int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } }
        else throw new IllegalArgumentException(kind); return X; }
    private static float[][] readFvecs(String path,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++) v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
