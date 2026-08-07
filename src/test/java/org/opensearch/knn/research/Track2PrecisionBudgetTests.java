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
 * Track 2, step 12 — ADAPTIVE PRECISION BUDGETING. Can smarter low-bit representations reach near-4-bit
 * HNSW candidate-generation quality at ~1.25-2 effective bits/dim, moving the 71.5% 4-bit mass down?
 *
 * Tests, on representative step-11 segments, at EQUAL candidate recall@10:
 *   A) smart 1-bit: conditional-mean levels (vs current min/max) [note: baseline ALREADY uses fp32-query
 *      ADC, so A2 is the baseline]; centering is a no-op for per-dim-minmax (documented).
 *   B) preconditioning: structured Hadamard (sign-flip + FWHT) applied consistently -- NOT dense random.
 *   C) FRACTIONAL / unequal precision: base 1-bit + extra bit on a fraction f of dims (eff = 1+f), with
 *      allocation policies: random(control) / variance / recon-error / neighbourhood-discrimination.
 *   D) uniform 1/2/4-bit + fp32 references.
 * Candidate recall is the PRIMARY metric (not final-reranked). Consistent geometry: build AND navigate
 * on the same reconstruction. Storage = actual bit-packed bytes = ceil(sum bits/8) + sidecars.
 * Warm/one-host (arm64); representative small segments to fit the suite budget. Honest: the hypothesis
 * is NOT assumed -- for isotropic data, no dim is special, so fractional cannot beat uniform (a floor).
 */
public class Track2PrecisionBudgetTests extends KNNTestCase {

    private static final int K = 10, M = 16, BEAM = 100;
    private static final int NCAL = 100, NEVAL = 100, NQ = NCAL + NEVAL;
    private static final int[] EFS = { 20, 50, 100, 200, 500 };
    private static final double TARGET = 0.95;
    private static final long SEED = 71L;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs", QUERY = "research/acorn/data/sift/sift_query.fvecs";

    // representative segments (small N for tractability): type,dim,N,seed,siftOff
    static final Object[][] SEGS = {
        { "sift128", 128, 8000, 101, 0 },        // real, structured (1-bit anchor)
        { "iid128", 128, 8000, 102, 0 },         // isotropic 128 (information floor?)
        { "iid384", 384, 7000, 103, 0 },         // isotropic 384
        { "manifold768", 768, 6000, 104, 0 },    // low-rank 768 (variance hidden in rotated basis)
        { "iso768", 768, 5000, 105, 0 },         // isotropic 768 (hardest)
    };

    public void testPrecisionBudget() throws IOException {
        int start = Integer.getInteger("start", countDone());
        Path dir = outDir();
        String h = "seg,type,dim,N,rep,precond,policy,frac,eff_bits,code_bytes_per_vec,sidecar_bytes,total_bytes_per_vec,"
            + "ceil20,ceil50,ceil100,ceil200,ceil500,ceil_max,min_ef_095,feasible_095,build_ms";
        if (start == 0) writeFresh(dir, "step12_representations.csv", h);
        System.out.printf(java.util.Locale.ROOT, "[step12] start=%d nseg=%d%n", start, SEGS.length);

        for (int si = start; si < SEGS.length; si++) {
            Object[] sp = SEGS[si];
            String type = (String) sp[0]; int dim = (int) sp[1], N = (int) sp[2]; long sd = (int) sp[3];
            float[][] X = materialize(type, dim, N, sd, (int) sp[4]);
            float[][] Q = materializeQ(type, dim, sd);
            int[][] gt = groundTruth(X, Q);
            // per-dim stats for allocation policies (on X)
            double[] varJ = perDimVar(X);
            double[] reconJ = perDimReconErr1bit(X);
            double[] nbhdJ = perDimNeighbourDiscrim(X, gt, Q);   // dims that discriminate true neighbours
            List<String> rows = new ArrayList<>();

            // ---- reference + smart-1bit + uniform ----
            rows.add(profile(si, type, dim, N, "uniform1", "none", "uniform", 0.0, uniformBits(dim,1), false, X, Q, gt, 0));
            rows.add(profile(si, type, dim, N, "smart1", "none", "condmean", 0.0, uniformBits(dim,1), true, X, Q, gt, 0));
            rows.add(profile(si, type, dim, N, "uniform2", "none", "uniform", 0.0, uniformBits(dim,2), false, X, Q, gt, 0));
            rows.add(profile(si, type, dim, N, "uniform4", "none", "uniform", 0.0, uniformBits(dim,4), false, X, Q, gt, 0));

            // ---- fractional: base 1-bit + f fraction at 2-bit (eff=1+f) ----
            for (double f : new double[]{ 0.25, 0.5 }) {
                rows.add(profile(si,type,dim,N,"frac"+(1+f),"none","random", f, allocPlus1(dim,f,randScore(dim,sd)), false, X,Q,gt,0));
                rows.add(profile(si,type,dim,N,"frac"+(1+f),"none","variance",f, allocPlus1(dim,f,varJ),   false, X,Q,gt,0));
                rows.add(profile(si,type,dim,N,"frac"+(1+f),"none","recon",   f, allocPlus1(dim,f,reconJ), false, X,Q,gt,0));
                rows.add(profile(si,type,dim,N,"frac"+(1+f),"none","nbhd",    f, allocPlus1(dim,f,nbhdJ),  false, X,Q,gt,0));
            }

            // ---- preconditioning: Hadamard, then uniform-1bit and variance-allocated frac1.5 ----
            rows.add(profileHad(si,type,dim,N,"had_uniform1","hadamard","uniform",0.0,1, X,Q,gt,sd));
            rows.add(profileHad(si,type,dim,N,"had_frac1.5","hadamard","variance",0.5,-1, X,Q,gt,sd));

            // fp32 reference
            rows.add(profileFp32(si,type,dim,N,X,Q,gt));

            append(dir, "step12_representations.csv", rows);
            System.out.printf(java.util.Locale.ROOT, "[step12] seg %d %s dim=%d done%n", si, type, dim);
        }
        System.out.println("[step12] DONE");
    }

    // ---- profile one representation: build graph on variable-bit reconstruction, sweep ef, ceiling ----
    private String profile(int si, String type, int dim, int N, String rep, String precond, String policy, double frac,
                           int[] bits, boolean smart1, float[][] X, float[][] Q, int[][] gt, int sidecar) throws IOException {
        VarPacked p = encodeVar(X, bits, smart1);
        long t0 = System.nanoTime(); OnHeapHnswGraph g = buildGraph(p, N); long buildMs = (System.nanoTime()-t0)/1_000_000;
        double[] ceils = new double[EFS.length]; double cmax=0; int minEf=-1;
        for (int e=0;e<EFS.length;e++){ int[][] pos = posAt(g,p,Q,gt,EFS[e]); ceils[e]=recallAt(pos,NCAL,NQ,EFS[e]); cmax=Math.max(cmax,ceils[e]); if(minEf<0 && ceils[e]>=TARGET) minEf=EFS[e]; }
        return fmt(si,type,dim,N,rep,precond,policy,frac,p.effBits,p.codeBytes,sidecar,ceils,cmax,minEf,buildMs);
    }
    private String profileHad(int si, String type, int dim, int N, String rep, String precond, String policy, double frac, int uniformB,
                              float[][] X, float[][] Q, int[][] gt, long sd) throws IOException {
        int D = pow2(dim); float[] sign = signs(D, sd);
        float[][] Xt = hadamardAll(X, D, sign), Qt = hadamardAll(Q, D, sign);
        int[] bits = uniformB>0 ? uniformBits(D, uniformB) : allocPlus1(D, frac, perDimVar(Xt));
        VarPacked p = encodeVar(Xt, bits, false);
        long t0=System.nanoTime(); OnHeapHnswGraph g=buildGraph(p,N); long buildMs=(System.nanoTime()-t0)/1_000_000;
        double[] ceils=new double[EFS.length]; double cmax=0; int minEf=-1;
        for (int e=0;e<EFS.length;e++){ int[][] pos=posAt(g,p,Qt,gt,EFS[e]); ceils[e]=recallAt(pos,NCAL,NQ,EFS[e]); cmax=Math.max(cmax,ceils[e]); if(minEf<0&&ceils[e]>=TARGET) minEf=EFS[e]; }
        // storage: packed code over D padded dims + 4B seed sidecar amortized (negligible/seg); report eff over ORIGINAL dim
        double effOverOrig = p.effBits * D / dim;
        return String.format(java.util.Locale.ROOT,"%d,%s,%d,%d,%s,%s,%s,%.2f,%.3f,%d,%d,%d,%s,%.4f,%d,%d,%d",
            si,type,dim,N,rep,precond,policy,frac,effOverOrig,p.codeBytes,0,p.codeBytes,ceilStr(ceils),cmax,minEf,minEf>0?1:0,buildMs);
    }
    private String profileFp32(int si, String type, int dim, int N, float[][] X, float[][] Q, int[][] gt) throws IOException {
        VarPacked p=new VarPacked(); p.fp32=X; p.dim=dim; p.effBits=32; p.codeBytes=dim*4;
        long t0=System.nanoTime(); OnHeapHnswGraph g=buildGraph(p,N); long buildMs=(System.nanoTime()-t0)/1_000_000;
        double[] ceils=new double[EFS.length]; double cmax=0; int minEf=-1;
        for(int e=0;e<EFS.length;e++){ int[][] pos=posAt(g,p,Q,gt,EFS[e]); ceils[e]=recallAt(pos,NCAL,NQ,EFS[e]); cmax=Math.max(cmax,ceils[e]); if(minEf<0&&ceils[e]>=TARGET) minEf=EFS[e]; }
        return fmt(si,type,dim,N,"fp32","none","none",0.0,32,dim*4,0,ceils,cmax,minEf,buildMs);
    }
    private String fmt(int si,String type,int dim,int N,String rep,String pc,String pol,double frac,double eff,int codeB,int side,double[] ceils,double cmax,int minEf,long buildMs){
        return String.format(java.util.Locale.ROOT,"%d,%s,%d,%d,%s,%s,%s,%.2f,%.3f,%d,%d,%d,%s,%.4f,%d,%d,%d",
            si,type,dim,N,rep,pc,pol,frac,eff,codeB,side,codeB+side,ceilStr(ceils),cmax,minEf,minEf>0?1:0,buildMs);
    }
    private String ceilStr(double[] c){ StringBuilder b=new StringBuilder(); for(int i=0;i<c.length;i++){ if(i>0)b.append(","); b.append(String.format(java.util.Locale.ROOT,"%.4f",c[i])); } return b.toString(); }

    // ===================== variable-bit codec (per-dim bits, arbitrary level values, ADC) =====================
    static final class VarPacked { byte[][] levels; float[][] recon; int[] bits; int dim; double effBits; int codeBytes; float[][] fp32; }
    private static VarPacked encodeVar(float[][] X, int[] bits, boolean smart1) {
        int n=X.length, d=X[0].length; VarPacked p=new VarPacked(); p.bits=bits; p.dim=d;
        p.recon=new float[d][]; long totBits=0;
        float[] mn=new float[d], mx=new float[d]; Arrays.fill(mn,Float.POSITIVE_INFINITY); Arrays.fill(mx,Float.NEGATIVE_INFINITY);
        for(float[] v:X) for(int j=0;j<d;j++){ if(v[j]<mn[j])mn[j]=v[j]; if(v[j]>mx[j])mx[j]=v[j]; }
        for(int j=0;j<d;j++){ int L=1<<bits[j]; totBits+=bits[j]; p.recon[j]=new float[L];
            if(bits[j]==1 && smart1){ // conditional means around per-dim mean threshold
                double thr=0; for(float[] v:X) thr+=v[j]; thr/=n; double lo=0,hi=0; int nl=0,nh=0;
                for(float[] v:X){ if(v[j]<thr){lo+=v[j];nl++;} else {hi+=v[j];nh++;} }
                p.recon[j][0]=(float)(nl>0?lo/nl:mn[j]); p.recon[j][1]=(float)(nh>0?hi/nh:mx[j]);
            } else { for(int l=0;l<L;l++){ float rng=mx[j]-mn[j]; p.recon[j][l]= rng<=0?mn[j]: mn[j]+(float)l/(L-1)*rng; } }
        }
        p.effBits=(double)totBits/d; p.codeBytes=(int)((totBits+7)/8);
        p.levels=new byte[n][d];
        for(int i=0;i<n;i++) for(int j=0;j<d;j++){ float[] rc=p.recon[j]; int best=0; float bd=Float.MAX_VALUE; for(int l=0;l<rc.length;l++){ float e=X[i][j]-rc[l]; e=e*e; if(e<bd){bd=e;best=l;} } p.levels[i][j]=(byte)best; }
        return p;
    }
    private static float asymDist(VarPacked p, float[] q, byte[] lv){ float s=0; for(int j=0;j<p.dim;j++){ float x=q[j]-p.recon[j][lv[j]&0xFF]; s+=x*x; } return s; }
    private static float symDist(VarPacked p, byte[] a, byte[] b){ float s=0; for(int j=0;j<p.dim;j++){ float x=p.recon[j][a[j]&0xFF]-p.recon[j][b[j]&0xFF]; s+=x*x; } return s; }

    // ---- allocation policies: base 1 bit, +1 bit to top-frac dims by score ----
    private static int[] uniformBits(int d,int b){ int[] o=new int[d]; Arrays.fill(o,b); return o; }
    private static int[] allocPlus1(int d, double frac, double[] score){ int k=(int)Math.round(frac*d); Integer[] idx=new Integer[d]; for(int i=0;i<d;i++)idx[i]=i; Arrays.sort(idx,(a,b)->Double.compare(score[b],score[a])); int[] bits=new int[d]; Arrays.fill(bits,1); for(int i=0;i<k;i++) bits[idx[i]]=2; return bits; }
    private static double[] randScore(int d,long sd){ Random r=new Random(sd+7); double[] s=new double[d]; for(int j=0;j<d;j++)s[j]=r.nextDouble(); return s; }
    private double[] perDimVar(float[][] X){ int n=X.length,d=X[0].length; double[] m=new double[d],v=new double[d]; for(float[] x:X) for(int j=0;j<d;j++) m[j]+=x[j]; for(int j=0;j<d;j++)m[j]/=n; for(float[] x:X) for(int j=0;j<d;j++){ double e=x[j]-m[j]; v[j]+=e*e; } for(int j=0;j<d;j++)v[j]/=n; return v; }
    private double[] perDimReconErr1bit(float[][] X){ int n=X.length,d=X[0].length; float[] mn=new float[d],mx=new float[d]; Arrays.fill(mn,Float.POSITIVE_INFINITY);Arrays.fill(mx,Float.NEGATIVE_INFINITY); for(float[] x:X)for(int j=0;j<d;j++){if(x[j]<mn[j])mn[j]=x[j]; if(x[j]>mx[j])mx[j]=x[j];} double[] err=new double[d]; for(float[] x:X)for(int j=0;j<d;j++){ float rng=mx[j]-mn[j]; float rec= x[j]<(mn[j]+mx[j])/2? mn[j]:mx[j]; double e=x[j]-rec; err[j]+=e*e; } return err; }
    /** neighbourhood-discrimination: per-dim mean squared difference between each eval query's TRUE top-10 docs
     *  (dims where near neighbours differ most need precision to separate them). Calibration data only. */
    private double[] perDimNeighbourDiscrim(float[][] X, int[][] gt, float[][] Q){ int d=X[0].length; double[] s=new double[d]; int cnt=0;
        for(int qi=0; qi<NCAL; qi++){ int[] nn=gt[qi]; for(int a=0;a<nn.length;a++) for(int b=a+1;b<nn.length;b++){ float[] xa=X[nn[a]],xb=X[nn[b]]; for(int j=0;j<d;j++){ float e=xa[j]-xb[j]; s[j]+=e*e; } cnt++; } }
        return s;
    }

    // ---- Hadamard preconditioning ----
    private static int pow2(int d){ int p=1; while(p<d)p<<=1; return p; }
    private static float[] signs(int D,long sd){ Random r=new Random(sd+13); float[] s=new float[D]; for(int j=0;j<D;j++)s[j]=r.nextBoolean()?1f:-1f; return s; }
    private static float[][] hadamardAll(float[][] X,int D,float[] sign){ float[][] Y=new float[X.length][]; int d=X[0].length; float inv=1f/(float)Math.sqrt(D); for(int i=0;i<X.length;i++){ float[] y=new float[D]; for(int j=0;j<d;j++) y[j]=X[i][j]*sign[j]; fwht(y); for(int j=0;j<D;j++) y[j]*=inv; Y[i]=y; } return Y; }
    private static void fwht(float[] a){ int n=a.length; for(int len=1;len<n;len<<=1){ for(int i=0;i<n;i+=len<<1){ for(int j=i;j<i+len;j++){ float u=a[j],v=a[j+len]; a[j]=u+v; a[j+len]=u-v; } } } }

    // ===================== HNSW build/search (consistent geometry) =====================
    private OnHeapHnswGraph buildGraph(VarPacked p, int N) throws IOException {
        final boolean f=p.fp32!=null;
        RandomVectorScorerSupplier sup=new RandomVectorScorerSupplier(){ public UpdateableRandomVectorScorer scorer(){ return new UpdateableRandomVectorScorer(){ int cur=0; public int maxOrd(){return N;} public void setScoringOrdinal(int o){cur=o;} public float score(int j){ return f? -l2(p.fp32[cur],p.fp32[j]) : -symDist(p,p.levels[cur],p.levels[j]); } }; } public RandomVectorScorerSupplier copy(){return this;} };
        return HnswGraphBuilder.create(sup,M,BEAM,SEED).build(N);
    }
    static final class GV extends HnswGraph { final OnHeapHnswGraph g; int[] cur=new int[0]; int sz=0,up=0; GV(OnHeapHnswGraph g){this.g=g;} public void seek(int l,int nd){NeighborArray na=g.getNeighbors(l,nd);cur=na.nodes();sz=na.size();up=0;} public int size(){return g.size();} public int nextNeighbor(){return up<sz?cur[up++]:DocIdSetIterator.NO_MORE_DOCS;} public int numLevels()throws IOException{return g.numLevels();} public int maxConn(){return g.maxConn();} public int entryNode()throws IOException{return g.entryNode();} public int neighborCount(){return sz;} public HnswGraph.NodesIterator getNodesOnLevel(int l)throws IOException{return g.getNodesOnLevel(l);} }
    private int[] search(OnHeapHnswGraph g, VarPacked p, float[] q, int ef) throws IOException {
        final boolean f=p.fp32!=null;
        RandomVectorScorer qs=new RandomVectorScorer(){ public int maxOrd(){return f?p.fp32.length:p.levels.length;} public float score(int ord){ return f? -l2(q,p.fp32[ord]) : -asymDist(p,q,p.levels[ord]); } };
        TopKnnCollector col=new TopKnnCollector(ef,Integer.MAX_VALUE); HnswGraphSearcher.search(qs,col,new GV(g),null);
        ScoreDoc[] sd=col.topDocs().scoreDocs; int[] o=new int[sd.length]; for(int i=0;i<sd.length;i++)o[i]=sd[i].doc; return o;
    }
    private int[][] posAt(OnHeapHnswGraph g, VarPacked p, float[][] Q, int[][] gt, int ef) throws IOException {
        int[][] pos=new int[Q.length][K]; for(int qi=0;qi<Q.length;qi++){ int[] cand=search(g,p,Q[qi],ef); java.util.HashMap<Integer,Integer> rk=new java.util.HashMap<>(); for(int i=0;i<cand.length;i++)rk.put(cand[i],i); for(int t=0;t<K;t++) pos[qi][t]=rk.getOrDefault(gt[qi][t],-1); } return pos;
    }
    private double recallAt(int[][] pos,int from,int to,int depth){ double s=0;int c=0; for(int q=from;q<to;q++){ int h=0; for(int pp:pos[q]) if(pp>=0&&pp<depth)h++; s+=h/(double)K; c++; } return c==0?0:s/c; }

    // ===================== data + io =====================
    private int[][] groundTruth(float[][] X, float[][] Q){ int N=X.length; int[][] gt=new int[Q.length][K]; for(int qi=0;qi<Q.length;qi++){ float[] ex=new float[N]; for(int j=0;j<N;j++) ex[j]=l2(Q[qi],X[j]); gt[qi]=topIdx(ex,K); } return gt; }
    private float[][] materialize(String type,int dim,int N,long sd,int off) throws IOException {
        if(type.equals("sift128")){ Path repo=Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent(); return readFvecs(repo.resolve(BASE).toString(),off,N); }
        return gen(type,dim,N,new Random(sd));
    }
    private float[][] materializeQ(String type,int dim,long sd) throws IOException {
        if(type.equals("sift128")){ Path repo=Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent(); return readFvecs(repo.resolve(QUERY).toString(),0,NQ); }
        return gen(type,dim,NQ,new Random(sd+12345));
    }
    private static float[][] gen(String type,int d,int n,Random r){
        if(type.equals("manifold768")) return manifold(d,64,0.10f,n,r);
        float[][] X=new float[n][d]; for(int i=0;i<n;i++) for(int j=0;j<d;j++) X[i][j]=(float)r.nextGaussian(); return X; // iid128/iid384/iso768
    }
    private static float[][] manifold(int d,int r,float noise,int n,Random rng){ r=Math.min(r,d); float[][] basis=new float[r][d]; for(int a=0;a<r;a++)for(int j=0;j<d;j++)basis[a][j]=(float)rng.nextGaussian(); float[][] X=new float[n][d]; float inv=1f/(float)Math.sqrt(r); for(int i=0;i<n;i++){ float[] z=new float[r]; for(int k=0;k<r;k++)z[k]=(float)rng.nextGaussian(); for(int j=0;j<d;j++){ float s=0; for(int k=0;k<r;k++)s+=z[k]*basis[k][j]; X[i][j]=s*inv+(float)rng.nextGaussian()*noise; } } return X; }
    private static float l2(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static int[] topIdx(float[] sc,int m){ Integer[] idx=new Integer[sc.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(sc[x],sc[y])); int[] o=new int[m]; for(int i=0;i<m;i++)o[i]=idx[i]; return o; }
    private int countDone() throws IOException { Path p=outDir().resolve("step12_representations.csv"); if(!Files.exists(p)) return 0; int mx=-1; for(String ln:(Iterable<String>)Files.lines(p)::iterator){ if(ln.startsWith("seg,"))continue; int c=ln.indexOf(','); if(c>0){ try{ mx=Math.max(mx,Integer.parseInt(ln.substring(0,c))); }catch(Exception ignore){} } } return mx+1; }  // resume from last completed seg+1
    private Path outDir() throws IOException { Path o=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(o); return o; }
    private void writeFresh(Path dir,String name,String hdr) throws IOException { try(Writer w=Files.newBufferedWriter(dir.resolve(name))){ w.write(hdr); w.write("\n"); } }
    private void append(Path dir,String name,List<String> rows) throws IOException { try(Writer w=Files.newBufferedWriter(dir.resolve(name),StandardOpenOption.CREATE,StandardOpenOption.APPEND)){ for(String r:rows){ w.write(r); w.write("\n"); } } }
    private static float[][] readFvecs(String path,int skip,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; int seen=0; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; if(seen++<skip)continue; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++)v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
