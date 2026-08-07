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
 * Track 2, step 13 — SMART 1-BIT on REAL high-dim embeddings. Does the SIFT Hadamard-1-bit win
 * generalize to real 768/256-D embeddings, and can a RaBitQ-inspired co-designed 1-bit reach near-4-bit
 * candidate-generation quality (moving 4-bit segments to ~1 bit)?
 *
 * REAL datasets: fashion-mnist-784 (real 784-D images), nytimes-256 (real text embeddings) [downloaded
 * from ann-benchmarks]; iid-784 / iid-256 kept as ISOTROPIC CONTROLS. Primary metric = candidate
 * recall@10 BEFORE reranking. Every representation builds AND navigates on its OWN geometry.
 *
 * Representations: uniform-1/2/4-bit (baseline 1-bit already uses fp32-query ADC), Hadamard-1-bit C1
 * (padded, honest eff>1 bits) and C2 (dimension-preserving block-Hadamard, eff 1.0), RaBitQ-inspired
 * (centroid-residual unit-normalize + structured rotation + sign code + per-vector {norm, <o,ō>} scalars
 * + the co-designed L2 estimator), and fp32. Rotation is Fastfood-style (sign-flip + block-Hadamard),
 * NOT a full random matrix -> labelled RaBitQ-INSPIRED, not RaBitQ. Warm/one host; auto-resume.
 */
public class Track2SmartOneBitRealTests extends KNNTestCase {

    private static final int K = 10, M = 16, BEAM = 100;
    private static final int NEVAL = 200, NQ = 300;   // last NEVAL of NQ are held-out eval
    private static final int[] EFS = { 20, 50, 100, 200, 500 };
    private static final double TARGET = 0.95;
    private static final long SEED = 71L;
    private static final String REAL = "research/acorn/data/real/";

    // dataset: name, dim, N, isReal(fvecs prefix or "iid")
    static final Object[][] DS = {
        { "fmnist784", 784, 12000, "fmnist" },
        { "nytimes256", 256, 15000, "nytimes" },
        { "iid784", 784, 10000, "iid" },     // isotropic control
        { "iid256", 256, 12000, "iid" },     // isotropic control
    };
    static final String[] REPS = { "uniform1","uniform2","uniform4","had_c1_pad","had_c2_preserve","rabitq_insp","fp32" };

    public void testSmartOneBitReal() throws IOException {
        Path dir = outDir();
        String h = "dataset,dim,N,rep,eff_bits,code_bytes_per_vec,sidecar_bytes,total_bytes_per_vec,"
            + "ceil20,ceil50,ceil100,ceil200,ceil500,ceil_max,ef90,ef95,ef97,feasible95,build_ms,nn_overlap10";
        java.util.Set<String> done = doneKeys();
        if (done.isEmpty()) writeFresh(dir, "step13_real.csv", h);
        System.out.printf(java.util.Locale.ROOT, "[step13] resume: %d (dataset,rep) already done%n", done.size());
        for (int si = 0; si < DS.length; si++) {
            Object[] d = DS[si]; String name=(String)d[0]; int dim=(int)d[1], N=(int)d[2]; String src=(String)d[3];
            // skip dataset entirely if all reps done
            boolean any=false; for(String rp:REPS) if(!done.contains(name+"|"+rp)) any=true; if(!any) continue;
            float[][] X = load(src, dim, N, false); float[][] Q = load(src, dim, NQ, true);
            int[][] gt = groundTruth(X, Q);
            for (String rp : REPS) {
                if (done.contains(name+"|"+rp)) continue;
                Rep r = make(rp, X, si);
                String row = run(name,dim,N,rp, r, X,Q,gt);
                append(dir,"step13_real.csv", java.util.Collections.singletonList(row));
                System.out.printf(java.util.Locale.ROOT,"[step13] %s %s done%n",name,rp);
            }
        }
        System.out.println("[step13] DONE");
    }
    private Rep make(String rp, float[][] X, int si){
        switch(rp){ case "uniform1": return Rep.uniform(X,1); case "uniform2": return Rep.uniform(X,2); case "uniform4": return Rep.uniform(X,4);
            case "had_c1_pad": return Rep.hadamard(X,true,SEED+si); case "had_c2_preserve": return Rep.hadamard(X,false,SEED+si);
            case "rabitq_insp": return Rep.rabitq(X,SEED+si); default: return Rep.fp32(X); }
    }

    private String run(String ds,int dim,int N,String rep, Rep r, float[][] X, float[][] Q, int[][] gt) throws IOException {
        long t0=System.nanoTime(); OnHeapHnswGraph g=build(r,N); long buildMs=(System.nanoTime()-t0)/1_000_000;
        double[] ceil=new double[EFS.length]; double cmax=0; int[] efT={-1,-1,-1}; double[] targets={0.90,0.95,0.97};
        for(int e=0;e<EFS.length;e++){ int[][] pos=posAt(g,r,Q,gt,EFS[e]); ceil[e]=recallAt(pos,NQ-NEVAL,NQ,EFS[e]); cmax=Math.max(cmax,ceil[e]);
            for(int t=0;t<3;t++) if(efT[t]<0 && ceil[e]>=targets[t]) efT[t]=EFS[e]; }
        double nnov = neighbourOverlap(r, X, gt);   // top-10 overlap vs fp32 (flat, no graph) -- quantization fidelity
        StringBuilder cs=new StringBuilder(); for(int i=0;i<ceil.length;i++){ if(i>0)cs.append(","); cs.append(String.format(java.util.Locale.ROOT,"%.4f",ceil[i])); }
        return String.format(java.util.Locale.ROOT,"%s,%d,%d,%s,%.3f,%d,%d,%d,%s,%.4f,%d,%d,%d,%d,%d,%.4f",
            ds,dim,N,rep,r.effBits,r.codeBytes,r.sidecar,r.codeBytes+r.sidecar,cs,cmax,efT[0],efT[1],efT[2],efT[1]>0?1:0,buildMs,nnov);
    }

    // ================= Representation: unified scorer interface =================
    // Each Rep can score doc-doc (build) and query-doc (navigate) on ITS OWN consistent geometry.
    abstract static class Rep {
        double effBits; int codeBytes, sidecar, n, dim;
        abstract float symDist(int a, int b);            // build/prune
        abstract float asymDist(float[] qOrig, int b);   // navigate (raw query vector in)
        abstract float[] reconstruct(int i);             // for nn-overlap fidelity check (approx doc vector)

        static Rep fp32(float[][] X){ return new Rep(){ { n=X.length; dim=X[0].length; effBits=32; codeBytes=dim*4; }
            float symDist(int a,int b){ return l2(X[a],X[b]); } float asymDist(float[] q,int b){ return l2(q,X[b]); } float[] reconstruct(int i){ return X[i]; } }; }

        // uniform per-dim min/max b-bit, asymmetric fp32-query ADC (the current baseline family)
        static Rep uniform(float[][] X,int bits){ int n=X.length,d=X[0].length; int L=1<<bits;
            float[] mn=new float[d],mx=new float[d]; Arrays.fill(mn,Float.POSITIVE_INFINITY);Arrays.fill(mx,Float.NEGATIVE_INFINITY);
            for(float[] v:X)for(int j=0;j<d;j++){if(v[j]<mn[j])mn[j]=v[j];if(v[j]>mx[j])mx[j]=v[j];}
            byte[][] code=new byte[n][d]; float[][] rec=new float[d][L];
            for(int j=0;j<d;j++)for(int l=0;l<L;l++){float rng=mx[j]-mn[j];rec[j][l]=rng<=0?mn[j]:mn[j]+(float)l/(L-1)*rng;}
            for(int i=0;i<n;i++)for(int j=0;j<d;j++){float rng=mx[j]-mn[j];int lv=rng<=0?0:Math.round((X[i][j]-mn[j])/rng*(L-1));if(lv<0)lv=0;if(lv>L-1)lv=L-1;code[i][j]=(byte)lv;}
            return new Rep(){ { Rep.this0(this,n,d,(double)bits,(d*bits+7)/8,0); }
                float symDist(int a,int b){float s=0;for(int j=0;j<d;j++){float x=rec[j][code[a][j]&0xFF]-rec[j][code[b][j]&0xFF];s+=x*x;}return s;}
                float asymDist(float[] q,int b){float s=0;for(int j=0;j<d;j++){float x=q[j]-rec[j][code[b][j]&0xFF];s+=x*x;}return s;}
                float[] reconstruct(int i){float[] o=new float[d];for(int j=0;j<d;j++)o[j]=rec[j][code[i][j]&0xFF];return o;} };
        }

        // Hadamard-preconditioned 1-bit. padded=C1 (pad to pow2, eff=Dpad/d bits). !padded=C2 (block-Hadamard, dim-preserving, eff=1).
        static Rep hadamard(float[][] X, boolean padded, long sd){ int n=X.length,d=X[0].length;
            final int[] blocks = padded? new int[]{ pow2(d) } : blockSplit(d);
            final int Dt = sum(blocks);                          // padded: Dt=pow2(d)>=d ; preserve: Dt=d
            float[] sign=signs(Dt,sd);
            float[][] Y=new float[n][]; for(int i=0;i<n;i++) Y[i]=transform(X[i],blocks,sign);
            // per-dim symmetric 1-bit: level = sign(y); recon {-s_j,+s_j}, s_j=mean|y_j|
            float[] s=new float[Dt]; for(int i=0;i<n;i++)for(int j=0;j<Dt;j++)s[j]+=Math.abs(Y[i][j]); for(int j=0;j<Dt;j++)s[j]/=n;
            byte[][] bit=new byte[n][Dt]; for(int i=0;i<n;i++)for(int j=0;j<Dt;j++)bit[i][j]=(byte)(Y[i][j]>=0?1:0);
            double effB=(double)Dt/d;   // eff bits/ORIGINAL dim (padding honestly inflates)
            return new Rep(){ { Rep.this0(this,n,d,effB,(Dt+7)/8,0); }
                float rc(int i,int j){ return (bit[i][j]==1?s[j]:-s[j]); }
                float symDist(int a,int b){float t=0;for(int j=0;j<Dt;j++){float x=rc(a,j)-rc(b,j);t+=x*x;}return t;}
                float asymDist(float[] q,int b){ float[] yq=transform(q,blocks,sign); float t=0; for(int j=0;j<Dt;j++){float x=yq[j]-rc(b,j);t+=x*x;} return t; }
                float[] reconstruct(int i){ float[] o=new float[Dt]; for(int j=0;j<Dt;j++)o[j]=rc(i,j); return o; } };  // in rotated space (fidelity uses rotated q too)
        }

        // RaBitQ-inspired: residual = x - centroid; unit-normalize; structured rotation (Fastfood: sign+block-Hadamard);
        // sign code b; per-vector nx=||residual||, invDot=1/<o,ō> where ō=b/sqrt(Dt); L2 estimator (co-designed).
        static Rep rabitq(float[][] X, long sd){ int n=X.length,d=X[0].length; final int[] blocks=blockSplit(d); final int Dt=d; float[] sign=signs(Dt,sd);
            float[] cen=new float[d]; for(float[] v:X)for(int j=0;j<d;j++)cen[j]+=v[j]; for(int j=0;j<d;j++)cen[j]/=n;
            byte[][] bit=new byte[n][Dt]; float[] nx=new float[n], invDot=new float[n]; float inv=1f/(float)Math.sqrt(Dt);
            for(int i=0;i<n;i++){ float[] r=new float[d]; float nn=0; for(int j=0;j<d;j++){ r[j]=X[i][j]-cen[j]; nn+=r[j]*r[j]; } nn=(float)Math.sqrt(nn); nx[i]=nn;
                float[] u = nn>0? scale(r,1f/nn): r; float[] y=transform(u,blocks,sign);  // rotated unit residual
                float dot=0; for(int j=0;j<Dt;j++){ bit[i][j]=(byte)(y[j]>=0?1:0); dot += Math.abs(y[j]); } dot*=inv;   // <o,ō>=(1/sqrt Dt)Σ|y_j|
                invDot[i]= dot>1e-6? 1f/dot : 0f;
            }
            return new Rep(){ { Rep.this0(this,n,d,1.0,(Dt+7)/8, 8); }   // 1-bit code + 8B/vec (nx + invDot)
                // estimate <q_unit, x_unit> = <ō, yq_unit>*invDot ; dist^2 = nq^2 + nx^2 - 2 nq nx est
                float est(float[] yqUnit, int b){ float dot=0; for(int j=0;j<Dt;j++) dot += (bit[b][j]==1? yqUnit[j] : -yqUnit[j]); dot*= (1f/(float)Math.sqrt(Dt)); return dot*invDot[b]; }
                float[] qunit(float[] q){ float[] r=new float[d]; float nn=0; for(int j=0;j<d;j++){r[j]=q[j]-cen[j];nn+=r[j]*r[j];} nn=(float)Math.sqrt(nn); float[] u=nn>0?scale(r,1f/nn):r; float[] y=transform(u,blocks,sign); float[] out=new float[Dt+1]; System.arraycopy(y,0,out,0,Dt); out[Dt]=nn; return out; }
                float asymDist(float[] q,int b){ float[] qq=qunit(q); float nq=qq[Dt]; float[] yq=qq; float e=est(yq,b); float dd=nq*nq+nx[b]*nx[b]-2*nq*nx[b]*e; return dd<0?0:dd; }
                // GRAPH-CONSISTENT: build uses the SAME asymmetric estimator with doc a's original vector as the query
                float symDist(int a,int b){ return asymDist(X[a], b); }
                float[] reconstruct(int i){ float[] o=new float[Dt]; float s=nx[i]/(float)Math.sqrt(Dt); for(int j=0;j<Dt;j++)o[j]=(bit[i][j]==1?s:-s); return o; } };
        }
        // helper to set fields from anonymous subclass ctors
        static void this0(Rep r,int n,int d,double eff,int codeB,int side){ r.n=n; r.dim=d; r.effBits=eff; r.codeBytes=codeB; r.sidecar=side; }
    }

    // ---- transforms ----
    private static int sum(int[] a){ int s=0; for(int x:a) s+=x; return s; }
    private static int pow2(int d){ int p=1; while(p<d)p<<=1; return p; }
    private static int[] blockSplit(int d){ List<Integer> b=new ArrayList<>(); int rem=d; while(rem>0){ int p=1; while(p*2<=rem)p<<=1; b.add(p); rem-=p; } int[] o=new int[b.size()]; for(int i=0;i<o.length;i++)o[i]=b.get(i); return o; }
    private static float[] signs(int D,long sd){ Random r=new Random(sd+13); float[] s=new float[D]; for(int j=0;j<D;j++)s[j]=r.nextBoolean()?1f:-1f; return s; }
    /** apply sign-flip then block-Hadamard (each block FWHT-normalized). Output length = sum(blocks). Pads input with 0. */
    private static float[] transform(float[] x, int[] blocks, float[] sign){ int Dt=0; for(int b:blocks)Dt+=b; float[] y=new float[Dt]; int d=x.length;
        for(int j=0;j<Dt;j++) y[j] = (j<d? x[j]:0f) * sign[j];
        int off=0; for(int bl:blocks){ fwht(y,off,bl); float inv=1f/(float)Math.sqrt(bl); for(int j=off;j<off+bl;j++)y[j]*=inv; off+=bl; } return y; }
    private static void fwht(float[] a,int off,int n){ for(int len=1;len<n;len<<=1){ for(int i=off;i<off+n;i+=len<<1){ for(int j=i;j<i+len;j++){ float u=a[j],v=a[j+len]; a[j]=u+v; a[j+len]=u-v; } } } }
    private static float[] scale(float[] x,float s){ float[] o=new float[x.length]; for(int i=0;i<x.length;i++)o[i]=x[i]*s; return o; }

    // ---- fidelity: flat top-10 overlap of rep-distance vs fp32 (no graph) ----
    private double neighbourOverlap(Rep r, float[][] X, int[][] gt){ int probe=Math.min(30, gt.length); double ov=0;
        for(int qi=0; qi<probe; qi++){ // use eval queries against docs via asymDist; compare to fp32 gt
            float[] q = evalQ[qi];
            float[] sc=new float[X.length]; for(int j=0;j<X.length;j++) sc[j]=r.asymDist(q,j); int[] top=topIdx(sc,K);
            java.util.HashSet<Integer> t=new java.util.HashSet<>(); for(int x:top)t.add(x); int h=0; for(int x:gt[NQ-NEVAL+qi]) if(t.contains(x))h++; ov+=h/(double)K; }
        return ov/probe;
    }
    private float[][] evalQ;   // set per dataset for overlap probe

    // ================= HNSW =================
    private OnHeapHnswGraph build(Rep r,int N) throws IOException {
        RandomVectorScorerSupplier sup=new RandomVectorScorerSupplier(){ public UpdateableRandomVectorScorer scorer(){ return new UpdateableRandomVectorScorer(){ int cur=0; public int maxOrd(){return N;} public void setScoringOrdinal(int o){cur=o;} public float score(int j){ return -r.symDist(cur,j); } }; } public RandomVectorScorerSupplier copy(){return this;} };
        return HnswGraphBuilder.create(sup,M,BEAM,SEED).build(N);
    }
    static final class GV extends HnswGraph { final OnHeapHnswGraph g; int[] cur=new int[0]; int sz=0,up=0; GV(OnHeapHnswGraph g){this.g=g;} public void seek(int l,int nd){NeighborArray na=g.getNeighbors(l,nd);cur=na.nodes();sz=na.size();up=0;} public int size(){return g.size();} public int nextNeighbor(){return up<sz?cur[up++]:DocIdSetIterator.NO_MORE_DOCS;} public int numLevels()throws IOException{return g.numLevels();} public int maxConn(){return g.maxConn();} public int entryNode()throws IOException{return g.entryNode();} public int neighborCount(){return sz;} public HnswGraph.NodesIterator getNodesOnLevel(int l)throws IOException{return g.getNodesOnLevel(l);} }
    private int[][] posAt(OnHeapHnswGraph g, Rep r, float[][] Q, int[][] gt, int ef) throws IOException {
        int[][] pos=new int[Q.length][K]; for(int qi=0;qi<Q.length;qi++){ final float[] qv=Q[qi]; RandomVectorScorer qs=new RandomVectorScorer(){ public int maxOrd(){return r.n;} public float score(int ord){ return -r.asymDist(qv,ord); } };
            TopKnnCollector col=new TopKnnCollector(ef,Integer.MAX_VALUE); HnswGraphSearcher.search(qs,col,new GV(g),null); ScoreDoc[] sd=col.topDocs().scoreDocs;
            java.util.HashMap<Integer,Integer> rk=new java.util.HashMap<>(); for(int i=0;i<sd.length;i++)rk.put(sd[i].doc,i); for(int t=0;t<K;t++)pos[qi][t]=rk.getOrDefault(gt[qi][t],-1); } return pos;
    }
    private double recallAt(int[][] pos,int from,int to,int depth){ double s=0;int c=0; for(int q=from;q<to;q++){ int h=0; for(int pp:pos[q]) if(pp>=0&&pp<depth)h++; s+=h/(double)K; c++; } return c==0?0:s/c; }

    // ================= data =================
    private int[][] groundTruth(float[][] X, float[][] Q){ int N=X.length; int[][] gt=new int[Q.length][K]; for(int qi=0;qi<Q.length;qi++){ float[] ex=new float[N]; for(int j=0;j<N;j++) ex[j]=l2(Q[qi],X[j]); gt[qi]=topIdx(ex,K); } evalQ=new float[NEVAL][]; for(int i=0;i<NEVAL;i++) evalQ[i]=Q[NQ-NEVAL+i]; return gt; }
    private float[][] load(String src,int dim,int n,boolean query) throws IOException {
        if(src.equals("iid")){ Random r=new Random(SEED+(query?999:1)+dim); float[][] X=new float[n][dim]; for(int i=0;i<n;i++)for(int j=0;j<dim;j++)X[i][j]=(float)r.nextGaussian(); return X; }
        Path repo=Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
        String f = repo.resolve(REAL+src+(query?"_query.fvecs":"_base.fvecs")).toString(); return readFvecs(f,0,n);
    }
    private static float l2(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static int[] topIdx(float[] sc,int m){ Integer[] idx=new Integer[sc.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(sc[x],sc[y])); int[] o=new int[m]; for(int i=0;i<m;i++)o[i]=idx[i]; return o; }
    private java.util.Set<String> doneKeys() throws IOException { java.util.Set<String> s=new java.util.HashSet<>(); Path p=outDir().resolve("step13_real.csv"); if(!Files.exists(p))return s; for(String ln:(Iterable<String>)Files.lines(p)::iterator){ if(ln.startsWith("dataset,"))continue; String[] c=ln.split(","); if(c.length>=4) s.add(c[0]+"|"+c[3]); } return s; }
    private Path outDir() throws IOException { Path o=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(o); return o; }
    private void writeFresh(Path dir,String name,String hdr) throws IOException { try(Writer w=Files.newBufferedWriter(dir.resolve(name))){ w.write(hdr); w.write("\n"); } }
    private void append(Path dir,String name,List<String> rows) throws IOException { try(Writer w=Files.newBufferedWriter(dir.resolve(name),StandardOpenOption.CREATE,StandardOpenOption.APPEND)){ for(String r:rows){ w.write(r); w.write("\n"); } } }
    private static float[][] readFvecs(String path,int skip,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; int seen=0; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; if(seen++<skip)continue; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++)v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
