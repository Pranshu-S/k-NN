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
 * Track 2, step 3 — MINIMAL per-vector corrected bound estimator, plugged into the UNCHANGED
 * adaptive policy (all-remaining stop test + Bonferroni). Only the (pred_j, sigma_j) seam changes.
 *
 * Corrected estimator (geometry-derived, not a learned model):
 *   pred_j   = ||q||^2 + ||x_j||^2 - 2 * (q . xhat_j)        // xhat = 1-bit reconstruction (below/above means)
 *   residual = exact - pred = -2 q.(x_j - xhat_j)            // the irreducible dot-product quant error
 *   sigma_j  = c * 2*||q||*||x_j - xhat_j|| / sqrt(d)        // E[(q.r)^2]=||q||^2||r||^2/d ; c fit for coverage
 * Metadata per vector: ||x_j||^2 and ||x_j - xhat_j||  = 2 floats = 8 bytes. Query feature: ||q||.
 *
 * Calibration (a,b,c) fit on a HELD-OUT split (SIFT learn / synthetic cal-seed); evaluation on a
 * disjoint set (SIFT query / synthetic eval-seed). No leakage; fixed seeds.
 *
 * Compares bound estimators {ADC, Hamming(global sigma), Corrected(per-candidate sigma)} under a
 * FIXED ADC candidate ordering, so only estimator quality varies. Policy/correction/metrics unchanged.
 */
public class Track2CorrectedEstimatorTests extends KNNTestCase {

    private static final int N = 10000, NCAL = 80, NEVAL = 80, K = 10, POOL = 1000;
    private static final int MIN_RERANK = K, MAX_RERANK = POOL, CHECK_STEP = 2;
    private static final double CONFIDENCE = 0.99;
    private static final long SEED = 21L;
    private static final String BASE = "research/acorn/data/sift/sift_base.fvecs";
    private static final String LEARN = "research/acorn/data/sift/sift_learn.fvecs";
    private static final String QUERY = "research/acorn/data/sift/sift_query.fvecs";

    enum Bound { ADC, HAMMING, CORRECTED }

    public void testCorrectedEstimator() throws IOException {
        List<String> pq = new ArrayList<>();
        pq.add("dataset,dim,bound_estimator,correction,query_id,corrected_pred,sigma_j_mean,residual_rmse,std_resid_std,"
            + "adaptive_depth,oracle_depth,best_fixed_depth,stopped_by_bound,false_safe,recall,difficulty");
        List<String> agg = new ArrayList<>();
        agg.add("dataset,dim,bound_estimator,correction,metadata_bytes_per_vec,global_sigma_norm,boundary_sigma_norm,"
            + "cover90,cover95,cover99,cover95_boundary,rmse,mae,false_safe_rate,recall,mean_depth,p50_depth,p95_depth,p99_depth,"
            + "max_fallback_rate,mean_oversample,best_fixed_depth,oracle_mean_depth,savings_vs_best_fixed,gap_to_oracle");

        String[][] datasets = { {"sift_real","128"}, {"isotropic","768"}, {"clustered","128"} };
        for (String[] ds : datasets) {
            String dt = ds[0]; int d = Integer.parseInt(ds[1]);
            runDataset(pq, agg, dt, d);
        }
        write("track2_corrected_perquery.csv", pq);
        write("track2_corrected_summary.csv", agg);
    }

    private void runDataset(List<String> pq, List<String> agg, String dt, int d) throws IOException {
        // ---- load docs + disjoint cal/eval queries ----
        float[][] X, Qcal, Qeval; int dim;
        if (dt.equals("sift_real")) {
            Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
            X = readFvecs(repo.resolve(BASE).toString(), N); Qcal = readFvecs(repo.resolve(LEARN).toString(), NCAL); Qeval = readFvecs(repo.resolve(QUERY).toString(), NEVAL); dim = X[0].length;
        } else { Random r = new Random(SEED); X = gen(dt, d, N, r); Qcal = gen(dt, d, NCAL, new Random(SEED+1)); Qeval = gen(dt, d, NEVAL, new Random(SEED+2)); dim = d; }

        // ---- train quantizer, reconstruct xhat, compute per-vector metadata ----
        OneBitScalarQuantizer quant = new OneBitScalarQuantizer(false);
        final float[][] Xf = X;
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(X.length, false) {
            @Override public float[] getVectorAtThePosition(int p) { return Xf[p]; }
            @Override public void resetVectorValues() { }
        };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) quant.train(req);
        float[] thr = st.getMeanThresholds(), above = st.getAboveThresholdMeans(), below = st.getBelowThresholdMeans();
        byte[][] bits = new byte[N][]; float[][] xhat = new float[N][dim]; double[] normSq = new double[N], residNorm = new double[N];
        for (int j = 0; j < N; j++) {
            BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); quant.quantize(X[j].clone(), st, o); bits[j] = o.getQuantizedVector().clone();
            double ns = 0, rn = 0; for (int i = 0; i < dim; i++) { float xh = X[j][i] > thr[i] ? above[i] : below[i]; xhat[j][i] = xh; ns += (double) X[j][i]*X[j][i]; double e = X[j][i]-xh; rn += e*e; }
            normSq[j] = ns; residNorm[j] = Math.sqrt(rn);
        }
        double sqrtD = Math.sqrt(dim);

        // ---- calibrate on CAL queries (a,b for each estimator; c for corrected sigma) ----
        Cal cAdc = calibrateScore(quant, st, bits, X, Qcal, true);
        Cal cHam = calibrateScore(quant, st, bits, X, Qcal, false);
        double[] corC = calibrateCorrected(quant, st, X, xhat, normSq, residNorm, sqrtD, Qcal);  // {a,b,c,sigmaFloor,globalResidStd}

        // ---- evaluate on EVAL queries ----
        for (Bound b : new Bound[]{Bound.ADC, Bound.HAMMING, Bound.CORRECTED})
            for (String corr : new String[]{"none","bonferroni"})
                evaluate(pq, agg, dt, dim, b, corr, quant, st, bits, X, xhat, normSq, residNorm, sqrtD, Qeval, cAdc, cHam, corC);
    }

    static final class Cal { double a, b, sigma; }
    private Cal calibrateScore(OneBitScalarQuantizer q, OneBitScalarQuantizationState st, byte[][] bits, float[][] X, float[][] Qc, boolean adc) throws IOException {
        double n=0,se=0,st_=0,see=0,set=0; List<double[]> pairs = new ArrayList<>();
        for (int qi = 0; qi < Qc.length; qi++) {
            float[] qadc = Qc[qi].clone(); if (adc) q.transformWithADC(qadc, st, SpaceType.L2);
            byte[] qb = adc ? null : quantizeQ(q, Qc[qi], st);
            for (int j = 0; j < N; j += 7) { double e = adc ? KNNScoringUtil.l2SquaredADC(qadc, bits[j]) : hamming(qb, bits[j]); double t = l2sq(Qc[qi], X[j]); pairs.add(new double[]{e,t}); }
        }
        for (double[] p : pairs) { n++; se+=p[0]; st_+=p[1]; see+=p[0]*p[0]; set+=p[0]*p[1]; }
        double den=n*see-se*se; Cal c=new Cal(); c.a=den==0?0:(n*set-se*st_)/den; c.b=(st_-c.a*se)/n;
        double v=0; for (double[] p : pairs){ double r=p[1]-(c.a*p[0]+c.b); v+=r*r; } c.sigma=Math.sqrt(v/n); return c;
    }
    private double[] calibrateCorrected(OneBitScalarQuantizer q, OneBitScalarQuantizationState st, float[][] X, float[][] xhat, double[] normSq, double[] residNorm, double sqrtD, float[][] Qc) {
        List<double[]> pairs = new ArrayList<>();   // {correctedRaw, exact, rawSigma}
        for (int qi = 0; qi < Qc.length; qi++) { double qn2 = dot(Qc[qi], Qc[qi]); double qnorm = Math.sqrt(qn2);
            for (int j = 0; j < N; j += 7) { double raw = qn2 + normSq[j] - 2.0*dot(Qc[qi], xhat[j]); double e = l2sq(Qc[qi], X[j]); double rs = 2.0*qnorm*residNorm[j]/sqrtD; pairs.add(new double[]{raw,e,rs}); } }
        double n=0,se=0,st_=0,see=0,set=0; for (double[] p : pairs){ n++; se+=p[0]; st_+=p[1]; see+=p[0]*p[0]; set+=p[0]*p[1]; }
        double den=n*see-se*se; double a=den==0?0:(n*set-se*st_)/den, b=(st_-a*se)/n;
        // c = std of residual / rawSigma  (so standardized residuals ~ unit variance)
        double sv=0, cs=0; int cn=0; double gv=0; for (double[] p : pairs){ double r=p[1]-(a*p[0]+b); gv+=r*r; if (p[2]>1e-9){ double s=r/p[2]; sv+=s*s; cn++; } }
        double c = cn>0 ? Math.sqrt(sv/cn) : 1.0; double globalResidStd = Math.sqrt(gv/n);
        double sigmaFloor = 0.2 * globalResidStd;
        return new double[]{a,b,c,sigmaFloor,globalResidStd};
    }

    private void evaluate(List<String> pq, List<String> agg, String dt, int dim, Bound bound, String corr,
                          OneBitScalarQuantizer q, OneBitScalarQuantizationState st, byte[][] bits, float[][] X, float[][] xhat,
                          double[] normSq, double[] residNorm, double sqrtD, float[][] Qeval, Cal cAdc, Cal cHam, double[] corC) throws IOException {
        double alpha = 1.0 - CONFIDENCE;
        double distStd = 0; { // rough distance-scale for normalizing sigma reporting (from first eval query)
            float[] e0 = new float[N]; for (int j=0;j<N;j++) e0[j]=l2sq(Qeval[0],X[j]); distStd = std(e0); }
        int nEval = Qeval.length;
        double[] margins = new double[nEval];
        for (int qi=0; qi<nEval; qi++){ float[] ex=new float[N]; for(int j=0;j<N;j++) ex[j]=l2sq(Qeval[qi],X[j]); Arrays.sort(ex); margins[qi]=ex[K-1]-ex[K-2]; }
        double[] ms=margins.clone(); Arrays.sort(ms); double t1=ms[nEval/3], t2=ms[2*nEval/3];

        int nFalse=0, nMax=0; double recSum=0; long depthSum=0, oracleSum=0, bestFixedSum=0;
        int[] depths=new int[nEval];
        double rmseAcc=0, maeAcc=0; long resN=0; int cov90=0,cov95=0,cov99=0,covTot=0, covB95=0,covBTot=0; double sigSum=0; long sigN=0; double sigBSum=0; long sigBN=0;
        int[] FIXED={10,20,30,50,100};
        for (int qi=0; qi<nEval; qi++) {
            float[] q_=Qeval[qi]; double qn2=dot(q_,q_), qnorm=Math.sqrt(qn2);
            // scores over all N (ranking = ADC)
            float[] aSc=new float[N], hSc=new float[N], ex=new float[N];
            float[] qadc=q_.clone(); q.transformWithADC(qadc, st, SpaceType.L2); byte[] qb=quantizeQ(q,q_,st);
            for (int j=0;j<N;j++){ aSc[j]=KNNScoringUtil.l2SquaredADC(qadc,bits[j]); hSc[j]=hamming(qb,bits[j]); ex[j]=l2sq(q_,X[j]); }
            int[] cand=topIndices(aSc,POOL);
            java.util.HashSet<Integer> trueTopK=new java.util.HashSet<>(); { int[] g=topIndices(ex,K); for(int id:g) trueTopK.add(id); }
            int[] poolByExact=topExactWithin(cand,ex,K); java.util.HashSet<Integer> poolTopK=new java.util.HashSet<>(); for(int id:poolByExact) poolTopK.add(id);
            // pred + sigma for the chosen bound estimator over the pool
            double[] pred=new double[POOL], sigma=new double[POOL];
            float[] exSorted=ex.clone(); Arrays.sort(exSorted); double bnd=exSorted[K-1];
            for (int r=0;r<POOL;r++){ int j=cand[r];
                if (bound==Bound.ADC){ pred[r]=cAdc.a*aSc[j]+cAdc.b; sigma[r]=cAdc.sigma; }
                else if (bound==Bound.HAMMING){ pred[r]=cHam.a*hSc[j]+cHam.b; sigma[r]=cHam.sigma; }
                else { double raw=qn2+normSq[j]-2.0*dot(q_,xhat[j]); pred[r]=corC[0]*raw+corC[1]; sigma[r]=Math.max(corC[2]*2.0*qnorm*residNorm[j]/sqrtD, corC[3]); }
                // coverage bookkeeping
                double resid=ex[j]-pred[r]; double s=sigma[r]>0?resid/sigma[r]:0; covTot++; if(Math.abs(s)<=1.645)cov90++; if(Math.abs(s)<=1.960)cov95++; if(Math.abs(s)<=2.576)cov99++;
                rmseAcc+=resid*resid; maeAcc+=Math.abs(resid); resN++; sigSum+=sigma[r]; sigN++;
                if (Math.abs(ex[j]-bnd)<=0.05*Math.abs(bnd)+1e-6){ covBTot++; if(Math.abs(s)<=1.960)covB95++; sigBSum+=sigma[r]; sigBN++; }
            }
            // oracle + best-fixed
            int oracle=K; for(int r=0;r<POOL;r++) if(poolTopK.contains(cand[r])) oracle=r+1;
            double[] fixedRec=new double[FIXED.length]; for(int fi=0;fi<FIXED.length;fi++){ java.util.HashSet<Integer> gf=topExactSet(cand,ex,Math.min(FIXED[fi],POOL),K); fixedRec[fi]=overlap(gf,trueTopK)/(double)K; }
            // adaptive policy (UNCHANGED): all-remaining stop test with per-candidate sigma
            double[] heap=new double[K]; int hs=0; int depth=MAX_RERANK; boolean stopped=false;
            for (int r=1;r<=MAX_RERANK;r++){ double e=ex[cand[r-1]]; if(hs<K){heap[hs++]=e; if(hs==K)buildMax(heap);} else if(e<heap[0]){heap[0]=e; siftDown(heap,K);}
                if (r>=MIN_RERANK && r%CHECK_STEP==0 && hs==K){ int m=POOL-r; double z=zFor(alpha,m,corr); double exactK=heap[0];
                    double lb=Double.POSITIVE_INFINITY; for(int j=r;j<POOL;j++){ double v=pred[j]-z*sigma[j]; if(v<lb)lb=v; }   // min over ALL remaining
                    if (lb>exactK){ depth=r; stopped=true; break; } } }
            if(!stopped)nMax++;
            java.util.HashSet<Integer> got=topExactSet(cand,ex,depth,K);
            double rec=overlap(got,trueTopK)/(double)K; boolean falseSafe=stopped && !got.equals(poolTopK);
            if(falseSafe)nFalse++;
            // best fixed at equal recall (this query's adaptive recall)
            int bestFixed=FIXED[FIXED.length-1]; for(int fi=0;fi<FIXED.length;fi++) if(fixedRec[fi]>=rec-1e-9){ bestFixed=FIXED[fi]; break; }
            recSum+=rec; depthSum+=depth; oracleSum+=oracle; bestFixedSum+=bestFixed; depths[qi]=depth;
            int diff=margins[qi]<=t1?0:(margins[qi]<=t2?1:2);
            pq.add(String.format(java.util.Locale.ROOT,"%s,%d,%s,%s,%d,%.4g,%.4g,%.4g,%.3f,%d,%d,%d,%b,%b,%.3f,%s",
                dt,dim,bound,corr,qi, pred[Math.min(K-1,POOL-1)], sigSumForQuery(sigma), rmseForQuery(pred,ex,cand), 0.0,
                depth,oracle,bestFixed,stopped,falseSafe,rec, diff==0?"hard":diff==1?"medium":"easy"));
        }
        double meanDepth=depthSum/(double)nEval, oracleMean=oracleSum/(double)nEval, bestFixedMean=bestFixedSum/(double)nEval;
        int[] ds=depths.clone(); Arrays.sort(ds);
        double rmse=Math.sqrt(rmseAcc/resN), mae=maeAcc/resN;
        agg.add(String.format(java.util.Locale.ROOT,"%s,%d,%s,%s,%d,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4g,%.4g,%.4f,%.4f,%.1f,%d,%d,%d,%.4f,%.2f,%d,%.1f,%.3f,%.1f",
            dt,dim,bound,corr, bound==Bound.CORRECTED?8:0,
            distStd>0?(sigSum/sigN)/distStd:0, (sigBN>0&&distStd>0)?(sigBSum/sigBN)/distStd:0,
            cov90/(double)covTot, cov95/(double)covTot, cov99/(double)covTot, covBTot>0?covB95/(double)covBTot:0,
            rmse, mae, nFalse/(double)nEval, recSum/nEval, meanDepth, ds[nEval/2], ds[(int)(0.95*nEval)], ds[(int)(0.99*nEval)],
            nMax/(double)nEval, meanDepth/K, (int)Math.round(bestFixedMean), oracleMean, 1.0-meanDepth/Math.max(1,bestFixedMean), meanDepth-oracleMean));
        System.out.printf(java.util.Locale.ROOT,"%-10s d=%d %-9s %-10s | falseSafe=%.3f recall=%.3f meanDepth=%.0f (oracle=%.0f bestFixed=%.0f) sigma/dist=%.2f cover95=%.2f%n",
            dt,dim,bound,corr,nFalse/(double)nEval,recSum/nEval,meanDepth,oracleMean,bestFixedMean, distStd>0?(sigSum/sigN)/distStd:0, cov95/(double)covTot);
    }

    private static double sigSumForQuery(double[] s){ double a=0; for(double x:s)a+=x; return a/s.length; }
    private static double rmseForQuery(double[] pred, float[] ex, int[] cand){ double a=0; for(int r=0;r<pred.length;r++){ double d=ex[cand[r]]-pred[r]; a+=d*d; } return Math.sqrt(a/pred.length); }

    // ---- z + inverse normal (Acklam) ----
    private static double zFor(double alphaQ, int m, String corr){ double a=corr.equals("bonferroni")?alphaQ/Math.max(1,m):alphaQ; return invNorm(1.0-a/2.0); }
    private static double invNorm(double p){ if(p<=0)return -38; if(p>=1)return 38;
        final double[] a={-3.969683028665376e+01,2.209460984245205e+02,-2.759285104469687e+02,1.383577518672690e+02,-3.066479806614716e+01,2.506628277459239e+00};
        final double[] b={-5.447609879822406e+01,1.615858368580409e+02,-1.556989798598866e+02,6.680131188771972e+01,-1.328068155288572e+01};
        final double[] c={-7.784894002430293e-03,-3.223964580411365e-01,-2.400758277161838e+00,-2.549732539343734e+00,4.374664141464968e+00,2.938163982698783e+00};
        final double[] d={7.784695709041462e-03,3.224671290700398e-01,2.445134137142996e+00,3.754408661907416e+00};
        double pl=0.02425,ph=1-pl,q,r; if(p<pl){q=Math.sqrt(-2*Math.log(p));return(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);} if(p<=ph){q=p-0.5;r=q*q;return(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q/(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);} q=Math.sqrt(-2*Math.log(1-p));return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])/((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }

    // ---- helpers ----
    private static byte[] quantizeQ(OneBitScalarQuantizer q, float[] v, OneBitScalarQuantizationState st){ BinaryQuantizationOutput o=new BinaryQuantizationOutput(1); q.quantize(v.clone(),st,o); return o.getQuantizedVector().clone(); }
    private static int[] topIndices(float[] s,int m){ Integer[] idx=new Integer[s.length]; for(int i=0;i<idx.length;i++)idx[i]=i; Arrays.sort(idx,(x,y)->Float.compare(s[x],s[y])); int[] o=new int[Math.min(m,idx.length)]; for(int i=0;i<o.length;i++)o[i]=idx[i]; return o; }
    private static int[] topExactWithin(int[] cand,float[] ex,int k){ Integer[] idx=new Integer[cand.length]; for(int i=0;i<idx.length;i++)idx[i]=cand[i]; Arrays.sort(idx,(x,y)->Float.compare(ex[x],ex[y])); int[] o=new int[Math.min(k,idx.length)]; for(int i=0;i<o.length;i++)o[i]=idx[i]; return o; }
    private static java.util.HashSet<Integer> topExactSet(int[] cand,float[] ex,int depth,int k){ Integer[] idx=new Integer[Math.min(depth,cand.length)]; for(int i=0;i<idx.length;i++)idx[i]=cand[i]; Arrays.sort(idx,(x,y)->Float.compare(ex[x],ex[y])); java.util.HashSet<Integer> s=new java.util.HashSet<>(); for(int i=0;i<k&&i<idx.length;i++)s.add(idx[i]); return s; }
    private static int overlap(java.util.HashSet<Integer> a,java.util.HashSet<Integer> b){ int c=0; for(int x:a) if(b.contains(x))c++; return c; }
    private static void buildMax(double[] h){ for(int i=h.length/2-1;i>=0;i--)sd(h,i,h.length); }
    private static void siftDown(double[] h,int n){ sd(h,0,n); }
    private static void sd(double[] h,int i,int n){ while(true){ int l=2*i+1,r=2*i+2,m=i; if(l<n&&h[l]>h[m])m=l; if(r<n&&h[r]>h[m])m=r; if(m==i)break; double t=h[i];h[i]=h[m];h[m]=t;i=m; } }
    private static float l2sq(float[] a,float[] b){ float s=0; for(int i=0;i<a.length;i++){ float x=a[i]-b[i]; s+=x*x; } return s; }
    private static double dot(float[] a,float[] b){ double s=0; for(int i=0;i<a.length;i++) s+=(double)a[i]*b[i]; return s; }
    private static int hamming(byte[] a,byte[] b){ int s=0; for(int i=0;i<a.length;i++) s+=Integer.bitCount((a[i]^b[i])&0xFF); return s; }
    private static double std(float[] v){ double m=0; for(float x:v)m+=x; m/=v.length; double s=0; for(float x:v)s+=(x-m)*(x-m); return Math.sqrt(s/v.length); }
    private void write(String name,List<String> rows) throws IOException { Path out=Paths.get(System.getProperty("user.dir"),"research","track2_adaptive_rescore","results"); Files.createDirectories(out); try(Writer w=Files.newBufferedWriter(out.resolve(name))){ for(String r:rows){ w.write(r); w.write("\n"); } } System.out.println("WROTE "+rows.size()+" -> "+out.resolve(name)); }
    private static float[][] gen(String kind,int d,int n,Random rng){ float[][] X=new float[n][d];
        if(kind.equals("isotropic")){ for(int i=0;i<n;i++) for(int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); }
        else if(kind.equals("clustered")){ int ncl=32; float[][] c=new float[ncl][d]; for(int a=0;a<ncl;a++) for(int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f; for(int i=0;i<n;i++){ int a=rng.nextInt(ncl); for(int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } }
        else throw new IllegalArgumentException(kind); return X; }
    private static float[][] readFvecs(String path,int max) throws IOException { List<float[]> out=new ArrayList<>(); try(RandomAccessFile f=new RandomAccessFile(path,"r")){ byte[] h=new byte[4]; while(out.size()<max&&f.getFilePointer()<f.length()){ if(f.read(h)!=4)break; int dim=ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf=new byte[4*dim]; if(f.read(buf)!=4*dim)break; ByteBuffer bb=ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v=new float[dim]; for(int j=0;j<dim;j++) v[j]=bb.getFloat(); out.add(v); } } return out.toArray(new float[0][]); }
}
