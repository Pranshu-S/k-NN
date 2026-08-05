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
import java.util.List;
import java.util.Random;

/**
 * Track 2 — Adaptive Rescoring, STEP 1 (mandated): is the QFrame quantized score a CALIBRATED
 * distance estimator that can provide VALID error bounds? Do NOT assume it is.
 *
 * A bound-based adaptive policy ("include candidate i while best-possible(i) <= worst-possible(kth)")
 * needs, per candidate: an estimated distance AND a trustworthy uncertainty bound. This harness
 * measures whether the shipping score supports that:
 *   - affine calibration  true_L2 ~= a*estimate + b      (slope, intercept)
 *   - residual std (normalized by exact std)             (how noisy the estimate is)
 *   - residual skew / excess kurtosis                    (is the error Gaussian? bounds assume it)
 *   - homoscedasticity: residual std in low/high estimate bins (is a single sigma valid?)
 *   - bound COVERAGE: empirical P(|true - pred| <= z*sigma) vs nominal 90/95/99%
 *       under-coverage => the bound is UNSAFE (would skip rerank on truly-close candidates).
 *
 * Estimators (REAL code): symmetric 1-bit Hamming (baseline) and ADC (l2SquaredADC).
 * If the score cannot provide valid bounds, this quantifies the dependency Track 2 must add
 * (per-vector correction metadata / a better estimator), per the plan.
 */
public class Track2CalibrationTests extends KNNTestCase {

    private static final int N = 5000, NQ = 100, SAMPLES_PER_Q = 2000;
    private static final int[] DIMS = { 128, 768, 1536 };
    private static final String[] DATATYPES = { "isotropic", "anisotropic", "clustered" };
    private static final long SEED = 7L;
    private static final String SIFT = "research/acorn/data/sift/sift_base.fvecs";
    private static final String SIFT_Q = "research/acorn/data/sift/sift_query.fvecs";

    public void testQFrameCalibration() throws IOException {
        List<String> rows = new ArrayList<>();
        rows.add("estimator,datatype,dim,slope_a,intercept_b,resid_std_norm,skew,exkurt,sigma_lowbin,sigma_highbin,cover90,cover95,cover99");
        for (String dt : DATATYPES) for (int d : DIMS) {
            Random rng = new Random(SEED);
            float[][] X = gen(dt, d, N, rng), Q = gen(dt, d, NQ, rng);
            measure(rows, dt, d, X, Q, false);   // Hamming
            measure(rows, dt, d, X, Q, true);    // ADC
        }
        Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
        if (Files.exists(repo.resolve(SIFT))) {
            float[][] X = readFvecs(repo.resolve(SIFT).toString(), N), Q = readFvecs(repo.resolve(SIFT_Q).toString(), NQ);
            measure(rows, "sift_real", X[0].length, X, Q, false);
            measure(rows, "sift_real", X[0].length, X, Q, true);
        }
        Path out = Paths.get(System.getProperty("user.dir"), "research", "track2_adaptive_rescore", "results");
        Files.createDirectories(out);
        try (Writer w = Files.newBufferedWriter(out.resolve("track2_calibration.csv"))) { for (String r : rows) { w.write(r); w.write("\n"); } }
        System.out.println("WROTE calibration -> " + out.resolve("track2_calibration.csv"));
    }

    private void measure(List<String> rows, String dt, int d, float[][] X, float[][] Q, boolean adc) throws IOException {
        OneBitScalarQuantizer q = new OneBitScalarQuantizer(false);
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(X.length, false) {
            @Override public float[] getVectorAtThePosition(int p) { return X[p]; }
            @Override public void resetVectorValues() { }
        };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) q.train(req);
        byte[][] bits = new byte[X.length][];
        for (int j = 0; j < X.length; j++) { BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); q.quantize(X[j].clone(), st, o); bits[j] = o.getQuantizedVector().clone(); }
        Random rng = new Random(SEED + 1);
        List<double[]> pairs = new ArrayList<>();   // (estimate, trueL2)
        for (int qi = 0; qi < NQ; qi++) {
            float[] est;
            if (adc) { float[] qt = Q[qi].clone(); q.transformWithADC(qt, st, SpaceType.L2); est = qt; }
            else { BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); q.quantize(Q[qi].clone(), st, o); est = null; /*use bits*/ }
            byte[] qb = adc ? null : quantizeQ(q, Q[qi], st);
            for (int s = 0; s < SAMPLES_PER_Q; s++) {
                int j = rng.nextInt(X.length);
                double e = adc ? KNNScoringUtil.l2SquaredADC(est, bits[j]) : hamming(qb, bits[j]);
                double t = l2sq(Q[qi], X[j]);
                pairs.add(new double[] { e, t });
            }
        }
        // affine calibrate t ~= a*e + b
        double n = pairs.size(), se = 0, st_ = 0, see = 0, set = 0;
        for (double[] p : pairs) { se += p[0]; st_ += p[1]; see += p[0]*p[0]; set += p[0]*p[1]; }
        double den = n*see - se*se; double a = den == 0 ? 0 : (n*set - se*st_)/den; double b = (st_ - a*se)/n;
        // residuals
        double[] res = new double[pairs.size()]; double tmean = st_/n, tvar = 0;
        for (int i = 0; i < pairs.size(); i++) { double[] p = pairs.get(i); res[i] = p[1] - (a*p[0] + b); tvar += (p[1]-tmean)*(p[1]-tmean); }
        double tstd = Math.sqrt(tvar/n);
        double rmean = 0; for (double r : res) rmean += r; rmean /= n;
        double rvar = 0, m3 = 0, m4 = 0; for (double r : res) { double dd = r - rmean; rvar += dd*dd; }
        rvar /= n; double rstd = Math.sqrt(rvar);
        for (double r : res) { double dd = (r - rmean)/rstd; m3 += dd*dd*dd; m4 += dd*dd*dd*dd; }
        double skew = m3/n, exkurt = m4/n - 3.0;
        // homoscedasticity: residual std in low vs high estimate tertiles
        pairs.sort((x, y) -> Double.compare(x[0], y[0]));
        double sLo = binStd(pairs, res, a, b, 0, (int)(n/3)), sHi = binStd(pairs, res, a, b, (int)(2*n/3), (int)n);
        // bound coverage under Gaussian assumption using global sigma=rstd
        double c90 = coverage(res, rmean, rstd, 1.645), c95 = coverage(res, rmean, rstd, 1.960), c99 = coverage(res, rmean, rstd, 2.576);
        rows.add(String.format(java.util.Locale.ROOT, "%s,%s,%d,%.4g,%.4g,%.4f,%.3f,%.3f,%.4f,%.4f,%.3f,%.3f,%.3f",
            adc ? "ADC" : "Hamming", dt, d, a, b, tstd == 0 ? 0 : rstd/tstd, skew, exkurt,
            tstd == 0 ? 0 : sLo/tstd, tstd == 0 ? 0 : sHi/tstd, c90, c95, c99));
        System.out.printf(java.util.Locale.ROOT, "%-7s %-11s d=%4d  resid/exactstd=%.2f skew=%.2f exkurt=%.2f  homo(lo/hi)=%.2f/%.2f  cover95=%.2f%n",
            adc ? "ADC" : "Hamming", dt, d, tstd == 0 ? 0 : rstd/tstd, skew, exkurt, tstd == 0 ? 0 : sLo/tstd, tstd == 0 ? 0 : sHi/tstd, c95);
    }

    // recompute residual std within a sorted index range (already-sorted pairs; residual recomputed from a,b)
    private static double binStd(List<double[]> pairs, double[] resIgnored, double a, double b, int lo, int hi) {
        double m = 0; int c = 0; for (int i = lo; i < hi; i++) { double[] p = pairs.get(i); m += p[1]-(a*p[0]+b); c++; } m /= Math.max(1,c);
        double v = 0; for (int i = lo; i < hi; i++) { double[] p = pairs.get(i); double r = p[1]-(a*p[0]+b)-m; v += r*r; } return Math.sqrt(v/Math.max(1,c));
    }
    private static double coverage(double[] res, double mean, double sigma, double z) {
        if (sigma == 0) return 1; int in = 0; for (double r : res) if (Math.abs(r - mean) <= z*sigma) in++; return (double) in/res.length;
    }

    private static byte[] quantizeQ(OneBitScalarQuantizer q, float[] v, OneBitScalarQuantizationState st) {
        BinaryQuantizationOutput o = new BinaryQuantizationOutput(1); q.quantize(v.clone(), st, o); return o.getQuantizedVector().clone();
    }
    private static float l2sq(float[] a, float[] b) { float s = 0; for (int i = 0; i < a.length; i++) { float x = a[i]-b[i]; s += x*x; } return s; }
    private static int hamming(byte[] a, byte[] b) { int s = 0; for (int i = 0; i < a.length; i++) s += Integer.bitCount((a[i]^b[i]) & 0xFF); return s; }

    private static float[][] gen(String kind, int d, int n, Random rng) {
        float[][] X = new float[n][d];
        if (kind.equals("isotropic")) { for (int i=0;i<n;i++) for (int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); }
        else if (kind.equals("anisotropic")) { int r=Math.min(64,d); float[][] W=new float[r][d];
            for (int a=0;a<r;a++) for (int j=0;j<d;j++) W[a][j]=(float)rng.nextGaussian();
            float[] sc=new float[r]; for (int a=0;a<r;a++) sc[a]=(float)Math.pow(10.0,1.0*a/r);
            for (int i=0;i<n;i++){ float[] z=new float[r]; for (int a=0;a<r;a++) z[a]=(float)rng.nextGaussian()*sc[a];
                for (int j=0;j<d;j++){ float s=0; for (int a=0;a<r;a++) s+=z[a]*W[a][j]; X[i][j]=s+0.1f*(float)rng.nextGaussian(); } } }
        else if (kind.equals("clustered")) { int ncl=32; float[][] c=new float[ncl][d];
            for (int a=0;a<ncl;a++) for (int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f;
            for (int i=0;i<n;i++){ int a=rng.nextInt(ncl); for (int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } }
        else throw new IllegalArgumentException(kind);
        return X;
    }
    private static float[][] readFvecs(String path, int max) throws IOException {
        List<float[]> out = new ArrayList<>();
        try (RandomAccessFile f = new RandomAccessFile(path, "r")) { byte[] h = new byte[4];
            while (out.size() < max && f.getFilePointer() < f.length()) {
                if (f.read(h) != 4) break; int dim = ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt();
                byte[] buf = new byte[4*dim]; if (f.read(buf) != 4*dim) break; ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);
                float[] v = new float[dim]; for (int j=0;j<dim;j++) v[j]=bb.getFloat(); out.add(v);
            } }
        return out.toArray(new float[0][]);
    }
}
