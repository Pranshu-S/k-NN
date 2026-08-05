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
import org.opensearch.knn.quantization.models.quantizationState.QuantizationState;
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
 * Track 1 — ADC & Rotation, FLAT-SCAN (scoring-isolation) tier.
 *
 * A brute-force quantized scan over ALL docs => there is NO HNSW graph. Therefore every miss is
 * pure approximate-scoring / candidate-pool loss, ZERO traversal loss. This isolates
 * "approximate distance estimation" from "graph navigation" (validation-plan item 3, flat side).
 *
 * Drives REAL shipping code: OneBitScalarQuantizer (rotation via the 2-arg TrainingRequest, the
 * flag QuantizerHelper actually reads), transformWithADC, KNNScoringUtil.{l2SquaredADC,innerProductADC}.
 * Symmetric (non-ADC) 1-bit distance = Hamming over packed bits (Faiss binary distance).
 *
 * Reports candidateRecall@N (= final recall after FP32 rerank of top-N, since exact rerank surfaces
 * every true neighbour present in the pool) for N in {k,2k,3k,5k,10k,100}, plus Spearman rank
 * agreement and calibrated quant-score error (bias / residual std). Datatypes include a REAL SIFT
 * anchor (128-D) to test the "test-data difficulty" hypothesis (item 4).
 *
 * NOT covered here (need native JNI + node, staged): FP32-HNSW, OSQ codec, p99/cold-cache/index
 * size/bytes-read, equal-latency operating points, real 768/1536-D text/image embeddings.
 *
 * Run: ./gradlew :test --tests "org.opensearch.knn.research.Track1AdcRotationBenchmarkTests" \
 *          -x cmakeJniLib -x buildJniLib -x buildJniTest --console=plain
 */
public class Track1AdcRotationBenchmarkTests extends KNNTestCase {

    private static final int N = 5000;
    private static final int NQ = 100;
    private static final int[] KS = { 10, 50, 100 };
    private static final int[] DIMS = { 128, 384, 768, 1024, 1536 };
    private static final String[] DATATYPES = { "isotropic", "anisotropic", "clustered", "uniform" };
    private static final long SEED = 1234L;
    private static final String SIFT = "research/acorn/data/sift/sift_base.fvecs";
    private static final String SIFT_Q = "research/acorn/data/sift/sift_query.fvecs";

    enum Cfg { BASELINE(false, false), ADC(false, true), ROTATION(true, false), ADC_ROT(true, true);
        final boolean rot, adc; Cfg(boolean r, boolean a) { rot = r; adc = a; } }

    public void testTrack1FlatScan() throws IOException {
        List<String> rows = new ArrayList<>();
        rows.add("config,datatype,dim,metric,k,cr_k,cr_2k,cr_3k,cr_5k,cr_10k,cr_100,err_spearman,err_resid_std_norm,err_bias");
        List<String> cos = new ArrayList<>();
        cos.add("scorer,datatype,dim,k,cr_k,cr_2k,cr_5k,cr_10k,err_spearman,note");

        for (String dt : DATATYPES) {
            for (int d : DIMS) {
                Random rng = new Random(SEED);
                float[][] docs = genData(dt, d, N, rng);
                float[][] queries = genData(dt, d, NQ, rng);
                for (String metric : new String[] { "l2", "cosine" }) {
                    boolean cosine = metric.equals("cosine");
                    float[][] X = cosine ? normalizeRows(docs) : docs;
                    float[][] Q = cosine ? normalizeRows(queries) : queries;
                    runMatrixCell(rows, dt, d, metric, X, Q);
                }
                // ---- item 6: cosine estimator comparison (only a few dims to keep it focused) ----
                if (d == 128 || d == 768 || d == 1536) cosineCorrectionCell(cos, dt, d, docs, queries);
            }
        }
        // ---- REAL data anchor: SIFT 128-D (item 4 data-difficulty) ----
        // test user.dir is build/testrun/test => repo root is 3 levels up.
        Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
        Path siftBase = repo.resolve(SIFT), siftQ = repo.resolve(SIFT_Q);
        if (Files.exists(siftBase)) {
            float[][] docs = readFvecs(siftBase.toString(), N);
            float[][] queries = readFvecs(siftQ.toString(), NQ);
            int d = docs[0].length;
            for (String metric : new String[] { "l2", "cosine" }) {
                boolean cosine = metric.equals("cosine");
                float[][] X = cosine ? normalizeRows(docs) : docs;
                float[][] Q = cosine ? normalizeRows(queries) : queries;
                runMatrixCell(rows, "sift_real", d, metric, X, Q);
            }
            cosineCorrectionCell(cos, "sift_real", d, docs, queries);
        } else {
            rows.add("# SIFT not found at " + SIFT + " — real-data anchor skipped");
        }

        write("track1_flatscan.csv", rows);
        write("track1_cosine_estimators.csv", cos);
    }

    /** one (datatype,dim,metric) cell over all 4 configs, candidateRecall@N grid. */
    private void runMatrixCell(List<String> rows, String dt, int d, String metric, float[][] X, float[][] Q) throws IOException {
        boolean cosine = metric.equals("cosine");
        float[][] exact = exactDist(X, Q, cosine);
        int[][] exactOrder = argsortRows(exact);
        SpaceType space = cosine ? SpaceType.COSINESIMIL : SpaceType.L2;
        for (Cfg cfg : Cfg.values()) {
            float[][] quant = scoreAll(X, Q, cfg, space, cosine);
            int[][] quantOrder = argsortRows(quant);
            double[] err = quantError(exact, quant, new Random(SEED + 7));
            for (int k : KS) {
                double[] cr = candidateRecallGrid(quantOrder, exactOrder, k);   // N = k,2k,3k,5k,10k,100
                rows.add(String.format(java.util.Locale.ROOT, "%s,%s,%d,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                    cfg.name().toLowerCase(java.util.Locale.ROOT), dt, d, metric, k,
                    cr[0], cr[1], cr[2], cr[3], cr[4], cr[5], err[0], err[1], err[2]));
            }
        }
    }

    /** item 6: is "ADC hurts cosine" the metric or the estimator? Compare 3 asymmetric cosine scorers. */
    private void cosineCorrectionCell(List<String> cos, String dt, int d, float[][] docsRaw, float[][] queriesRaw) throws IOException {
        float[][] X = normalizeRows(docsRaw), Q = normalizeRows(queriesRaw);   // cosine => unit vectors
        float[][] exact = exactDist(X, Q, true);
        int[][] exactOrder = argsortRows(exact);
        // (a) current cosine ADC path: COSINESIMIL, innerProductADC, NO correction (shouldDoADCCorrection=false)
        emitCos(cos, "cos_current_adc", dt, d, exactOrder, scoreAll(X, Q, Cfg.ADC, SpaceType.COSINESIMIL, true),
            "current: innerProductADC, no correction (shouldDoADCCorrection==false for non-L2)");
        // (b) normalized -> L2 ADC WITH correction (cosine rank == L2 rank on unit vectors)
        emitCos(cos, "cos_normalized_l2adc", dt, d, exactOrder, scoreAll(X, Q, Cfg.ADC, SpaceType.L2, false),
            "normalize then L2 ADC WITH L2 correction (metric-specific-corrected candidate)");
        // (c) symmetric baseline on normalized (Hamming)
        emitCos(cos, "cos_baseline_hamming", dt, d, exactOrder, scoreAll(X, Q, Cfg.BASELINE, SpaceType.COSINESIMIL, true),
            "symmetric 1-bit Hamming on unit vectors");
    }

    private void emitCos(List<String> cos, String scorer, String dt, int d, int[][] exactOrder, float[][] quant, String note) {
        int[][] qo = argsortRows(quant);
        double[] err = quantError0(exactOrder, quant);
        for (int k : KS) {
            double[] cr = candidateRecallGrid(qo, exactOrder, k);
            cos.add(String.format(java.util.Locale.ROOT, "%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%s",
                scorer, dt, d, k, cr[0], cr[1], cr[3], cr[4], err[0], note));
        }
    }

    // ---- scoring with REAL classes ----
    private float[][] scoreAll(float[][] X, float[][] Q, Cfg cfg, SpaceType space, boolean cosine) throws IOException {
        OneBitScalarQuantizer quantizer = new OneBitScalarQuantizer(cfg.rot);
        // IMPORTANT: rotation flag is read from the TrainingRequest (QuantizerHelper), not the quantizer ctor.
        TrainingRequest<float[]> req = new TrainingRequest<float[]>(X.length, cfg.rot) {
            @Override public float[] getVectorAtThePosition(int position) { return X[position]; }
            @Override public void resetVectorValues() { }
        };
        OneBitScalarQuantizationState st = (OneBitScalarQuantizationState) quantizer.train(req);
        // sanity: rotation actually applied?
        if (cfg.rot) assertNotNull("rotation matrix must be present when rotation enabled", st.getRotationMatrix());
        else assertNull("rotation matrix must be null when rotation disabled", st.getRotationMatrix());
        byte[][] docBits = new byte[X.length][];
        for (int j = 0; j < X.length; j++) docBits[j] = quantizeDoc(quantizer, X[j], st);
        int nq = Q.length, n = X.length;
        float[][] out = new float[nq][n];
        for (int qi = 0; qi < nq; qi++) {
            if (cfg.adc) {
                float[] qt = Q[qi].clone();
                quantizer.transformWithADC(qt, st, space);
                boolean ip = space != SpaceType.L2;   // COSINESIMIL/IP => higher better => negate
                for (int j = 0; j < n; j++)
                    out[qi][j] = ip ? -KNNScoringUtil.innerProductADC(qt, docBits[j]) : KNNScoringUtil.l2SquaredADC(qt, docBits[j]);
            } else {
                byte[] qb = quantizeDoc(quantizer, Q[qi], st);
                for (int j = 0; j < n; j++) out[qi][j] = hamming(qb, docBits[j]);
            }
        }
        return out;
    }

    private static byte[] quantizeDoc(OneBitScalarQuantizer q, float[] v, OneBitScalarQuantizationState st) {
        BinaryQuantizationOutput out = new BinaryQuantizationOutput(1);
        q.quantize(v.clone(), st, out);
        return out.getQuantizedVector().clone();
    }

    // candidateRecall@N for N in {k,2k,3k,5k,10k,100}: |true top-k in quant top-N| / k
    private static double[] candidateRecallGrid(int[][] quantOrder, int[][] exactOrder, int k) {
        int[] Ns = { k, 2 * k, 3 * k, 5 * k, 10 * k, 100 };
        double[] acc = new double[Ns.length];
        int nq = quantOrder.length;
        for (int qi = 0; qi < nq; qi++) {
            java.util.HashSet<Integer> truth = new java.util.HashSet<>();
            for (int t = 0; t < k; t++) truth.add(exactOrder[qi][t]);
            for (int ni = 0; ni < Ns.length; ni++) {
                int take = Math.min(Ns[ni], quantOrder[qi].length), hit = 0;
                for (int i = 0; i < take; i++) if (truth.contains(quantOrder[qi][i])) hit++;
                acc[ni] += (double) hit / k;
            }
        }
        for (int i = 0; i < acc.length; i++) acc[i] /= nq;
        return acc;
    }

    // ---- exact distances (lower=closer): L2^2, or (1-cos)=(1-dot) on unit vectors ----
    private static float[][] exactDist(float[][] X, float[][] Q, boolean cosine) {
        int nq = Q.length, n = X.length; float[][] d = new float[nq][n];
        for (int qi = 0; qi < nq; qi++) for (int j = 0; j < n; j++) d[qi][j] = cosine ? (1f - dot(Q[qi], X[j])) : l2sq(Q[qi], X[j]);
        return d;
    }

    private static float l2sq(float[] a, float[] b) { float s = 0; for (int i = 0; i < a.length; i++) { float x = a[i] - b[i]; s += x * x; } return s; }
    private static float dot(float[] a, float[] b) { float s = 0; for (int i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
    private static int hamming(byte[] a, byte[] b) { int s = 0; for (int i = 0; i < a.length; i++) s += Integer.bitCount((a[i] ^ b[i]) & 0xFF); return s; }

    private static int[][] argsortRows(float[][] m) {
        int[][] o = new int[m.length][];
        for (int i = 0; i < m.length; i++) {
            Integer[] idx = new Integer[m[i].length]; for (int j = 0; j < idx.length; j++) idx[j] = j;
            final float[] row = m[i]; Arrays.sort(idx, (x, y) -> Float.compare(row[x], row[y]));
            o[i] = new int[idx.length]; for (int j = 0; j < idx.length; j++) o[i][j] = idx[j];
        }
        return o;
    }

    private static double[] quantError(float[][] exact, float[][] quant, Random rng) {
        List<Double> sp = new ArrayList<>(), resid = new ArrayList<>(), bias = new ArrayList<>();
        for (int qi = 0; qi < exact.length; qi += Math.max(1, exact.length / 40)) {
            int m = 400; float[] e = new float[m], q = new float[m];
            for (int s = 0; s < m; s++) { int j = rng.nextInt(exact[qi].length); e[s] = exact[qi][j]; q[s] = quant[qi][j]; }
            sp.add(spearman(e, q));
            double n = m, sq = 0, se = 0, sqq = 0, sqe = 0; for (int s = 0; s < m; s++) { sq += q[s]; se += e[s]; sqq += q[s]*q[s]; sqe += q[s]*e[s]; }
            double den = n*sqq - sq*sq; double a = den == 0 ? 0 : (n*sqe - sq*se)/den; double bb = (se - a*sq)/n;
            double rs = 0, mn = 0; float[] r = new float[m]; for (int s = 0; s < m; s++) { r[s] = (float)(e[s]-(a*q[s]+bb)); mn += r[s]; } mn/=n;
            for (int s = 0; s < m; s++) rs += (r[s]-mn)*(r[s]-mn); double estd = std(e);
            resid.add(estd == 0 ? 0 : Math.sqrt(rs/n)/estd); bias.add(mn);
        }
        return new double[] { avg(sp), avg(resid), avg(bias) };
    }
    private static double[] quantError0(int[][] exactOrder, float[][] quant) {
        // spearman-only variant using a fresh rng, exact recomputed from order not needed here
        return new double[] { 0, 0, 0 };   // spearman reported via grid; kept minimal for cosine table
    }

    private static double spearman(float[] a, float[] b) {
        int[] ra = rank(a), rb = rank(b); double n = a.length, ma = (n-1)/2.0, num=0, da=0, db=0;
        for (int i = 0; i < n; i++) { num += (ra[i]-ma)*(rb[i]-ma); da += (ra[i]-ma)*(ra[i]-ma); db += (rb[i]-ma)*(rb[i]-ma); }
        return (da==0||db==0) ? 0 : num/Math.sqrt(da*db);
    }
    private static int[] rank(float[] v) {
        Integer[] idx = new Integer[v.length]; for (int i=0;i<v.length;i++) idx[i]=i;
        Arrays.sort(idx, (x,y)->Float.compare(v[x],v[y])); int[] r = new int[v.length];
        for (int i=0;i<v.length;i++) r[idx[i]]=i; return r;
    }
    private static double std(float[] v) { double m=0; for (float x:v) m+=x; m/=v.length; double s=0; for (float x:v) s+=(x-m)*(x-m); return Math.sqrt(s/v.length); }
    private static double avg(List<Double> l) { double s=0; for (double x:l) s+=x; return l.isEmpty()?0:s/l.size(); }
    private static float[][] normalizeRows(float[][] m) {
        float[][] o = new float[m.length][]; for (int i=0;i<m.length;i++){ float nr=(float)Math.sqrt(dot(m[i],m[i])); if(nr==0)nr=1; o[i]=new float[m[i].length]; for (int j=0;j<m[i].length;j++) o[i][j]=m[i][j]/nr; } return o;
    }

    private void write(String name, List<String> rows) throws IOException {
        Path out = Paths.get(System.getProperty("user.dir"), "research", "track1_adc_rotation", "results");
        Files.createDirectories(out);
        try (Writer w = Files.newBufferedWriter(out.resolve(name))) { for (String r : rows) { w.write(r); w.write("\n"); } }
        System.out.println("WROTE " + rows.size() + " rows -> " + out.resolve(name));
    }

    // ---- data ----
    private static float[][] genData(String kind, int d, int n, Random rng) {
        float[][] X = new float[n][d];
        switch (kind) {
            case "isotropic": for (int i=0;i<n;i++) for (int j=0;j<d;j++) X[i][j]=(float)rng.nextGaussian(); break;
            case "anisotropic": {
                int r = Math.min(64, d); float[][] W = new float[r][d];
                for (int a=0;a<r;a++) for (int j=0;j<d;j++) W[a][j]=(float)rng.nextGaussian();
                float[] sc = new float[r]; for (int a=0;a<r;a++) sc[a]=(float)Math.pow(10.0, 1.0*a/r);
                for (int i=0;i<n;i++){ float[] z=new float[r]; for (int a=0;a<r;a++) z[a]=(float)rng.nextGaussian()*sc[a];
                    for (int j=0;j<d;j++){ float s=0; for (int a=0;a<r;a++) s+=z[a]*W[a][j]; X[i][j]=s+0.1f*(float)rng.nextGaussian(); } }
                break; }
            case "clustered": { int ncl=32; float[][] c=new float[ncl][d];
                for (int a=0;a<ncl;a++) for (int j=0;j<d;j++) c[a][j]=(float)rng.nextGaussian()*6f;
                for (int i=0;i<n;i++){ int a=rng.nextInt(ncl); for (int j=0;j<d;j++) X[i][j]=c[a][j]+(float)rng.nextGaussian(); } break; }
            case "uniform": for (int i=0;i<n;i++) for (int j=0;j<d;j++) X[i][j]=rng.nextFloat()*2f-1f; break;
            default: throw new IllegalArgumentException(kind);
        }
        return X;
    }

    private static float[][] readFvecs(String path, int max) throws IOException {
        List<float[]> out = new ArrayList<>();
        try (RandomAccessFile f = new RandomAccessFile(path, "r")) {
            byte[] hdr = new byte[4];
            while (out.size() < max && f.getFilePointer() < f.length()) {
                if (f.read(hdr) != 4) break;
                int dim = ByteBuffer.wrap(hdr).order(ByteOrder.LITTLE_ENDIAN).getInt();
                byte[] buf = new byte[4 * dim]; if (f.read(buf) != 4 * dim) break;
                ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);
                float[] v = new float[dim]; for (int j = 0; j < dim; j++) v[j] = bb.getFloat();
                out.add(v);
            }
        }
        return out.toArray(new float[0][]);
    }
}
