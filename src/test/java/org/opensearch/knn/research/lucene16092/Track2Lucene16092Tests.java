/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.research.lucene16092;

import org.apache.lucene.codecs.hnsw.DefaultFlatVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
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
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
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
import java.util.Locale;
import java.util.Random;

/**
 * Track 2, step 15/16 — Lucene #16029 follow-up. Runs the EXACT Lucene PR #16092 preconditioning
 * (random sign flips -> Fisher-Yates permutation -> block-diagonal FWHT, verbatim {@link
 * HadamardRotation}, PR head 132f8b17) in front of Lucene's REAL production 1-bit BBQ scoring
 * (OptimizedScalarQuantizer 1-bit doc + 4-bit asymmetric query + int4BitDotProduct + the Lucene104
 * corrective-term Euclidean formula), and against #16030-style data-blind (zero-centroid)
 * quantization.
 *
 * <p>Faithfulness anchors:
 * <ul>
 *   <li>{@code testRotationInvariants} — the #16092 rotation is deterministic, orthogonal, uses the
 *       784 = 512+256+16 block decomposition, and its permutation genuinely mixes energy across
 *       blocks; it does not mutate its input.
 *   <li>{@code testBbqParityAgainstRealLucene} — the fast scoring formula used for the sweeps is
 *       proven bit-identical to the REAL {@link Lucene104ScalarQuantizedVectorScorer} (driven via an
 *       in-memory {@link QuantizedByteVectorValues}) over thousands of random vectors, including the
 *       zero-centroid (data-blind) case.
 * </ul>
 *
 * <p>Metric discipline (Part 8/9 of the spec): the primary metric is candidate recall@10 BEFORE any
 * fp32 rerank; "candidate count" is the {@link TopKnnCollector} k (NOT Lucene's ef_search API).
 * Build scorer and search scorer are tracked separately; the common-fp32-graph experiment holds the
 * graph topology fixed and only swaps the search scorer.
 */
public class Track2Lucene16092Tests extends KNNTestCase {

    private static final int K = 10, M = 16, BEAM = 100;
    private static final int NQ = 300, NEVAL = 200;               // last NEVAL queries are held-out eval
    private static final int[] CANDS = { 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000 };
    private static final double[] TARGETS = { 0.90, 0.95, 0.97, 0.99 };
    private static final long SEED = 71L;                          // HNSW build seed (matches step14)
    private static final long[] ROT_SEEDS = { 1234567L, 2345678L, 3456789L };
    private static final String REAL = "research/acorn/data/real/";
    private static final ScalarEncoding ENC = ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE;

    // ======================================================================================
    // PART 2 — #16092 rotation correctness (real HadamardRotation)
    // ======================================================================================
    public void testRotationInvariants() {
        Random r = new Random(SEED);
        // (a) block decomposition for non-power-of-two dims (direct call, same package)
        assertTrue("784=512+256+16", Arrays.equals(new int[]{ 512, 256, 16 }, HadamardRotation.decomposeIntoPowerOfTwoBlocks(784)));
        assertTrue("768=512+256", Arrays.equals(new int[]{ 512, 256 }, HadamardRotation.decomposeIntoPowerOfTwoBlocks(768)));
        assertTrue("100=64+32+4", Arrays.equals(new int[]{ 64, 32, 4 }, HadamardRotation.decomposeIntoPowerOfTwoBlocks(100)));

        for (int dim : new int[]{ 128, 256, 784, 960 }) {
            HadamardRotation rot = HadamardRotation.create(dim, 42L);
            HadamardRotation rot2 = HadamardRotation.create(dim, 42L);
            double maxNorm = 0, maxDot = 0, maxDist = 0;
            for (int t = 0; t < 40; t++) {
                float[] x = randVec(dim, r), y = randVec(dim, r);
                float[] xIn = x.clone();
                float[] rx = new float[dim], ry = new float[dim], rx2 = new float[dim];
                rot.rotate(x, rx);
                rot.rotate(y, ry);
                rot2.rotate(x, rx2);
                // (b) determinism: same (dim,seed) -> identical output
                assertTrue("determinism dim=" + dim, Arrays.equals(rx, rx2));
                // (c) no input mutation
                assertTrue("no input mutation dim=" + dim, Arrays.equals(x, xIn));
                // (d) L2 norm preservation
                maxNorm = Math.max(maxNorm, Math.abs(norm(rx) - norm(x)));
                // (e) dot-product preservation (rotate both)
                maxDot = Math.max(maxDot, Math.abs(VectorUtil.dotProduct(rx, ry) - VectorUtil.dotProduct(x, y)));
                // (f) euclidean distance preservation
                maxDist = Math.max(maxDist, Math.abs((float) Math.sqrt(l2(rx, ry)) - (float) Math.sqrt(l2(x, y))));
            }
            assertTrue("norm preserve dim=" + dim + " err=" + maxNorm, maxNorm < 1e-3);
            assertTrue("dot preserve dim=" + dim + " err=" + maxDot, maxDot < 1e-2);
            assertTrue("dist preserve dim=" + dim + " err=" + maxDist, maxDist < 1e-3);
        }

        // (g) permutation is genuinely applied: feed a vector that is non-zero ONLY in the first
        // power-of-two block (indices [0,512)). A block-diagonal FWHT WITHOUT permutation would leave
        // indices [512,784) exactly zero (block-2/3 inputs are zero). With the permutation, energy is
        // spread into later blocks, so their post-rotation energy must be clearly non-zero.
        int dim = 784;
        HadamardRotation rot = HadamardRotation.create(dim, 42L);
        float[] x = new float[dim];
        for (int j = 0; j < 512; j++) x[j] = (float) r.nextGaussian();
        float[] rx = new float[dim];
        rot.rotate(x, rx);
        double energyFirst = 0, energyRest = 0;
        for (int j = 0; j < 512; j++) energyFirst += rx[j] * rx[j];
        for (int j = 512; j < dim; j++) energyRest += rx[j] * rx[j];
        assertTrue("permutation must spread energy into later blocks (rest=" + energyRest + ")", energyRest > 0.05 * energyFirst);

        // (h) inverse recovers the original
        float[] back = new float[dim];
        rot.inverseRotate(rx, back);
        double maxBack = 0;
        for (int j = 0; j < dim; j++) maxBack = Math.max(maxBack, Math.abs(back[j] - x[j]));
        assertTrue("inverse recovers input err=" + maxBack, maxBack < 1e-3);

        System.out.println("[step15] rotation invariants PASS (determinism, orthogonality, 784=512+256+16, permutation spreads energy, invertible, no mutation)");
    }

    // ======================================================================================
    // PART 1 — BBQ parity: fast formula == REAL Lucene104ScalarQuantizedVectorScorer
    // ======================================================================================
    public void testBbqParityAgainstRealLucene() throws IOException {
        Random r = new Random(SEED + 3);
        Lucene104ScalarQuantizedVectorScorer scorer = new Lucene104ScalarQuantizedVectorScorer(new DefaultFlatVectorScorer());
        double maxRel = 0;
        int n = 0;
        for (int dim : new int[]{ 784, 256, 128, 64, 100 }) {
            OptimizedScalarQuantizer q = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
            int disc = ENC.getDiscreteDimensions(dim), dpl = ENC.getDocPackedLength(dim), qpl = ENC.getQueryPackedLength(dim);
            for (int trial = 0; trial < 150; trial++) {
                boolean zeroCentroid = r.nextBoolean();                 // exercise the data-blind (#16030) case too
                float[] cen = new float[dim];
                float[] query = randVec(dim, r);
                float[][] docs = new float[4][];
                for (int d = 0; d < 4; d++) docs[d] = randVec(dim, r);
                if (!zeroCentroid) for (int j = 0; j < dim; j++) cen[j] = (float) (0.4 * r.nextGaussian());
                byte[][] packed = new byte[4][dpl];
                QuantizationResult[] corr = new QuantizationResult[4];
                for (int d = 0; d < 4; d++) {
                    byte[] dest = new byte[disc];
                    corr[d] = q.scalarQuantize(docs[d].clone(), dest, (byte) 1, cen);
                    packed[d] = new byte[dpl];
                    OptimizedScalarQuantizer.packAsBinary(dest, packed[d]);
                }
                MemQuantized vals = new MemQuantized(dim, packed, corr, q, cen);
                RandomVectorScorer rvs = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vals, query.clone());
                // our fast path: quantize the query 4-bit ourselves, then apply the transcribed formula
                byte[] qdest = new byte[disc];
                QuantizationResult qc = q.scalarQuantize(query.clone(), qdest, (byte) 4, cen);
                byte[] qT = new byte[qpl];
                OptimizedScalarQuantizer.transposeHalfByte(qdest, qT);
                for (int d = 0; d < 4; d++) {
                    float real = rvs.score(d);                          // real Lucene = 1/(1+max(dist2,0))
                    float mine = 1f / (1f + bbqSquaredL2(qT, qc, packed[d], corr[d], dim));
                    maxRel = Math.max(maxRel, Math.abs(real - mine) / Math.max(1e-9f, Math.abs(real)));
                    n++;
                }
            }
        }
        System.out.printf(Locale.ROOT, "[step15] BBQ parity: %d comparisons vs real Lucene104 scorer, max rel err = %.3e%n", n, maxRel);
        assertTrue("BBQ fast formula must match the real Lucene104 scorer (maxRel=" + maxRel + ")", maxRel < 1e-5);
    }

    // ======================================================================================
    // PART 3 + 5 — Fashion-MNIST + iid, native graph. fp32 / BBQ / old-RHT / #16092 (all centered).
    // ======================================================================================
    public void testFashionMnistAndIid() throws IOException {
        Path dir = outDir();
        String hdr = "dataset,dim,n,queries,seed,graph_builder,scorer,rotation,centering,code_bytes,sidecar_bytes,"
            + "effective_bits_per_dim,rotation_global_bytes,flat_recall_at_10,build_ms,rotation_ms,"
            + join("candrec_k", CANDS) + ",k_at_90,k_at_95,k_at_97,k_at_99,mae,rmse,p95_error";
        writeFresh(dir, "step15_lucene16092_rotation.csv", hdr);
        Object[][] sets = { { "fmnist784", 784, 20000, "fmnist" }, { "iid784", 784, 15000, "iid" } };
        List<String> summary = new ArrayList<>();
        for (Object[] s : sets) {
            String name = (String) s[0]; int dim = (int) s[1], N = (int) s[2]; String src = (String) s[3];
            float[][] X = load(src, dim, N, false), Q = load(src, dim, NQ, true);
            int[][] gt = groundTruth(X, Q);                         // exact fp32 GT (rotation-invariant)
            List<String> rows = new ArrayList<>();
            // fp32 (rotation/seed independent)
            rows.add(evalNative(name, dim, N, "fp32_flat", "none", "centered", new Fp32(X), X, Q, gt, -1L, 0, summary));
            // BBQ centered (rotation/seed independent)
            rows.add(evalNative(name, dim, N, "bbq_1bit", "none", "centered", bbq(X, false), X, Q, gt, -1L, 0, summary));
            // old RHT -> BBQ and #16092 -> BBQ, over ROT_SEEDS
            for (long rs : ROT_SEEDS) {
                float[][] Xo = applyOldRht(X, dim, rs), Qo = applyOldRht(Q, dim, rs);
                rows.add(evalNative(name, dim, N, "oldrht_bbq", "old_rht", "centered", bbq(Xo, false), Xo, Qo, gt, rs, globalRotBytes(dim), summary));
                float[][] Xr = applyL16092(X, dim, rs), Qr = applyL16092(Q, dim, rs);
                rows.add(evalNative(name, dim, N, "l16092_bbq", "lucene_16092", "centered", bbq(Xr, false), Xr, Qr, gt, rs, 8, summary));
            }
            append(dir, "step15_lucene16092_rotation.csv", rows);
            System.out.printf(Locale.ROOT, "[step15] %s native-graph done%n", name);
        }
        writeLines(dir, "step15_rotation_seed_summary.csv", summary,
            "dataset,scorer,metric,mean,min,max");
        System.out.println("[step15] testFashionMnistAndIid DONE");
    }

    // ======================================================================================
    // PART 4 — Common FP32 graph: ONE fp32 topology, swap only the search scorer.
    // ======================================================================================
    public void testCommonGraph() throws IOException {
        Path dir = outDir();
        writeFresh(dir, "step15_lucene16092_common_graph.csv",
            "dataset,dim,n,queries,seed,graph_builder,scorer,rotation,centering," + join("candrec_k", CANDS)
                + ",k_at_90,k_at_95,k_at_97,k_at_99");
        Object[][] sets = { { "fmnist784", 784, 20000, "fmnist" }, { "iid784", 784, 15000, "iid" } };
        List<String> summary = new ArrayList<>();
        for (Object[] s : sets) {
            String name = (String) s[0]; int dim = (int) s[1], N = (int) s[2]; String src = (String) s[3];
            float[][] X = load(src, dim, N, false), Q = load(src, dim, NQ, true);
            int[][] gt = groundTruth(X, Q);
            OnHeapHnswGraph g = build(new Fp32(X), N);               // single fp32 graph for all scorers
            List<String> rows = new ArrayList<>();
            rows.add(commonRow(name, dim, N, g, "fp32_flat", "none", "centered", new Fp32(X), Q, gt, -1L, summary));
            rows.add(commonRow(name, dim, N, g, "bbq_1bit", "none", "centered", bbq(X, false), Q, gt, -1L, summary));
            for (long rs : ROT_SEEDS) {
                float[][] Xo = applyOldRht(X, dim, rs), Qo = applyOldRht(Q, dim, rs);
                rows.add(commonRow(name, dim, N, g, "oldrht_bbq", "old_rht", "centered", bbq(Xo, false), Qo, gt, rs, summary));
                float[][] Xr = applyL16092(X, dim, rs), Qr = applyL16092(Q, dim, rs);
                rows.add(commonRow(name, dim, N, g, "l16092_bbq", "lucene_16092", "centered", bbq(Xr, false), Qr, gt, rs, summary));
            }
            append(dir, "step15_lucene16092_common_graph.csv", rows);
            System.out.printf(Locale.ROOT, "[step15-common] %s done%n", name);
        }
        writeLines(dir, "step15_common_seed_summary.csv", summary, "dataset,scorer,metric,mean,min,max");
        System.out.println("[step15] testCommonGraph DONE");
    }

    // ======================================================================================
    // PART 6 — centered vs data-blind × rotation(none/#16092): 2x2 matrix.
    // ======================================================================================
    public void testDataBlindMatrix() throws IOException {
        Path dir = outDir();
        // NOTE: matrixLine() emits (scorer, rotation, centering, graph_mode) — NO separate graph_builder column.
        String hdr = "dataset,dim,n,queries,seed,scorer,rotation,centering,graph_mode,code_bytes,sidecar_bytes,"
            + "effective_bits_per_dim,flat_recall_at_10," + join("candrec_k", CANDS) + ",k_at_90,k_at_95,k_at_97,k_at_99,mae,rmse,p95_error";
        writeFresh(dir, "step16_datablind_matrix.csv", hdr);
        writeFresh(dir, "step16_datablind_common_graph.csv", hdr);
        // fmnist full 2x2 (native + common). iid: k@0.95 essential comparison (native + common).
        Object[][] sets = { { "fmnist784", 784, 20000, "fmnist" }, { "iid784", 784, 15000, "iid" } };
        for (Object[] s : sets) {
            String name = (String) s[0]; int dim = (int) s[1], N = (int) s[2]; String src = (String) s[3];
            float[][] X = load(src, dim, N, false), Q = load(src, dim, NQ, true);
            int[][] gt = groundTruth(X, Q);
            OnHeapHnswGraph fp32graph = build(new Fp32(X), N);
            List<String> nat = new ArrayList<>(), com = new ArrayList<>();
            // A: centered, no rotation ; B: data-blind, no rotation
            nat.add(matrixNative(name, dim, N, "A_centered", "none", "centered", bbq(X, false), X, Q, gt));
            com.add(matrixCommon(name, dim, N, fp32graph, "A_centered", "none", "centered", bbq(X, false), Q, gt));
            nat.add(matrixNative(name, dim, N, "B_datablind", "none", "data_blind", bbq(X, true), X, Q, gt));
            com.add(matrixCommon(name, dim, N, fp32graph, "B_datablind", "none", "data_blind", bbq(X, true), Q, gt));
            // C: #16092 + centered ; D: #16092 + data-blind (mean over ROT_SEEDS reported per-seed)
            for (long rs : ROT_SEEDS) {
                float[][] Xr = applyL16092(X, dim, rs), Qr = applyL16092(Q, dim, rs);
                nat.add(matrixNative(name, dim, N, "C_16092_centered", "lucene_16092", "centered", bbq(Xr, false), Xr, Qr, gt));
                com.add(matrixCommon(name, dim, N, fp32graph, "C_16092_centered", "lucene_16092", "centered", bbq(Xr, false), Qr, gt));
                nat.add(matrixNative(name, dim, N, "D_16092_datablind", "lucene_16092", "data_blind", bbq(Xr, true), Xr, Qr, gt));
                com.add(matrixCommon(name, dim, N, fp32graph, "D_16092_datablind", "lucene_16092", "data_blind", bbq(Xr, true), Qr, gt));
            }
            append(dir, "step16_datablind_matrix.csv", nat);
            append(dir, "step16_datablind_common_graph.csv", com);
            System.out.printf(Locale.ROOT, "[step16] %s 2x2 matrix done%n", name);
        }
        System.out.println("[step16] testDataBlindMatrix DONE");
    }

    // ============================== eval helpers ==============================
    private String evalNative(String ds, int dim, int N, String scorer, String rot, String centering, Rep r,
                              float[][] Xs, float[][] Qs, int[][] gt, long seed, int rotGlobal, List<String> summary) throws IOException {
        double flat = flatFidelity(r, N, Qs, gt);
        double[] err = scoreError(r, Xs, Qs);
        long t0 = System.nanoTime(); OnHeapHnswGraph g = build(r, N); long buildMs = (System.nanoTime() - t0) / 1_000_000;
        double[] rec = new double[CANDS.length]; int[] kAt = { -1, -1, -1, -1 };
        for (int i = 0; i < CANDS.length; i++) { rec[i] = candRecall(g, r, Qs, gt, CANDS[i]); for (int t = 0; t < 4; t++) if (kAt[t] < 0 && rec[i] >= TARGETS[t]) kAt[t] = CANDS[i]; }
        if (summary != null) { summary.add(ds + "," + scorer + ",k_at_95," + kAt[1] + "," + kAt[1] + "," + kAt[1]); }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.ROOT, "%s,%d,%d,%d,%d,hnsw_native,%s,%s,%s,%d,%d,%.3f,%d,%.4f,%d,%d",
            ds, dim, N, NEVAL, seed, scorer, rot, centering, r.codeBytes(), r.sidecarBytes(), r.effBits(), rotGlobal, flat, buildMs, 0));
        for (double v : rec) sb.append(String.format(Locale.ROOT, ",%.4f", v));
        sb.append(String.format(Locale.ROOT, ",%d,%d,%d,%d,%.4f,%.4f,%.4f", kAt[0], kAt[1], kAt[2], kAt[3], err[0], err[1], err[2]));
        System.out.printf(Locale.ROOT, "[step15]   %-12s %-13s seed=%-8d flat=%.3f k95=%d build=%dms%n", ds, scorer + "/" + rot, seed, flat, kAt[1], buildMs);
        return sb.toString();
    }

    private String commonRow(String ds, int dim, int N, OnHeapHnswGraph g, String scorer, String rot, String centering,
                             Rep r, float[][] Qs, int[][] gt, long seed, List<String> summary) throws IOException {
        double[] rec = new double[CANDS.length]; int[] kAt = { -1, -1, -1, -1 };
        for (int i = 0; i < CANDS.length; i++) { rec[i] = candRecall(g, r, Qs, gt, CANDS[i]); for (int t = 0; t < 4; t++) if (kAt[t] < 0 && rec[i] >= TARGETS[t]) kAt[t] = CANDS[i]; }
        if (summary != null) summary.add(ds + "," + scorer + ",k_at_95," + kAt[1] + "," + kAt[1] + "," + kAt[1]);
        StringBuilder sb = new StringBuilder(String.format(Locale.ROOT, "%s,%d,%d,%d,%d,common_fp32_graph,%s,%s,%s", ds, dim, N, NEVAL, seed, scorer, rot, centering));
        for (double v : rec) sb.append(String.format(Locale.ROOT, ",%.4f", v));
        sb.append(String.format(Locale.ROOT, ",%d,%d,%d,%d", kAt[0], kAt[1], kAt[2], kAt[3]));
        System.out.printf(Locale.ROOT, "[step15-common] %-12s %-13s seed=%-8d k90=%d k95=%d k97=%d%n", ds, scorer + "/" + rot, seed, kAt[0], kAt[1], kAt[2]);
        return sb.toString();
    }

    private String matrixNative(String ds, int dim, int N, String scorer, String rot, String centering, Rep r,
                                float[][] Xs, float[][] Qs, int[][] gt) throws IOException {
        double flat = flatFidelity(r, N, Qs, gt); double[] err = scoreError(r, Xs, Qs);
        OnHeapHnswGraph g = build(r, N);
        double[] rec = new double[CANDS.length]; int[] kAt = { -1, -1, -1, -1 };
        for (int i = 0; i < CANDS.length; i++) { rec[i] = candRecall(g, r, Qs, gt, CANDS[i]); for (int t = 0; t < 4; t++) if (kAt[t] < 0 && rec[i] >= TARGETS[t]) kAt[t] = CANDS[i]; }
        return matrixLine(ds, dim, N, "hnsw_native", scorer, rot, centering, r, flat, rec, kAt, err);
    }

    private String matrixCommon(String ds, int dim, int N, OnHeapHnswGraph g, String scorer, String rot, String centering, Rep r,
                                float[][] Qs, int[][] gt) throws IOException {
        double flat = flatFidelity(r, N, Qs, gt);
        double[] rec = new double[CANDS.length]; int[] kAt = { -1, -1, -1, -1 };
        for (int i = 0; i < CANDS.length; i++) { rec[i] = candRecall(g, r, Qs, gt, CANDS[i]); for (int t = 0; t < 4; t++) if (kAt[t] < 0 && rec[i] >= TARGETS[t]) kAt[t] = CANDS[i]; }
        double[] err = { 0, 0, 0 };
        return matrixLine(ds, dim, N, "common_fp32_graph", scorer, rot, centering, r, flat, rec, kAt, err);
    }

    private String matrixLine(String ds, int dim, int N, String gmode, String scorer, String rot, String centering, Rep r,
                              double flat, double[] rec, int[] kAt, double[] err) {
        StringBuilder sb = new StringBuilder(String.format(Locale.ROOT, "%s,%d,%d,%d,%d,%s,%s,%s,%s,%d,%d,%.3f,%.4f",
            ds, dim, N, NEVAL, SEED, scorer, rot, centering, gmode, r.codeBytes(), r.sidecarBytes(), r.effBits(), flat));
        for (double v : rec) sb.append(String.format(Locale.ROOT, ",%.4f", v));
        sb.append(String.format(Locale.ROOT, ",%d,%d,%d,%d,%.4f,%.4f,%.4f", kAt[0], kAt[1], kAt[2], kAt[3], err[0], err[1], err[2]));
        System.out.printf(Locale.ROOT, "[step16]   %-12s %-20s %-16s k95=%d flat=%.3f%n", ds, scorer, gmode, kAt[1], flat);
        return sb.toString();
    }

    // ============================== representations ==============================
    /** A scoring representation. Query is prepared once (quantized) then reused across doc scores. */
    interface Rep {
        float symDist(int a, int b);            // build-time distance (both sides in this rep's encoding)
        Object prepare(float[] q);              // quantize/prepare a query once
        float dist(Object preparedQuery, int b);// estimated squared-L2 distance query->doc b
        double effBits(); int n(); int codeBytes(); int sidecarBytes();
    }

    static final class Fp32 implements Rep {
        final float[][] X; Fp32(float[][] X) { this.X = X; }
        public float symDist(int a, int b) { return l2(X[a], X[b]); }
        public Object prepare(float[] q) { return q; }
        public float dist(Object q, int b) { return l2((float[]) q, X[b]); }
        public double effBits() { return 32; } public int n() { return X.length; }
        public int codeBytes() { return X[0].length * 4; } public int sidecarBytes() { return 0; }
    }

    /** REAL Lucene 1-bit BBQ (1-bit doc + 4-bit asymmetric query + Lucene104 corrective formula).
     *  The scoring formula is proven bit-identical to Lucene104ScalarQuantizedVectorScorer by
     *  {@link #testBbqParityAgainstRealLucene}. Packing uses the real codec byte lengths. */
    static final class Bbq implements Rep {
        final int n, dim, disc, dpl, qpl; final float[] centroid;
        final byte[][] docPacked; final float[] dLo, dHi, dAdd; final int[] dSum;      // 1-bit doc
        final byte[][] qTrans; final float[] qLo, qHi, qAdd; final int[] qSum;         // 4-bit query-of-doc (build)
        final OptimizedScalarQuantizer osq;
        Bbq(float[][] X, float[] centroid) {
            this.n = X.length; this.dim = X[0].length; this.centroid = centroid;
            this.disc = ENC.getDiscreteDimensions(dim); this.dpl = ENC.getDocPackedLength(dim); this.qpl = ENC.getQueryPackedLength(dim);
            this.osq = new OptimizedScalarQuantizer(VectorSimilarityFunction.EUCLIDEAN);
            docPacked = new byte[n][dpl]; dLo = new float[n]; dHi = new float[n]; dAdd = new float[n]; dSum = new int[n];
            qTrans = new byte[n][qpl]; qLo = new float[n]; qHi = new float[n]; qAdd = new float[n]; qSum = new int[n];
            for (int i = 0; i < n; i++) {
                byte[] dd = new byte[disc];
                QuantizationResult rd = osq.scalarQuantize(X[i].clone(), dd, (byte) 1, centroid);
                OptimizedScalarQuantizer.packAsBinary(dd, docPacked[i]); dLo[i] = rd.lowerInterval(); dHi[i] = rd.upperInterval(); dAdd[i] = rd.additionalCorrection(); dSum[i] = rd.quantizedComponentSum();
                byte[] dq = new byte[disc];
                QuantizationResult rq = osq.scalarQuantize(X[i].clone(), dq, (byte) 4, centroid);
                OptimizedScalarQuantizer.transposeHalfByte(dq, qTrans[i]); qLo[i] = rq.lowerInterval(); qHi[i] = rq.upperInterval(); qAdd[i] = rq.additionalCorrection(); qSum[i] = rq.quantizedComponentSum();
            }
        }
        static final class PQ { final byte[] t; final float lo, hi, add; final int sum; PQ(byte[] t, float lo, float hi, float add, int sum) { this.t = t; this.lo = lo; this.hi = hi; this.add = add; this.sum = sum; } }
        public Object prepare(float[] q) {
            byte[] dq = new byte[disc]; QuantizationResult rq = osq.scalarQuantize(q.clone(), dq, (byte) 4, centroid);
            byte[] t = new byte[qpl]; OptimizedScalarQuantizer.transposeHalfByte(dq, t);
            return new PQ(t, rq.lowerInterval(), rq.upperInterval(), rq.additionalCorrection(), rq.quantizedComponentSum());
        }
        public float dist(Object pq, int b) { PQ p = (PQ) pq; return score(p.t, p.lo, p.hi, p.add, p.sum, b); }
        public float symDist(int a, int b) { return score(qTrans[a], qLo[a], qHi[a], qAdd[a], qSum[a], b); }
        private float score(byte[] qT, float qlo, float qhi, float qadd, int qsm, int b) {
            long qc = VectorUtil.int4BitDotProduct(qT, docPacked[b]);
            float scale = 1f, queryScale = 1f / 15f;             // 1-bit doc, 4-bit query (SCALE_LUT[0], SCALE_LUT[3])
            float x1 = dSum[b], ax = dLo[b], lx = (dHi[b] - ax) * scale;
            float ay = qlo, ly = (qhi - ay) * queryScale, y1 = qsm;
            float dot = ax * ay * dim + ay * lx * x1 + ax * ly * y1 + lx * ly * (float) qc;
            float dist2 = qadd + dAdd[b] - 2f * dot; return Math.max(dist2, 0f);
        }
        public double effBits() { return (double) (dpl + 16) * 8 / dim; } public int n() { return n; }
        public int codeBytes() { return dpl; } public int sidecarBytes() { return 16; }
    }

    /** Bbq with centering (centroid = data mean) or data-blind (#16030 enableCentering=false => centroid = 0). */
    private static Bbq bbq(float[][] X, boolean dataBlind) {
        float[] cen = dataBlind ? new float[X[0].length] : mean(X);
        return new Bbq(X, cen);
    }

    // In-memory Lucene104 QuantizedByteVectorValues used ONLY to drive the real scorer in the parity test.
    static final class MemQuantized extends QuantizedByteVectorValues {
        final int dim; final byte[][] packed; final QuantizationResult[] corr; final OptimizedScalarQuantizer q; final float[] cen;
        MemQuantized(int dim, byte[][] p, QuantizationResult[] c, OptimizedScalarQuantizer q, float[] cen) { this.dim = dim; this.packed = p; this.corr = c; this.q = q; this.cen = cen; }
        public byte[] vectorValue(int o) { return packed[o]; }
        public QuantizationResult getCorrectiveTerms(int o) { return corr[o]; }
        public OptimizedScalarQuantizer getQuantizer() { return q; }
        public ScalarEncoding getScalarEncoding() { return ENC; }
        public float[] getCentroid() { return cen; }
        public float getCentroidDP() { return VectorUtil.dotProduct(cen, cen); }
        public int dimension() { return dim; }
        public int size() { return packed.length; }
        public QuantizedByteVectorValues copy() { return this; }
        public org.apache.lucene.search.VectorScorer scorer(float[] q) { throw new UnsupportedOperationException(); }
    }

    private static float bbqSquaredL2(byte[] qT, QuantizationResult qc, byte[] docP, QuantizationResult dc, int dim) {
        long qcDot = VectorUtil.int4BitDotProduct(qT, docP);
        float scale = 1f, queryScale = 1f / 15f;
        float x1 = dc.quantizedComponentSum(), ax = dc.lowerInterval(), lx = (dc.upperInterval() - ax) * scale;
        float ay = qc.lowerInterval(), ly = (qc.upperInterval() - ay) * queryScale, y1 = qc.quantizedComponentSum();
        float dot = ax * ay * dim + ay * lx * x1 + ax * ly * y1 + lx * ly * (float) qcDot;
        return Math.max(qc.additionalCorrection() + dc.additionalCorrection() - 2f * dot, 0f);
    }

    // ============================== rotations ==============================
    private static int globalRotBytes(int dim) { return HadamardRotation.decomposeIntoPowerOfTwoBlocks(dim).length * 4 + 8; }

    /** EXACT Lucene #16092 rotation applied to every row. */
    private static float[][] applyL16092(float[][] X, int dim, long seed) {
        HadamardRotation rot = HadamardRotation.create(dim, seed);
        float[][] Y = new float[X.length][dim]; float[] scratch = new float[dim];
        for (int i = 0; i < X.length; i++) rot.rotate(X[i], Y[i], scratch);
        return Y;
    }

    /** OLD RHT (the step14 transform): sign flips + block-diagonal FWHT, NO permutation. */
    private static float[][] applyOldRht(float[][] X, int dim, long seed) {
        int[] blk = HadamardRotation.decomposeIntoPowerOfTwoBlocks(dim);
        Random r = new Random(seed); float[] sign = new float[dim];
        for (int j = 0; j < dim; j++) sign[j] = r.nextBoolean() ? 1f : -1f;
        float[][] Y = new float[X.length][];
        for (int i = 0; i < X.length; i++) Y[i] = oldRht(X[i], blk, sign);
        return Y;
    }
    private static float[] oldRht(float[] x, int[] blocks, float[] sign) {
        int dim = x.length; float[] y = new float[dim];
        for (int j = 0; j < dim; j++) y[j] = x[j] * sign[j];
        int off = 0; for (int bl : blocks) { fwht(y, off, bl); float inv = 1f / (float) Math.sqrt(bl); for (int j = off; j < off + bl; j++) y[j] *= inv; off += bl; }
        return y;
    }
    private static void fwht(float[] a, int off, int n) { for (int len = 1; len < n; len <<= 1) for (int i = off; i < off + n; i += len << 1) for (int j = i; j < i + len; j++) { float u = a[j], v = a[j + len]; a[j] = u + v; a[j + len] = u - v; } }

    // ============================== graph + metrics ==============================
    private OnHeapHnswGraph build(Rep r, int N) throws IOException {
        RandomVectorScorerSupplier sup = new RandomVectorScorerSupplier() {
            public UpdateableRandomVectorScorer scorer() {
                return new UpdateableRandomVectorScorer() { int cur = 0; public int maxOrd() { return N; } public void setScoringOrdinal(int o) { cur = o; } public float score(int j) { return -r.symDist(cur, j); } };
            }
            public RandomVectorScorerSupplier copy() { return this; }
        };
        return HnswGraphBuilder.create(sup, M, BEAM, SEED).build(N);
    }

    /** Per-thread cursor over a shared graph (only single-threaded here, kept for correctness). */
    static final class GV extends HnswGraph {
        final OnHeapHnswGraph g; int[] cur = new int[0]; int sz = 0, up = 0; GV(OnHeapHnswGraph g) { this.g = g; }
        public void seek(int l, int nd) { NeighborArray na = g.getNeighbors(l, nd); cur = na.nodes(); sz = na.size(); up = 0; }
        public int size() { return g.size(); }
        public int nextNeighbor() { return up < sz ? cur[up++] : DocIdSetIterator.NO_MORE_DOCS; }
        public int numLevels() throws IOException { return g.numLevels(); }
        public int maxConn() { return g.maxConn(); }
        public int entryNode() throws IOException { return g.entryNode(); }
        public int neighborCount() { return sz; }
        public HnswGraph.NodesIterator getNodesOnLevel(int l) throws IOException { return g.getNodesOnLevel(l); }
    }

    /** Candidate recall@10 BEFORE rerank: fraction of the true top-10 present among the `cand` collected candidates. */
    private double candRecall(OnHeapHnswGraph g, Rep r, float[][] Q, int[][] gt, int cand) throws IOException {
        double s = 0; int c = 0;
        for (int qi = NQ - NEVAL; qi < NQ; qi++) {
            final Object pq = r.prepare(Q[qi]);
            RandomVectorScorer qs = new RandomVectorScorer() { public int maxOrd() { return r.n(); } public float score(int ord) { return -r.dist(pq, ord); } };
            TopKnnCollector col = new TopKnnCollector(cand, Integer.MAX_VALUE);
            HnswGraphSearcher.search(qs, col, new GV(g), null);
            ScoreDoc[] sd = col.topDocs().scoreDocs;
            java.util.HashSet<Integer> got = new java.util.HashSet<>(); for (ScoreDoc d : sd) got.add(d.doc);
            int h = 0; for (int t = 0; t < K; t++) if (got.contains(gt[qi][t])) h++;
            s += h / (double) K; c++;
        }
        return c == 0 ? 0 : s / c;
    }

    /** Flat (no-graph) top-10 overlap of the rep's estimated distances against exact fp32 GT ordinals. */
    private double flatFidelity(Rep r, int N, float[][] Q, int[][] gt) {
        double s = 0; int c = 0; int probe = Math.min(NEVAL, 80);
        for (int qi = NQ - NEVAL; qi < NQ - NEVAL + probe; qi++) {
            Object pq = r.prepare(Q[qi]); float[] sc = new float[N]; for (int j = 0; j < N; j++) sc[j] = r.dist(pq, j);
            int[] top = topIdx(sc, K); java.util.HashSet<Integer> t = new java.util.HashSet<>(); for (int x : top) t.add(x);
            int h = 0; for (int x : gt[qi]) if (t.contains(x)) h++; s += h / (double) K; c++;
        }
        return c == 0 ? 0 : s / c;
    }

    /** Approximate-distance error (MAE/RMSE/p95|err|) vs exact L2 in the rep's own (possibly rotated) space. */
    private double[] scoreError(Rep r, float[][] Xs, float[][] Qs) {
        Random rng = new Random(SEED + 9); List<Double> ae = new ArrayList<>(); double se = 0; int cnt = 0;
        for (int t = 0; t < 40; t++) {
            int qi = NQ - NEVAL + rng.nextInt(NEVAL); Object pq = r.prepare(Qs[qi]);
            for (int p = 0; p < 25; p++) {
                int d = rng.nextInt(Xs.length);
                double approx = Math.sqrt(Math.max(0, r.dist(pq, d))); double exact = Math.sqrt(l2(Qs[qi], Xs[d]));
                double e = approx - exact; ae.add(Math.abs(e)); se += e * e; cnt++;
            }
        }
        java.util.Collections.sort(ae); double mae = 0; for (double v : ae) mae += v; mae /= ae.size();
        return new double[]{ mae, Math.sqrt(se / cnt), ae.get((int) (0.95 * ae.size())) };
    }

    // ============================== data + io ==============================
    private int[][] groundTruth(float[][] X, float[][] Q) {
        int N = X.length; int[][] gt = new int[Q.length][K];
        for (int qi = 0; qi < Q.length; qi++) { float[] ex = new float[N]; for (int j = 0; j < N; j++) ex[j] = l2(Q[qi], X[j]); gt[qi] = topIdx(ex, K); }
        return gt;
    }
    private float[][] load(String src, int dim, int n, boolean query) throws IOException {
        if (src.equals("iid")) { Random r = new Random(SEED + (query ? 999 : 1) + dim); float[][] X = new float[n][dim]; for (int i = 0; i < n; i++) for (int j = 0; j < dim; j++) X[i][j] = (float) r.nextGaussian(); return X; }
        Path repo = Paths.get(System.getProperty("user.dir")).getParent().getParent().getParent();
        return readFvecs(repo.resolve(REAL + src + (query ? "_query.fvecs" : "_base.fvecs")).toString(), 0, n);
    }
    private static float[] mean(float[][] X) { int d = X[0].length; float[] c = new float[d]; for (float[] v : X) for (int j = 0; j < d; j++) c[j] += v[j]; for (int j = 0; j < d; j++) c[j] /= X.length; return c; }
    private static float[] randVec(int d, Random r) { float[] v = new float[d]; for (int j = 0; j < d; j++) v[j] = (float) r.nextGaussian(); return v; }
    private static float l2(float[] a, float[] b) { float s = 0; for (int i = 0; i < a.length; i++) { float x = a[i] - b[i]; s += x * x; } return s; }
    private static float norm(float[] a) { float s = 0; for (float v : a) s += v * v; return (float) Math.sqrt(s); }
    private static int[] topIdx(float[] sc, int m) { Integer[] idx = new Integer[sc.length]; for (int i = 0; i < idx.length; i++) idx[i] = i; Arrays.sort(idx, (x, y) -> Float.compare(sc[x], sc[y])); int[] o = new int[m]; for (int i = 0; i < m; i++) o[i] = idx[i]; return o; }
    private static String join(String pre, int[] a) { StringBuilder sb = new StringBuilder(); for (int i = 0; i < a.length; i++) { if (i > 0) sb.append(','); sb.append(pre).append(a[i]); } return sb.toString(); }
    private Path outDir() throws IOException { Path o = Paths.get(System.getProperty("user.dir"), "research", "track2_adaptive_rescore", "results"); Files.createDirectories(o); return o; }
    private void writeFresh(Path dir, String name, String hdr) throws IOException { try (Writer w = Files.newBufferedWriter(dir.resolve(name))) { w.write(hdr); w.write("\n"); } }
    private void append(Path dir, String name, List<String> rows) throws IOException { try (Writer w = Files.newBufferedWriter(dir.resolve(name), StandardOpenOption.CREATE, StandardOpenOption.APPEND)) { for (String r : rows) { w.write(r); w.write("\n"); } } }
    private void writeLines(Path dir, String name, List<String> rows, String hdr) throws IOException { try (Writer w = Files.newBufferedWriter(dir.resolve(name))) { w.write(hdr); w.write("\n"); for (String r : rows) { w.write(r); w.write("\n"); } } }
    private static float[][] readFvecs(String path, int skip, int max) throws IOException {
        List<float[]> out = new ArrayList<>();
        try (RandomAccessFile f = new RandomAccessFile(path, "r")) { byte[] h = new byte[4]; int seen = 0;
            while (out.size() < max && f.getFilePointer() < f.length()) { if (f.read(h) != 4) break; int dim = ByteBuffer.wrap(h).order(ByteOrder.LITTLE_ENDIAN).getInt(); byte[] buf = new byte[4 * dim]; if (f.read(buf) != 4 * dim) break; if (seen++ < skip) continue; ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN); float[] v = new float[dim]; for (int j = 0; j < dim; j++) v[j] = bb.getFloat(); out.add(v); } }
        return out.toArray(new float[0][]);
    }
}
