## Decision framework — best family per workload (k=10)

Winner = lowest mean latency among baselines reaching the recall tier (tries 0.90, else 0.80, 0.50, any).

| dataset | filter | sel_target | cand | tier | WINNER | recall | µs | runner-up | µs |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| n1m_d128_l2 | adversarial | 0.001 | 1000 | 0.90 | exact `F_exact` | 1.000 | 1120.6 | - `-` | - |
| n1m_d128_l2 | adversarial | 0.005 | 5000 | 0.90 | exact `F_exact` | 1.000 | 1465.8 | - `-` | - |
| n1m_d128_l2 | adversarial | 0.010 | 10000 | 0.90 | exact `F_exact` | 1.000 | 1900.4 | - `-` | - |
| n1m_d128_l2 | adversarial | 0.020 | 20000 | 0.90 | exact `F_exact` | 1.000 | 3075.2 | - `-` | - |
| n1m_d128_l2 | adversarial | 0.050 | 50000 | 0.90 | exact `F_exact` | 1.000 | 8850.7 | - `-` | - |
| n1m_d128_l2 | adversarial | 0.100 | 100000 | 0.90 | exact `F_exact` | 1.000 | 14719.7 | - `-` | - |
| n1m_d128_l2 | adversarial | 0.250 | 250000 | 0.90 | exact `F_exact` | 1.000 | 31241.8 | - `-` | - |
| n1m_d128_l2 | adversarial | 0.500 | 500000 | 0.90 | exact `F_exact` | 1.000 | 50207.0 | - `-` | - |
| n1m_d128_l2 | cluster | 0.001 | 1000 | 0.90 | ACORN-γ `C_acorn_g4` | 0.974 | 100.7 | ACORN-γ `C_acorn_g8` | 102.1 |
| n1m_d128_l2 | cluster | 0.005 | 5000 | 0.90 | exact `F_exact` | 1.000 | 1543.7 | - `-` | - |
| n1m_d128_l2 | cluster | 0.010 | 10000 | 0.90 | exact `F_exact` | 1.000 | 1973.0 | - `-` | - |
| n1m_d128_l2 | cluster | 0.020 | 20000 | 0.90 | exact `F_exact` | 1.000 | 3387.3 | - `-` | - |
| n1m_d128_l2 | cluster | 0.050 | 50000 | 0.90 | exact `F_exact` | 1.000 | 8259.6 | - `-` | - |
| n1m_d128_l2 | cluster | 0.100 | 100000 | 0.90 | exact `F_exact` | 1.000 | 14365.8 | - `-` | - |
| n1m_d128_l2 | cluster | 0.250 | 250000 | 0.90 | exact `F_exact` | 1.000 | 31580.7 | - `-` | - |
| n1m_d128_l2 | cluster | 0.500 | 500000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.908 | 1172.3 | exact `F_exact` | 50438.6 |
| n1m_d128_l2 | correlated | 0.001 | 1000 | 0.90 | exact `F_exact` | 1.000 | 1119.3 | - `-` | - |
| n1m_d128_l2 | correlated | 0.005 | 5000 | 0.90 | exact `F_exact` | 1.000 | 1475.1 | - `-` | - |
| n1m_d128_l2 | correlated | 0.010 | 10000 | 0.90 | exact `F_exact` | 1.000 | 1914.5 | - `-` | - |
| n1m_d128_l2 | correlated | 0.020 | 20000 | 0.90 | exact `F_exact` | 1.000 | 3123.2 | - `-` | - |
| n1m_d128_l2 | correlated | 0.050 | 50000 | 0.90 | exact `F_exact` | 1.000 | 7625.8 | - `-` | - |
| n1m_d128_l2 | correlated | 0.100 | 100000 | 0.90 | exact `F_exact` | 1.000 | 15538.3 | - `-` | - |
| n1m_d128_l2 | correlated | 0.250 | 250000 | 0.90 | exact `F_exact` | 1.000 | 31291.2 | - `-` | - |
| n1m_d128_l2 | correlated | 0.500 | 500000 | 0.90 | exact `F_exact` | 1.000 | 49899.9 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.001 | 1000 | 0.90 | exact `F_exact` | 1.000 | 1137.9 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.005 | 5000 | 0.90 | exact `F_exact` | 1.000 | 1607.9 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.010 | 10000 | 0.90 | exact `F_exact` | 1.000 | 2283.8 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.020 | 19807 | 0.90 | exact `F_exact` | 1.000 | 3267.4 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.050 | 49287 | 0.90 | exact `F_exact` | 1.000 | 8890.0 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.100 | 89378 | 0.90 | exact `F_exact` | 1.000 | 18167.7 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.250 | 216358 | 0.90 | exact `F_exact` | 1.000 | 27649.2 | - `-` | - |
| n1m_d128_l2 | multicluster | 0.500 | 364450 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.928 | 1780.1 | exact `F_exact` | 40363.3 |
| n1m_d128_l2 | random | 0.001 | 1000 | 0.90 | exact `F_exact` | 1.000 | 1149.2 | - `-` | - |
| n1m_d128_l2 | random | 0.005 | 5000 | 0.90 | exact `F_exact` | 1.000 | 1509.8 | - `-` | - |
| n1m_d128_l2 | random | 0.010 | 10000 | 0.90 | larger-M `D_large48_standard` | 0.946 | 176.6 | standard-HNSW `A_std_standard` | 290.0 |
| n1m_d128_l2 | random | 0.020 | 20000 | 0.90 | larger-M `D_large24_standard` | 0.942 | 162.2 | larger-M `D_large48_standard` | 196.9 |
| n1m_d128_l2 | random | 0.050 | 50000 | 0.90 | larger-M `D_large24_standard` | 0.966 | 170.4 | larger-M `D_large48_standard` | 172.5 |
| n1m_d128_l2 | random | 0.100 | 100000 | 0.90 | larger-M `D_large24_standard` | 0.904 | 111.3 | larger-M `D_large48_standard` | 124.1 |
| n1m_d128_l2 | random | 0.250 | 250000 | 0.90 | larger-M `D_large24_standard` | 0.928 | 105.3 | larger-M `D_large48_standard` | 135.9 |
| n1m_d128_l2 | random | 0.500 | 500000 | 0.90 | standard-HNSW `A_std_standard` | 0.936 | 167.2 | larger-M `D_large48_standard` | 203.8 |

**Workload wins by family:** **exact**=31, **larger-M**=5, **larger-M+ACORN**=2, **ACORN-γ**=1, **standard-HNSW**=1

## Q1 — ACORN traversal vs standard filtered HNSW (recall @ ef=100, k=10)

| filter | sel | A standard recall | B ACORN-1 recall | Δ |
|---|---:|---:|---:|---:|
| adversarial | 0.001 | 0.020 | 0.000 | -0.020 |
| adversarial | 0.005 | 0.058 | 0.000 | -0.058 |
| adversarial | 0.010 | 0.078 | 0.000 | -0.078 |
| adversarial | 0.020 | 0.126 | 0.000 | -0.126 |
| adversarial | 0.050 | 0.204 | 0.042 | -0.162 |
| adversarial | 0.100 | 0.266 | 0.218 | -0.048 |
| adversarial | 0.250 | 0.402 | 0.378 | -0.024 |
| adversarial | 0.500 | 0.698 | 0.424 | -0.274 |
| cluster | 0.001 | 0.000 | 1.000 | +1.000 |
| cluster | 0.005 | 0.020 | 0.452 | +0.432 |
| cluster | 0.010 | 0.038 | 0.376 | +0.338 |
| cluster | 0.020 | 0.018 | 0.292 | +0.274 |
| cluster | 0.050 | 0.116 | 0.478 | +0.362 |
| cluster | 0.100 | 0.254 | 0.434 | +0.180 |
| cluster | 0.250 | 0.372 | 0.392 | +0.020 |
| cluster | 0.500 | 0.752 | 0.578 | -0.174 |
| correlated | 0.001 | 0.000 | 0.000 | +0.000 |
| correlated | 0.005 | 0.006 | 0.454 | +0.448 |
| correlated | 0.010 | 0.022 | 0.342 | +0.320 |
| correlated | 0.020 | 0.042 | 0.320 | +0.278 |
| correlated | 0.050 | 0.076 | 0.444 | +0.368 |
| correlated | 0.100 | 0.138 | 0.300 | +0.162 |
| correlated | 0.250 | 0.398 | 0.378 | -0.020 |
| correlated | 0.500 | 0.610 | 0.490 | -0.120 |
| multicluster | 0.001 | 0.004 | 0.000 | -0.004 |
| multicluster | 0.005 | 0.040 | 0.000 | -0.040 |
| multicluster | 0.010 | 0.040 | 0.254 | +0.214 |
| multicluster | 0.020 | 0.038 | 0.348 | +0.310 |
| multicluster | 0.050 | 0.144 | 0.372 | +0.228 |
| multicluster | 0.100 | 0.322 | 0.452 | +0.130 |
| multicluster | 0.250 | 0.444 | 0.448 | +0.004 |
| multicluster | 0.500 | 0.604 | 0.352 | -0.252 |
| random | 0.001 | 0.146 | 0.000 | -0.146 |
| random | 0.005 | 0.622 | 0.000 | -0.622 |
| random | 0.010 | 0.782 | 0.000 | -0.782 |
| random | 0.020 | 0.876 | 0.060 | -0.816 |
| random | 0.050 | 0.882 | 0.254 | -0.628 |
| random | 0.100 | 0.908 | 0.496 | -0.412 |
| random | 0.250 | 0.916 | 0.630 | -0.286 |
| random | 0.500 | 0.936 | 0.730 | -0.206 |

## Q8 — γ sensitivity (recall by γ, ACORN-γ, ef=250, k=10)

| dataset | filter | sel | γ=4 | γ=8 | γ=16 |
|---|---|---:|---:|---:|---:|
| n1m_d128_l2 | adversarial | 0.001 | 0.000 | 0.000 | 0.000 |
| n1m_d128_l2 | adversarial | 0.005 | 0.000 | 0.000 | 0.024 |
| n1m_d128_l2 | adversarial | 0.010 | 0.000 | 0.000 | 0.090 |
| n1m_d128_l2 | adversarial | 0.020 | 0.038 | 0.090 | 0.128 |
| n1m_d128_l2 | adversarial | 0.050 | 0.006 | 0.028 | 0.084 |
| n1m_d128_l2 | adversarial | 0.100 | 0.088 | 0.148 | 0.164 |
| n1m_d128_l2 | adversarial | 0.250 | 0.102 | 0.158 | 0.176 |
| n1m_d128_l2 | adversarial | 0.500 | 0.216 | 0.238 | 0.322 |
| n1m_d128_l2 | cluster | 0.001 | 1.000 | 1.000 | 1.000 |
| n1m_d128_l2 | cluster | 0.005 | 0.000 | 0.592 | 0.572 |
| n1m_d128_l2 | cluster | 0.010 | 0.158 | 0.360 | 0.650 |
| n1m_d128_l2 | cluster | 0.020 | 0.000 | 0.200 | 0.244 |
| n1m_d128_l2 | cluster | 0.050 | 0.210 | 0.244 | 0.268 |
| n1m_d128_l2 | cluster | 0.100 | 0.250 | 0.308 | 0.424 |
| n1m_d128_l2 | cluster | 0.250 | 0.336 | 0.420 | 0.378 |
| n1m_d128_l2 | cluster | 0.500 | 0.598 | 0.566 | 0.536 |
| n1m_d128_l2 | correlated | 0.001 | 0.000 | 0.000 | 0.000 |
| n1m_d128_l2 | correlated | 0.005 | 0.602 | 0.600 | 0.540 |
| n1m_d128_l2 | correlated | 0.010 | 0.000 | 0.000 | 0.370 |
| n1m_d128_l2 | correlated | 0.020 | 0.226 | 0.286 | 0.426 |
| n1m_d128_l2 | correlated | 0.050 | 0.324 | 0.422 | 0.540 |
| n1m_d128_l2 | correlated | 0.100 | 0.124 | 0.170 | 0.272 |
| n1m_d128_l2 | correlated | 0.250 | 0.280 | 0.392 | 0.420 |
| n1m_d128_l2 | correlated | 0.500 | 0.362 | 0.462 | 0.520 |
| n1m_d128_l2 | multicluster | 0.001 | 0.000 | 0.000 | 0.136 |
| n1m_d128_l2 | multicluster | 0.005 | 0.000 | 0.000 | 0.210 |
| n1m_d128_l2 | multicluster | 0.010 | 0.000 | 0.000 | 0.000 |
| n1m_d128_l2 | multicluster | 0.020 | 0.050 | 0.190 | 0.324 |
| n1m_d128_l2 | multicluster | 0.050 | 0.184 | 0.226 | 0.334 |
| n1m_d128_l2 | multicluster | 0.100 | 0.174 | 0.234 | 0.372 |
| n1m_d128_l2 | multicluster | 0.250 | 0.332 | 0.386 | 0.470 |
| n1m_d128_l2 | multicluster | 0.500 | 0.434 | 0.470 | 0.378 |

## Index build & memory overhead (per graph, from build stats)

| dataset | baseline | build_ms | graph_bytes | bytes/vec | avg_deg0 | max_deg0 |
|---|---|---:|---:|---:|---:|---:|
| n1m_d128_l2 | A_std_standard | 140884 | 140,258,120 | 140.3 | 24.5 | 32 |
| n1m_d128_l2 | B_std_acorn1 | 140884 | 140,258,120 | 140.3 | 24.5 | 32 |
| n1m_d128_l2 | C_acorn_g16 | 1186951 | 236,129,800 | 236.1 | 24.6 | 40 |
| n1m_d128_l2 | C_acorn_g4 | 222708 | 185,032,456 | 185.0 | 24.6 | 40 |
| n1m_d128_l2 | C_acorn_g8 | 316873 | 202,064,904 | 202.1 | 24.6 | 40 |
| n1m_d128_l2 | D_large24_standard | 132908 | 204,166,792 | 204.2 | 30.4 | 48 |
| n1m_d128_l2 | D_large48_standard | 142299 | 396,080,968 | 396.1 | 35.9 | 96 |
| n1m_d128_l2 | E_large24_acorn1 | 132908 | 204,166,792 | 204.2 | 30.4 | 48 |
| n1m_d128_l2 | E_large48_acorn1 | 142299 | 396,080,968 | 396.1 | 35.9 | 96 |

