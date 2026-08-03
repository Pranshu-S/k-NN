## Decision framework — best family per workload (k=10)

Winner = lowest mean latency among baselines reaching the recall tier (tries 0.90, else 0.80, 0.50, any).

| dataset | filter | sel_target | cand | tier | WINNER | recall | µs | runner-up | µs |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| n100k_d128_ip | adversarial | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 110.4 | - `-` | - |
| n100k_d128_ip | adversarial | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 133.9 | - `-` | - |
| n100k_d128_ip | adversarial | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 168.3 | - `-` | - |
| n100k_d128_ip | adversarial | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 238.2 | - `-` | - |
| n100k_d128_ip | adversarial | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 460.4 | - `-` | - |
| n100k_d128_ip | adversarial | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 865.2 | - `-` | - |
| n100k_d128_ip | adversarial | 0.250 | 25000 | 0.90 | exact `F_exact` | 1.000 | 2255.0 | - `-` | - |
| n100k_d128_ip | adversarial | 0.500 | 50000 | 0.90 | exact `F_exact` | 1.000 | 4417.3 | - `-` | - |
| n100k_d128_ip | cluster | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 111.0 | larger-M+ACORN `E_large24_acorn1` | 228.8 |
| n100k_d128_ip | cluster | 0.005 | 500 | 0.90 | ACORN-γ `C_acorn_g32` | 0.967 | 55.2 | exact `F_exact` | 135.3 |
| n100k_d128_ip | cluster | 0.010 | 1000 | 0.90 | ACORN-γ `C_acorn_g16` | 0.903 | 52.6 | larger-M+ACORN `E_large48_acorn1` | 159.1 |
| n100k_d128_ip | cluster | 0.020 | 2000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.909 | 92.2 | larger-M+ACORN `E_large24_acorn1` | 109.9 |
| n100k_d128_ip | cluster | 0.050 | 5000 | 0.90 | ACORN-γ `C_acorn_g16` | 0.928 | 122.7 | ACORN-1 `B_std_acorn1` | 204.7 |
| n100k_d128_ip | cluster | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 889.1 | larger-M+ACORN `E_large48_acorn1` | 1947.3 |
| n100k_d128_ip | cluster | 0.250 | 25000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.922 | 1676.2 | exact `F_exact` | 2814.8 |
| n100k_d128_ip | cluster | 0.500 | 50000 | 0.90 | larger-M+ACORN `E_large24_acorn1` | 0.907 | 556.8 | larger-M+ACORN `E_large48_acorn1` | 659.3 |
| n100k_d128_ip | correlated | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 118.9 | - `-` | - |
| n100k_d128_ip | correlated | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 138.7 | - `-` | - |
| n100k_d128_ip | correlated | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 171.8 | larger-M+ACORN `E_large48_acorn1` | 1380.5 |
| n100k_d128_ip | correlated | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 243.5 | - `-` | - |
| n100k_d128_ip | correlated | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 475.2 | - `-` | - |
| n100k_d128_ip | correlated | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 894.9 | - `-` | - |
| n100k_d128_ip | correlated | 0.250 | 25000 | 0.90 | exact `F_exact` | 1.000 | 2289.2 | - `-` | - |
| n100k_d128_ip | correlated | 0.500 | 50000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.954 | 450.4 | larger-M+ACORN `E_large24_acorn1` | 1045.5 |
| n100k_d128_ip | multicluster | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 110.7 | - `-` | - |
| n100k_d128_ip | multicluster | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 136.4 | - `-` | - |
| n100k_d128_ip | multicluster | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 167.0 | - `-` | - |
| n100k_d128_ip | multicluster | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 236.1 | - `-` | - |
| n100k_d128_ip | multicluster | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 461.8 | larger-M+ACORN `E_large48_acorn1` | 915.8 |
| n100k_d128_ip | multicluster | 0.100 | 10000 | 0.90 | ACORN-1 `B_std_acorn1` | 0.911 | 241.5 | larger-M+ACORN `E_large24_acorn1` | 260.5 |
| n100k_d128_ip | multicluster | 0.250 | 21174 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.916 | 1211.6 | exact `F_exact` | 1852.2 |
| n100k_d128_ip | multicluster | 0.500 | 35189 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.904 | 420.4 | larger-M+ACORN `E_large24_acorn1` | 646.0 |
| n100k_d128_ip | random | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 114.5 | - `-` | - |
| n100k_d128_ip | random | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 142.1 | standard-HNSW `A_std_standard` | 533.7 |
| n100k_d128_ip | random | 0.010 | 1000 | 0.90 | larger-M `D_large48_standard` | 0.916 | 137.2 | exact `F_exact` | 172.1 |
| n100k_d128_ip | random | 0.020 | 2000 | 0.90 | larger-M `D_large24_standard` | 0.936 | 128.7 | larger-M `D_large48_standard` | 134.8 |
| n100k_d128_ip | random | 0.050 | 5000 | 0.90 | larger-M `D_large48_standard` | 0.911 | 88.5 | standard-HNSW `A_std_standard` | 118.8 |
| n100k_d128_ip | random | 0.100 | 10000 | 0.90 | larger-M `D_large48_standard` | 0.921 | 88.9 | standard-HNSW `A_std_standard` | 120.5 |
| n100k_d128_ip | random | 0.250 | 25000 | 0.90 | larger-M `D_large24_standard` | 0.928 | 79.2 | larger-M `D_large48_standard` | 90.0 |
| n100k_d128_ip | random | 0.500 | 50000 | 0.90 | larger-M `D_large24_standard` | 0.942 | 81.2 | larger-M `D_large48_standard` | 89.3 |

**Workload wins by family:** **exact**=24, **larger-M+ACORN**=6, **larger-M**=6, **ACORN-γ**=3, **ACORN-1**=1

## Q1 — ACORN traversal vs standard filtered HNSW (recall @ ef=100, k=10)

| filter | sel | A standard recall | B ACORN-1 recall | Δ |
|---|---:|---:|---:|---:|
| adversarial | 0.001 | 0.046 | 0.000 | -0.046 |
| adversarial | 0.005 | 0.063 | 0.000 | -0.063 |
| adversarial | 0.010 | 0.069 | 0.000 | -0.069 |
| adversarial | 0.020 | 0.100 | 0.000 | -0.100 |
| adversarial | 0.050 | 0.177 | 0.051 | -0.126 |
| adversarial | 0.100 | 0.213 | 0.285 | +0.072 |
| adversarial | 0.250 | 0.406 | 0.521 | +0.115 |
| adversarial | 0.500 | 0.651 | 0.750 | +0.099 |
| cluster | 0.001 | 0.030 | 0.000 | -0.030 |
| cluster | 0.005 | 0.019 | 0.000 | -0.019 |
| cluster | 0.010 | 0.020 | 0.000 | -0.020 |
| cluster | 0.020 | 0.030 | 0.856 | +0.826 |
| cluster | 0.050 | 0.095 | 0.937 | +0.842 |
| cluster | 0.100 | 0.201 | 0.522 | +0.321 |
| cluster | 0.250 | 0.445 | 0.688 | +0.243 |
| cluster | 0.500 | 0.681 | 0.766 | +0.085 |
| correlated | 0.001 | 0.023 | 0.000 | -0.023 |
| correlated | 0.005 | 0.026 | 0.000 | -0.026 |
| correlated | 0.010 | 0.078 | 0.441 | +0.363 |
| correlated | 0.020 | 0.046 | 0.598 | +0.552 |
| correlated | 0.050 | 0.037 | 0.203 | +0.166 |
| correlated | 0.100 | 0.193 | 0.669 | +0.476 |
| correlated | 0.250 | 0.465 | 0.603 | +0.138 |
| correlated | 0.500 | 0.656 | 0.751 | +0.095 |
| multicluster | 0.001 | 0.090 | 0.000 | -0.090 |
| multicluster | 0.005 | 0.090 | 0.210 | +0.120 |
| multicluster | 0.010 | 0.119 | 0.180 | +0.061 |
| multicluster | 0.020 | 0.110 | 0.266 | +0.156 |
| multicluster | 0.050 | 0.109 | 0.000 | -0.109 |
| multicluster | 0.100 | 0.060 | 0.771 | +0.711 |
| multicluster | 0.250 | 0.470 | 0.678 | +0.208 |
| multicluster | 0.500 | 0.644 | 0.754 | +0.110 |
| random | 0.001 | 0.133 | 0.000 | -0.133 |
| random | 0.005 | 0.634 | 0.039 | -0.595 |
| random | 0.010 | 0.805 | 0.002 | -0.803 |
| random | 0.020 | 0.881 | 0.030 | -0.851 |
| random | 0.050 | 0.933 | 0.350 | -0.583 |
| random | 0.100 | 0.925 | 0.660 | -0.265 |
| random | 0.250 | 0.959 | 0.989 | +0.030 |
| random | 0.500 | 0.971 | 0.938 | -0.033 |

## Q8 — γ sensitivity (recall by γ, ACORN-γ, ef=250, k=10)

| dataset | filter | sel | γ=4 | γ=8 | γ=16 | γ=32 |
|---|---|---:|---:|---:|---:|---:|
| n100k_d128_ip | adversarial | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d128_ip | adversarial | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d128_ip | adversarial | 0.010 | 0.000 | 0.000 | 0.000 | 0.201 |
| n100k_d128_ip | adversarial | 0.020 | 0.000 | 0.000 | 0.000 | 0.124 |
| n100k_d128_ip | adversarial | 0.050 | 0.066 | 0.057 | 0.446 | 0.368 |
| n100k_d128_ip | adversarial | 0.100 | 0.085 | 0.198 | 0.573 | 0.511 |
| n100k_d128_ip | adversarial | 0.250 | 0.377 | 0.432 | 0.455 | 0.432 |
| n100k_d128_ip | adversarial | 0.500 | 0.658 | 0.640 | 0.704 | 0.492 |
| n100k_d128_ip | cluster | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d128_ip | cluster | 0.005 | 0.000 | 0.000 | 0.000 | 0.988 |
| n100k_d128_ip | cluster | 0.010 | 0.000 | 0.518 | 0.984 | 0.000 |
| n100k_d128_ip | cluster | 0.020 | 0.000 | 0.000 | 0.949 | 0.281 |
| n100k_d128_ip | cluster | 0.050 | 0.836 | 0.925 | 0.986 | 0.897 |
| n100k_d128_ip | cluster | 0.100 | 0.388 | 0.433 | 0.615 | 0.474 |
| n100k_d128_ip | cluster | 0.250 | 0.507 | 0.553 | 0.622 | 0.700 |
| n100k_d128_ip | cluster | 0.500 | 0.720 | 0.723 | 0.744 | 0.699 |
| n100k_d128_ip | correlated | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d128_ip | correlated | 0.005 | 0.000 | 0.000 | 0.000 | 0.430 |
| n100k_d128_ip | correlated | 0.010 | 0.000 | 0.000 | 0.662 | 0.676 |
| n100k_d128_ip | correlated | 0.020 | 0.700 | 0.692 | 0.665 | 0.690 |
| n100k_d128_ip | correlated | 0.050 | 0.223 | 0.228 | 0.830 | 0.812 |
| n100k_d128_ip | correlated | 0.100 | 0.518 | 0.546 | 0.634 | 0.726 |
| n100k_d128_ip | correlated | 0.250 | 0.260 | 0.266 | 0.284 | 0.394 |
| n100k_d128_ip | correlated | 0.500 | 0.481 | 0.591 | 0.621 | 0.604 |
| n100k_d128_ip | multicluster | 0.001 | 0.000 | 0.000 | 0.000 | 0.176 |
| n100k_d128_ip | multicluster | 0.005 | 0.210 | 0.210 | 0.146 | 0.146 |
| n100k_d128_ip | multicluster | 0.010 | 0.180 | 0.180 | 0.206 | 0.177 |
| n100k_d128_ip | multicluster | 0.020 | 0.235 | 0.134 | 0.547 | 0.315 |
| n100k_d128_ip | multicluster | 0.050 | 0.000 | 0.108 | 0.360 | 0.328 |
| n100k_d128_ip | multicluster | 0.100 | 0.619 | 0.694 | 0.778 | 0.724 |
| n100k_d128_ip | multicluster | 0.250 | 0.539 | 0.584 | 0.606 | 0.611 |
| n100k_d128_ip | multicluster | 0.500 | 0.692 | 0.694 | 0.680 | 0.667 |

## Index build & memory overhead (per graph, from build stats)

| dataset | baseline | build_ms | graph_bytes | bytes/vec | avg_deg0 | max_deg0 |
|---|---|---:|---:|---:|---:|---:|
| n100k_d128_ip | A_std_standard | 11837 | 14,033,032 | 140.3 | 24.5 | 32 |
| n100k_d128_ip | B_std_acorn1 | 11837 | 14,033,032 | 140.3 | 24.5 | 32 |
| n100k_d128_ip | C_acorn_g16 | 39364 | 23,728,392 | 237.3 | 24.6 | 40 |
| n100k_d128_ip | C_acorn_g32 | 99105 | 30,656,776 | 306.6 | 24.6 | 40 |
| n100k_d128_ip | C_acorn_g4 | 12336 | 18,532,104 | 185.3 | 24.5 | 40 |
| n100k_d128_ip | C_acorn_g8 | 19234 | 20,264,200 | 202.6 | 24.6 | 40 |
| n100k_d128_ip | D_large24_standard | 10422 | 20,424,136 | 204.2 | 30.4 | 48 |
| n100k_d128_ip | D_large48_standard | 9627 | 39,618,184 | 396.2 | 35.7 | 96 |
| n100k_d128_ip | E_large24_acorn1 | 10422 | 20,424,136 | 204.2 | 30.4 | 48 |
| n100k_d128_ip | E_large48_acorn1 | 9627 | 39,618,184 | 396.2 | 35.7 | 96 |

