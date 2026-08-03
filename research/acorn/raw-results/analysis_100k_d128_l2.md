## Decision framework — best family per workload (k=10)

Winner = lowest mean latency among baselines reaching the recall tier (tries 0.90, else 0.80, 0.50, any).

| dataset | filter | sel_target | cand | tier | WINNER | recall | µs | runner-up | µs |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| n100k_d128_l2 | adversarial | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 111.3 | - `-` | - |
| n100k_d128_l2 | adversarial | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 138.8 | - `-` | - |
| n100k_d128_l2 | adversarial | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 172.7 | - `-` | - |
| n100k_d128_l2 | adversarial | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 243.3 | - `-` | - |
| n100k_d128_l2 | adversarial | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 470.2 | - `-` | - |
| n100k_d128_l2 | adversarial | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 883.5 | - `-` | - |
| n100k_d128_l2 | adversarial | 0.250 | 25000 | 0.90 | exact `F_exact` | 1.000 | 2380.6 | - `-` | - |
| n100k_d128_l2 | adversarial | 0.500 | 50000 | 0.90 | exact `F_exact` | 1.000 | 4371.1 | - `-` | - |
| n100k_d128_l2 | cluster | 0.001 | 100 | 0.90 | ACORN-γ `C_acorn_g32` | 1.000 | 60.3 | ACORN-γ `C_acorn_g16` | 61.1 |
| n100k_d128_l2 | cluster | 0.005 | 500 | 0.90 | ACORN-γ `C_acorn_g32` | 0.970 | 52.2 | exact `F_exact` | 139.6 |
| n100k_d128_l2 | cluster | 0.010 | 1000 | 0.90 | ACORN-γ `C_acorn_g8` | 0.959 | 92.5 | ACORN-γ `C_acorn_g16` | 94.5 |
| n100k_d128_l2 | cluster | 0.020 | 2000 | 0.90 | larger-M+ACORN `E_large24_acorn1` | 0.913 | 117.1 | larger-M+ACORN `E_large48_acorn1` | 145.3 |
| n100k_d128_l2 | cluster | 0.050 | 5000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.966 | 269.4 | exact `F_exact` | 477.1 |
| n100k_d128_l2 | cluster | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 885.3 | - `-` | - |
| n100k_d128_l2 | cluster | 0.250 | 25000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.954 | 1028.8 | exact `F_exact` | 2356.7 |
| n100k_d128_l2 | cluster | 0.500 | 50000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.913 | 285.8 | larger-M+ACORN `E_large24_acorn1` | 319.5 |
| n100k_d128_l2 | correlated | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 112.1 | - `-` | - |
| n100k_d128_l2 | correlated | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 138.9 | - `-` | - |
| n100k_d128_l2 | correlated | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 173.6 | - `-` | - |
| n100k_d128_l2 | correlated | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 242.1 | - `-` | - |
| n100k_d128_l2 | correlated | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 470.9 | - `-` | - |
| n100k_d128_l2 | correlated | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 968.0 | - `-` | - |
| n100k_d128_l2 | correlated | 0.250 | 25000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.918 | 1773.0 | exact `F_exact` | 2279.1 |
| n100k_d128_l2 | correlated | 0.500 | 50000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.909 | 645.1 | exact `F_exact` | 4389.4 |
| n100k_d128_l2 | multicluster | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 111.7 | - `-` | - |
| n100k_d128_l2 | multicluster | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 135.8 | - `-` | - |
| n100k_d128_l2 | multicluster | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 169.3 | - `-` | - |
| n100k_d128_l2 | multicluster | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 242.1 | - `-` | - |
| n100k_d128_l2 | multicluster | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 470.7 | larger-M+ACORN `E_large48_acorn1` | 1007.3 |
| n100k_d128_l2 | multicluster | 0.100 | 10000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.917 | 175.4 | larger-M+ACORN `E_large24_acorn1` | 273.6 |
| n100k_d128_l2 | multicluster | 0.250 | 18988 | 0.90 | exact `F_exact` | 1.000 | 1695.9 | - `-` | - |
| n100k_d128_l2 | multicluster | 0.500 | 33761 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.929 | 380.4 | larger-M+ACORN `E_large24_acorn1` | 566.1 |
| n100k_d128_l2 | random | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 112.1 | - `-` | - |
| n100k_d128_l2 | random | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 138.9 | standard-HNSW `A_std_standard` | 461.4 |
| n100k_d128_l2 | random | 0.010 | 1000 | 0.90 | larger-M `D_large48_standard` | 0.921 | 132.3 | exact `F_exact` | 171.6 |
| n100k_d128_l2 | random | 0.020 | 2000 | 0.90 | larger-M `D_large24_standard` | 0.940 | 126.4 | larger-M `D_large48_standard` | 139.4 |
| n100k_d128_l2 | random | 0.050 | 5000 | 0.90 | larger-M `D_large48_standard` | 0.912 | 89.4 | standard-HNSW `A_std_standard` | 116.4 |
| n100k_d128_l2 | random | 0.100 | 10000 | 0.90 | larger-M `D_large24_standard` | 0.901 | 81.1 | larger-M `D_large48_standard` | 88.4 |
| n100k_d128_l2 | random | 0.250 | 25000 | 0.90 | larger-M `D_large24_standard` | 0.937 | 82.8 | larger-M `D_large48_standard` | 89.8 |
| n100k_d128_l2 | random | 0.500 | 50000 | 0.90 | standard-HNSW `A_std_standard` | 0.900 | 73.3 | larger-M `D_large48_standard` | 90.8 |

**Workload wins by family:** **exact**=23, **larger-M+ACORN**=8, **larger-M**=5, **ACORN-γ**=3, **standard-HNSW**=1

## Q1 — ACORN traversal vs standard filtered HNSW (recall @ ef=100, k=10)

| filter | sel | A standard recall | B ACORN-1 recall | Δ |
|---|---:|---:|---:|---:|
| adversarial | 0.001 | 0.024 | 0.000 | -0.024 |
| adversarial | 0.005 | 0.057 | 0.000 | -0.057 |
| adversarial | 0.010 | 0.067 | 0.000 | -0.067 |
| adversarial | 0.020 | 0.116 | 0.000 | -0.116 |
| adversarial | 0.050 | 0.180 | 0.313 | +0.133 |
| adversarial | 0.100 | 0.236 | 0.305 | +0.069 |
| adversarial | 0.250 | 0.452 | 0.481 | +0.029 |
| adversarial | 0.500 | 0.796 | 0.626 | -0.170 |
| cluster | 0.001 | 0.030 | 0.000 | -0.030 |
| cluster | 0.005 | 0.030 | 0.000 | -0.030 |
| cluster | 0.010 | 0.019 | 0.000 | -0.019 |
| cluster | 0.020 | 0.030 | 0.000 | -0.030 |
| cluster | 0.050 | 0.142 | 0.844 | +0.702 |
| cluster | 0.100 | 0.202 | 0.607 | +0.405 |
| cluster | 0.250 | 0.374 | 0.710 | +0.336 |
| cluster | 0.500 | 0.756 | 0.828 | +0.072 |
| correlated | 0.001 | 0.049 | 0.000 | -0.049 |
| correlated | 0.005 | 0.018 | 0.000 | -0.018 |
| correlated | 0.010 | 0.103 | 0.000 | -0.103 |
| correlated | 0.020 | 0.046 | 0.571 | +0.525 |
| correlated | 0.050 | 0.059 | 0.586 | +0.527 |
| correlated | 0.100 | 0.190 | 0.605 | +0.415 |
| correlated | 0.250 | 0.470 | 0.576 | +0.106 |
| correlated | 0.500 | 0.633 | 0.693 | +0.060 |
| multicluster | 0.001 | 0.089 | 0.000 | -0.089 |
| multicluster | 0.005 | 0.101 | 0.399 | +0.298 |
| multicluster | 0.010 | 0.120 | 0.248 | +0.128 |
| multicluster | 0.020 | 0.110 | 0.325 | +0.215 |
| multicluster | 0.050 | 0.107 | 0.000 | -0.107 |
| multicluster | 0.100 | 0.062 | 0.683 | +0.621 |
| multicluster | 0.250 | 0.364 | 0.645 | +0.281 |
| multicluster | 0.500 | 0.666 | 0.788 | +0.122 |
| random | 0.001 | 0.135 | 0.000 | -0.135 |
| random | 0.005 | 0.652 | 0.035 | -0.617 |
| random | 0.010 | 0.813 | 0.026 | -0.787 |
| random | 0.020 | 0.883 | 0.086 | -0.797 |
| random | 0.050 | 0.942 | 0.330 | -0.612 |
| random | 0.100 | 0.938 | 0.658 | -0.280 |
| random | 0.250 | 0.959 | 0.936 | -0.023 |
| random | 0.500 | 0.969 | 0.945 | -0.024 |

## Q8 — γ sensitivity (recall by γ, ACORN-γ, ef=250, k=10)

| dataset | filter | sel | γ=4 | γ=8 | γ=16 | γ=32 |
|---|---|---:|---:|---:|---:|---:|
| n100k_d128_l2 | adversarial | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d128_l2 | adversarial | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d128_l2 | adversarial | 0.010 | 0.000 | 0.000 | 0.000 | 0.551 |
| n100k_d128_l2 | adversarial | 0.020 | 0.000 | 0.000 | 0.000 | 0.530 |
| n100k_d128_l2 | adversarial | 0.050 | 0.152 | 0.150 | 0.345 | 0.572 |
| n100k_d128_l2 | adversarial | 0.100 | 0.300 | 0.295 | 0.444 | 0.574 |
| n100k_d128_l2 | adversarial | 0.250 | 0.367 | 0.558 | 0.619 | 0.582 |
| n100k_d128_l2 | adversarial | 0.500 | 0.535 | 0.610 | 0.598 | 0.503 |
| n100k_d128_l2 | cluster | 0.001 | 0.000 | 0.000 | 1.000 | 1.000 |
| n100k_d128_l2 | cluster | 0.005 | 0.000 | 0.000 | 0.000 | 1.000 |
| n100k_d128_l2 | cluster | 0.010 | 0.000 | 0.976 | 1.000 | 0.000 |
| n100k_d128_l2 | cluster | 0.020 | 0.000 | 0.000 | 0.951 | 0.956 |
| n100k_d128_l2 | cluster | 0.050 | 0.809 | 0.830 | 0.885 | 0.848 |
| n100k_d128_l2 | cluster | 0.100 | 0.313 | 0.468 | 0.549 | 0.653 |
| n100k_d128_l2 | cluster | 0.250 | 0.518 | 0.585 | 0.583 | 0.712 |
| n100k_d128_l2 | cluster | 0.500 | 0.667 | 0.679 | 0.679 | 0.659 |
| n100k_d128_l2 | correlated | 0.001 | 0.000 | 0.516 | 0.000 | 0.376 |
| n100k_d128_l2 | correlated | 0.005 | 0.000 | 0.000 | 0.000 | 0.492 |
| n100k_d128_l2 | correlated | 0.010 | 0.000 | 0.000 | 0.736 | 0.740 |
| n100k_d128_l2 | correlated | 0.020 | 0.643 | 0.637 | 0.682 | 0.638 |
| n100k_d128_l2 | correlated | 0.050 | 0.241 | 0.241 | 0.379 | 0.662 |
| n100k_d128_l2 | correlated | 0.100 | 0.373 | 0.682 | 0.676 | 0.678 |
| n100k_d128_l2 | correlated | 0.250 | 0.295 | 0.287 | 0.270 | 0.401 |
| n100k_d128_l2 | correlated | 0.500 | 0.565 | 0.632 | 0.653 | 0.647 |
| n100k_d128_l2 | multicluster | 0.001 | 0.000 | 0.000 | 0.000 | 0.194 |
| n100k_d128_l2 | multicluster | 0.005 | 0.399 | 0.399 | 0.123 | 0.339 |
| n100k_d128_l2 | multicluster | 0.010 | 0.248 | 0.248 | 0.066 | 0.451 |
| n100k_d128_l2 | multicluster | 0.020 | 0.442 | 0.325 | 0.657 | 0.770 |
| n100k_d128_l2 | multicluster | 0.050 | 0.000 | 0.595 | 0.763 | 0.696 |
| n100k_d128_l2 | multicluster | 0.100 | 0.603 | 0.741 | 0.757 | 0.654 |
| n100k_d128_l2 | multicluster | 0.250 | 0.534 | 0.636 | 0.608 | 0.604 |
| n100k_d128_l2 | multicluster | 0.500 | 0.555 | 0.618 | 0.618 | 0.627 |

## Index build & memory overhead (per graph, from build stats)

| dataset | baseline | build_ms | graph_bytes | bytes/vec | avg_deg0 | max_deg0 |
|---|---|---:|---:|---:|---:|---:|
| n100k_d128_l2 | A_std_standard | 11605 | 14,033,032 | 140.3 | 24.6 | 32 |
| n100k_d128_l2 | B_std_acorn1 | 11605 | 14,033,032 | 140.3 | 24.6 | 32 |
| n100k_d128_l2 | C_acorn_g16 | 39344 | 23,728,392 | 237.3 | 24.6 | 40 |
| n100k_d128_l2 | C_acorn_g32 | 101522 | 30,656,776 | 306.6 | 24.6 | 40 |
| n100k_d128_l2 | C_acorn_g4 | 12923 | 18,532,104 | 185.3 | 24.5 | 40 |
| n100k_d128_l2 | C_acorn_g8 | 20868 | 20,264,200 | 202.6 | 24.6 | 40 |
| n100k_d128_l2 | D_large24_standard | 10822 | 20,424,136 | 204.2 | 30.6 | 48 |
| n100k_d128_l2 | D_large48_standard | 10315 | 39,618,184 | 396.2 | 36.1 | 96 |
| n100k_d128_l2 | E_large24_acorn1 | 10822 | 20,424,136 | 204.2 | 30.6 | 48 |
| n100k_d128_l2 | E_large48_acorn1 | 10315 | 39,618,184 | 396.2 | 36.1 | 96 |

