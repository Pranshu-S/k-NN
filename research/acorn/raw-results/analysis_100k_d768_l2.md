## Decision framework — best family per workload (k=10)

Winner = lowest mean latency among baselines reaching the recall tier (tries 0.90, else 0.80, 0.50, any).

| dataset | filter | sel_target | cand | tier | WINNER | recall | µs | runner-up | µs |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| n100k_d768_l2 | adversarial | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 117.8 | - `-` | - |
| n100k_d768_l2 | adversarial | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 170.5 | - `-` | - |
| n100k_d768_l2 | adversarial | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 236.5 | - `-` | - |
| n100k_d768_l2 | adversarial | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 426.6 | - `-` | - |
| n100k_d768_l2 | adversarial | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 983.8 | - `-` | - |
| n100k_d768_l2 | adversarial | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 2572.6 | - `-` | - |
| n100k_d768_l2 | adversarial | 0.250 | 25000 | 0.90 | exact `F_exact` | 1.000 | 5913.8 | - `-` | - |
| n100k_d768_l2 | adversarial | 0.500 | 50000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.926 | 1475.3 | exact `F_exact` | 8611.6 |
| n100k_d768_l2 | cluster | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 118.0 | - `-` | - |
| n100k_d768_l2 | cluster | 0.005 | 500 | 0.90 | ACORN-γ `C_acorn_g16` | 0.926 | 86.5 | ACORN-γ `C_acorn_g8` | 143.3 |
| n100k_d768_l2 | cluster | 0.010 | 1000 | 0.90 | ACORN-γ `C_acorn_g8` | 0.941 | 148.1 | ACORN-γ `C_acorn_g4` | 148.4 |
| n100k_d768_l2 | cluster | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 372.7 | larger-M+ACORN `E_large48_acorn1` | 413.1 |
| n100k_d768_l2 | cluster | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 1018.2 | - `-` | - |
| n100k_d768_l2 | cluster | 0.100 | 10000 | 0.90 | larger-M+ACORN `E_large24_acorn1` | 0.919 | 992.1 | larger-M+ACORN `E_large48_acorn1` | 1121.7 |
| n100k_d768_l2 | cluster | 0.250 | 25000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.905 | 2289.2 | exact `F_exact` | 5659.3 |
| n100k_d768_l2 | cluster | 0.500 | 50000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.937 | 1337.4 | exact `F_exact` | 8270.1 |
| n100k_d768_l2 | correlated | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 116.9 | - `-` | - |
| n100k_d768_l2 | correlated | 0.005 | 500 | 0.90 | ACORN-γ `C_acorn_g16` | 0.939 | 139.3 | ACORN-γ `C_acorn_g32` | 139.7 |
| n100k_d768_l2 | correlated | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 228.2 | - `-` | - |
| n100k_d768_l2 | correlated | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 361.1 | larger-M+ACORN `E_large48_acorn1` | 470.1 |
| n100k_d768_l2 | correlated | 0.050 | 5000 | 0.90 | exact `F_exact` | 1.000 | 897.6 | - `-` | - |
| n100k_d768_l2 | correlated | 0.100 | 10000 | 0.90 | exact `F_exact` | 1.000 | 3692.5 | - `-` | - |
| n100k_d768_l2 | correlated | 0.250 | 25000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.905 | 1408.2 | exact `F_exact` | 10534.4 |
| n100k_d768_l2 | correlated | 0.500 | 50000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.967 | 1620.8 | exact `F_exact` | 8257.5 |
| n100k_d768_l2 | multicluster | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 117.3 | - `-` | - |
| n100k_d768_l2 | multicluster | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 169.6 | - `-` | - |
| n100k_d768_l2 | multicluster | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 232.8 | - `-` | - |
| n100k_d768_l2 | multicluster | 0.020 | 2000 | 0.90 | exact `F_exact` | 1.000 | 376.5 | - `-` | - |
| n100k_d768_l2 | multicluster | 0.050 | 5000 | 0.90 | larger-M+ACORN `E_large24_acorn1` | 0.928 | 391.0 | ACORN-γ `C_acorn_g16` | 391.1 |
| n100k_d768_l2 | multicluster | 0.100 | 10000 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.930 | 664.6 | exact `F_exact` | 2565.1 |
| n100k_d768_l2 | multicluster | 0.250 | 19251 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.911 | 1244.0 | exact `F_exact` | 4845.8 |
| n100k_d768_l2 | multicluster | 0.500 | 44355 | 0.90 | larger-M+ACORN `E_large48_acorn1` | 0.932 | 898.2 | larger-M+ACORN `E_large24_acorn1` | 1857.1 |
| n100k_d768_l2 | random | 0.001 | 100 | 0.90 | exact `F_exact` | 1.000 | 115.5 | - `-` | - |
| n100k_d768_l2 | random | 0.005 | 500 | 0.90 | exact `F_exact` | 1.000 | 165.2 | - `-` | - |
| n100k_d768_l2 | random | 0.010 | 1000 | 0.90 | exact `F_exact` | 1.000 | 229.1 | larger-M `D_large48_standard` | 316.9 |
| n100k_d768_l2 | random | 0.020 | 2000 | 0.90 | larger-M `D_large24_standard` | 0.923 | 293.5 | larger-M `D_large48_standard` | 317.2 |
| n100k_d768_l2 | random | 0.050 | 5000 | 0.90 | larger-M `D_large24_standard` | 0.934 | 293.5 | larger-M `D_large48_standard` | 317.4 |
| n100k_d768_l2 | random | 0.100 | 10000 | 0.90 | standard-HNSW `A_std_standard` | 0.917 | 268.2 | larger-M `D_large24_standard` | 301.8 |
| n100k_d768_l2 | random | 0.250 | 25000 | 0.90 | larger-M `D_large48_standard` | 0.909 | 241.3 | standard-HNSW `A_std_standard` | 285.2 |
| n100k_d768_l2 | random | 0.500 | 50000 | 0.90 | larger-M `D_large48_standard` | 0.931 | 247.1 | standard-HNSW `A_std_standard` | 269.6 |

**Workload wins by family:** **exact**=22, **larger-M+ACORN**=10, **larger-M**=4, **ACORN-γ**=3, **standard-HNSW**=1

## Q1 — ACORN traversal vs standard filtered HNSW (recall @ ef=100, k=10)

| filter | sel | A standard recall | B ACORN-1 recall | Δ |
|---|---:|---:|---:|---:|
| adversarial | 0.001 | 0.095 | 0.000 | -0.095 |
| adversarial | 0.005 | 0.120 | 0.235 | +0.115 |
| adversarial | 0.010 | 0.133 | 0.376 | +0.243 |
| adversarial | 0.020 | 0.167 | 0.299 | +0.132 |
| adversarial | 0.050 | 0.264 | 0.112 | -0.152 |
| adversarial | 0.100 | 0.448 | 0.201 | -0.247 |
| adversarial | 0.250 | 0.657 | 0.501 | -0.156 |
| adversarial | 0.500 | 0.785 | 0.667 | -0.118 |
| cluster | 0.001 | 0.020 | 0.000 | -0.020 |
| cluster | 0.005 | 0.010 | 1.000 | +0.990 |
| cluster | 0.010 | 0.045 | 0.997 | +0.952 |
| cluster | 0.020 | 0.065 | 0.753 | +0.688 |
| cluster | 0.050 | 0.171 | 0.719 | +0.548 |
| cluster | 0.100 | 0.233 | 0.665 | +0.432 |
| cluster | 0.250 | 0.503 | 0.547 | +0.044 |
| cluster | 0.500 | 0.632 | 0.663 | +0.031 |
| correlated | 0.001 | 0.034 | 0.000 | -0.034 |
| correlated | 0.005 | 0.029 | 0.000 | -0.029 |
| correlated | 0.010 | 0.070 | 0.657 | +0.587 |
| correlated | 0.020 | 0.083 | 0.767 | +0.684 |
| correlated | 0.050 | 0.137 | 0.292 | +0.155 |
| correlated | 0.100 | 0.248 | 0.380 | +0.132 |
| correlated | 0.250 | 0.417 | 0.670 | +0.253 |
| correlated | 0.500 | 0.598 | 0.673 | +0.075 |
| multicluster | 0.001 | 0.129 | 0.000 | -0.129 |
| multicluster | 0.005 | 0.089 | 0.186 | +0.097 |
| multicluster | 0.010 | 0.137 | 0.507 | +0.370 |
| multicluster | 0.020 | 0.128 | 0.000 | -0.128 |
| multicluster | 0.050 | 0.082 | 0.710 | +0.628 |
| multicluster | 0.100 | 0.173 | 0.645 | +0.472 |
| multicluster | 0.250 | 0.283 | 0.676 | +0.393 |
| multicluster | 0.500 | 0.769 | 0.747 | -0.022 |
| random | 0.001 | 0.126 | 0.000 | -0.126 |
| random | 0.005 | 0.655 | 0.019 | -0.636 |
| random | 0.010 | 0.809 | 0.000 | -0.809 |
| random | 0.020 | 0.853 | 0.091 | -0.762 |
| random | 0.050 | 0.884 | 0.370 | -0.514 |
| random | 0.100 | 0.917 | 0.650 | -0.267 |
| random | 0.250 | 0.931 | 0.980 | +0.049 |
| random | 0.500 | 0.935 | 0.950 | +0.015 |

## Q8 — γ sensitivity (recall by γ, ACORN-γ, ef=250, k=10)

| dataset | filter | sel | γ=4 | γ=8 | γ=16 | γ=32 |
|---|---|---:|---:|---:|---:|---:|
| n100k_d768_l2 | adversarial | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d768_l2 | adversarial | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d768_l2 | adversarial | 0.010 | 0.000 | 0.000 | 0.000 | 0.426 |
| n100k_d768_l2 | adversarial | 0.020 | 0.000 | 0.370 | 0.374 | 0.513 |
| n100k_d768_l2 | adversarial | 0.050 | 0.110 | 0.219 | 0.417 | 0.477 |
| n100k_d768_l2 | adversarial | 0.100 | 0.126 | 0.306 | 0.343 | 0.368 |
| n100k_d768_l2 | adversarial | 0.250 | 0.416 | 0.544 | 0.580 | 0.415 |
| n100k_d768_l2 | adversarial | 0.500 | 0.487 | 0.514 | 0.561 | 0.383 |
| n100k_d768_l2 | cluster | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d768_l2 | cluster | 0.005 | 0.000 | 0.958 | 1.000 | 0.000 |
| n100k_d768_l2 | cluster | 0.010 | 0.999 | 1.000 | 1.000 | 1.000 |
| n100k_d768_l2 | cluster | 0.020 | 0.000 | 0.000 | 0.000 | 0.569 |
| n100k_d768_l2 | cluster | 0.050 | 0.400 | 0.488 | 0.551 | 0.641 |
| n100k_d768_l2 | cluster | 0.100 | 0.684 | 0.709 | 0.687 | 0.688 |
| n100k_d768_l2 | cluster | 0.250 | 0.479 | 0.529 | 0.560 | 0.516 |
| n100k_d768_l2 | cluster | 0.500 | 0.498 | 0.556 | 0.554 | 0.590 |
| n100k_d768_l2 | correlated | 0.001 | 0.000 | 0.847 | 0.847 | 0.856 |
| n100k_d768_l2 | correlated | 0.005 | 0.000 | 0.876 | 0.963 | 0.967 |
| n100k_d768_l2 | correlated | 0.010 | 0.641 | 0.652 | 0.655 | 0.655 |
| n100k_d768_l2 | correlated | 0.020 | 0.040 | 0.473 | 0.795 | 0.646 |
| n100k_d768_l2 | correlated | 0.050 | 0.219 | 0.239 | 0.613 | 0.610 |
| n100k_d768_l2 | correlated | 0.100 | 0.124 | 0.295 | 0.332 | 0.369 |
| n100k_d768_l2 | correlated | 0.250 | 0.617 | 0.628 | 0.643 | 0.624 |
| n100k_d768_l2 | correlated | 0.500 | 0.633 | 0.678 | 0.709 | 0.674 |
| n100k_d768_l2 | multicluster | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 |
| n100k_d768_l2 | multicluster | 0.005 | 0.000 | 0.000 | 0.383 | 0.561 |
| n100k_d768_l2 | multicluster | 0.010 | 0.293 | 0.292 | 0.508 | 0.300 |
| n100k_d768_l2 | multicluster | 0.020 | 0.000 | 0.150 | 0.287 | 0.273 |
| n100k_d768_l2 | multicluster | 0.050 | 0.701 | 0.698 | 0.938 | 0.937 |
| n100k_d768_l2 | multicluster | 0.100 | 0.356 | 0.591 | 0.603 | 0.762 |
| n100k_d768_l2 | multicluster | 0.250 | 0.748 | 0.754 | 0.758 | 0.641 |
| n100k_d768_l2 | multicluster | 0.500 | 0.565 | 0.614 | 0.738 | 0.591 |

## Index build & memory overhead (per graph, from build stats)

| dataset | baseline | build_ms | graph_bytes | bytes/vec | avg_deg0 | max_deg0 |
|---|---|---:|---:|---:|---:|---:|
| n100k_d768_l2 | A_std_standard | 41700 | 14,033,032 | 140.3 | 25.8 | 32 |
| n100k_d768_l2 | B_std_acorn1 | 41700 | 14,033,032 | 140.3 | 25.8 | 32 |
| n100k_d768_l2 | C_acorn_g16 | 95724 | 23,728,392 | 237.3 | 24.3 | 40 |
| n100k_d768_l2 | C_acorn_g32 | 242975 | 30,656,776 | 306.6 | 24.3 | 40 |
| n100k_d768_l2 | C_acorn_g4 | 35004 | 18,532,104 | 185.3 | 24.2 | 40 |
| n100k_d768_l2 | C_acorn_g8 | 55273 | 20,264,200 | 202.6 | 24.3 | 40 |
| n100k_d768_l2 | D_large24_standard | 36202 | 20,424,136 | 204.2 | 32.9 | 48 |
| n100k_d768_l2 | D_large48_standard | 26937 | 39,618,184 | 396.2 | 41.1 | 96 |
| n100k_d768_l2 | E_large24_acorn1 | 36202 | 20,424,136 | 204.2 | 32.9 | 48 |
| n100k_d768_l2 | E_large48_acorn1 | 26937 | 39,618,184 | 396.2 | 41.1 | 96 |

