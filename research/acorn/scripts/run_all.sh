#!/usr/bin/env bash
# Reproduce the ACORN-γ benchmark matrix. Writes CSVs to research/acorn/raw-results/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
RR=research/acorn/raw-results
bash research/acorn/scripts/build.sh bench

# Primary matrix: 100K, dims {128,768}, metrics {l2,ip}; gammas {4,8,16,32}.
./research/acorn/build/bench 100000 128 l2 100 42 $RR/bench_100k_d128_l2.csv n100k_d128_l2
./research/acorn/build/bench 100000 128 ip 100 42 $RR/bench_100k_d128_ip.csv n100k_d128_ip
./research/acorn/build/bench 100000 768 l2 100 42 $RR/bench_100k_d768_l2.csv n100k_d768_l2

# Scale test: 1M, gammas capped to {4,8,16} to bound build cost.
ACORN_GAMMAS=4,8,16 ./research/acorn/build/bench 1000000 128 l2 60 42 $RR/bench_1m_d128_l2.csv n1m_d128_l2

# Concurrency scaling (1/4/8/16 threads) over an ACORN-γ index.
./research/acorn/build/bench concurrency 100000 128 l2 $RR/concurrency.csv conc_100k_d128
echo "all runs complete."
