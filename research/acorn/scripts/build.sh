#!/usr/bin/env bash
# Build the ACORN-γ research prototype against a minimal subset of Faiss 1.11.0.
# No cmake / BLAS-install / OpenMP required: uses macOS Accelerate for BLAS and a
# no-op omp.h shim for a deterministic single-thread build.
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
ROOT="$(pwd)"
FAISS="$ROOT/jni/external/faiss"
SHIM="$ROOT/research/acorn/src/shim"
SRC="$ROOT/research/acorn/src"
BUILD="$ROOT/research/acorn/build"
OBJ="$BUILD/obj"
mkdir -p "$OBJ"

CXX=${CXX:-clang++}
JOBS=${JOBS:-8}
FLAGS=(-std=c++17 -O3 -DNDEBUG -DFINTEGER=int -I"$SHIM" -I"$FAISS" -I"$SRC"
       -Wno-unknown-pragmas -Wno-unused-variable -Wno-unused-function
       -Wno-deprecated-declarations -funroll-loops)
LDFLAGS=(-framework Accelerate)

# ---- Faiss minimal subset -------------------------------------------------
# Drives faiss::HNSW + IndexFlat directly (no IndexHNSW/IndexIDMap monolith).
FAISS_SRCS=(
  Index.cpp IndexFlat.cpp IndexFlatCodes.cpp
  impl/HNSW.cpp impl/IDSelector.cpp impl/AuxIndexStructures.cpp
  impl/FaissException.cpp
  utils/distances.cpp utils/distances_simd.cpp utils/extra_distances.cpp
  utils/distances_fused/distances_fused.cpp utils/distances_fused/simdlib_based.cpp
  utils/random.cpp utils/Heap.cpp utils/utils.cpp utils/sorting.cpp
  utils/partitioning.cpp impl/CodePacker.cpp
)

# ---- prototype sources ----------------------------------------------------
PROTO_SRCS=(
  acorn_hnsw.cpp acorn_gamma_builder.cpp racorn.cpp
)

compile() {  # <src-abs> <obj-abs>
  local src="$1" obj="$2"
  if [ ! -f "$obj" ] || [ "$src" -nt "$obj" ]; then
    echo "  CC $(basename "$obj")"
    "$CXX" "${FLAGS[@]}" -c "$src" -o "$obj"
  fi
}

echo "[1/3] Faiss subset"
for f in "${FAISS_SRCS[@]}"; do
  o="$OBJ/faiss_$(echo "$f" | tr '/' '_').o"
  compile "$FAISS/faiss/$f" "$o"
done

echo "[2/3] prototype"
for f in "${PROTO_SRCS[@]}"; do
  [ -f "$SRC/$f" ] || { echo "  (skip missing $f)"; continue; }
  compile "$SRC/$f" "$OBJ/proto_${f%.cpp}.o"
done

echo "[3/3] link targets"
ALL_OBJ=("$OBJ"/faiss_*.o)
for f in "${PROTO_SRCS[@]}"; do
  [ -f "$OBJ/proto_${f%.cpp}.o" ] && ALL_OBJ+=("$OBJ/proto_${f%.cpp}.o")
done
for tgt in "$@"; do
  echo "  LINK $tgt"
  "$CXX" "${FLAGS[@]}" "$SRC/$tgt.cpp" "${ALL_OBJ[@]}" "${LDFLAGS[@]}" -o "$BUILD/$tgt"
done
echo "done."
