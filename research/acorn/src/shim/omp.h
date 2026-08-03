// Minimal no-op OpenMP shim for the ACORN-γ research prototype.
//
// The prototype compiles a minimal subset of Faiss 1.11.0 with clang++ WITHOUT
// linking libomp, so that the benchmark harness runs deterministically on a
// single thread (concurrency is driven at the harness level with std::thread).
// The `#pragma omp ...` directives in Faiss are simply ignored by the compiler
// (unknown pragmas), and the few omp_* API calls Faiss makes on the HNSW build
// path are satisfied by these inline no-ops.
//
// This is a TEST-ONLY build shim. It is NOT used by the production k-NN JNI
// build, which links real OpenMP.
#pragma once

#include <cstdint>

typedef struct { int _dummy; } omp_lock_t;

static inline void   omp_init_lock(omp_lock_t*)    {}
static inline void   omp_destroy_lock(omp_lock_t*) {}
static inline void   omp_set_lock(omp_lock_t*)     {}
static inline void   omp_unset_lock(omp_lock_t*)   {}
static inline int    omp_test_lock(omp_lock_t*)    { return 1; }

static inline int    omp_get_max_threads()   { return 1; }
static inline int    omp_get_num_threads()   { return 1; }
static inline int    omp_get_thread_num()    { return 0; }
static inline void   omp_set_num_threads(int){}
static inline int    omp_get_num_procs()     { return 1; }
static inline int    omp_in_parallel()       { return 0; }
static inline double omp_get_wtime()         { return 0.0; }
static inline void   omp_set_nested(int)     {}
static inline int    omp_get_nested()        { return 0; }
