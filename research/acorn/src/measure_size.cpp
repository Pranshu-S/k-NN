/* Measure HNSW graph + total index size for standard (M=16) vs ACORN-gamma (M=16*g). */
#include <cstdio>
#include <vector>
#include <faiss/IndexHNSW.h>
#include <faiss/utils/random.h>
using faiss::idx_t;
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):100000, d=128;
    faiss::RandomGenerator r(1); std::vector<float> X((size_t)N*d); for(auto&v:X)v=r.rand_float();
    long vecBytes=(long)N*d*4;
    int gammas[]={1,4,8,12,24}; // gamma=1 => standard M=16
    printf("n=%d d=%d ; vectors=%.1f MB (same for all)\n\n",N,d,vecBytes/1e6);
    printf("%-8s %-6s %-14s %-14s %-14s %-8s\n","gamma","M","graph MB","graph/node B","total MB","total x");
    double stdTotal=0;
    for(int g:gammas){ int M=16*g;
        auto* h=new faiss::IndexHNSWFlat(d,M,faiss::METRIC_L2); h->hnsw.efConstruction=100;
        h->add(N,X.data());
        long gBytes=(long)h->hnsw.neighbors.size()*sizeof(faiss::HNSW::storage_idx_t);
        long total=gBytes+vecBytes;
        if(g==1) stdTotal=total/1e6;
        printf("%-8d %-6d %-14.1f %-14.0f %-14.1f %-8.2f\n",
               g,M,gBytes/1e6,(double)gBytes/N,total/1e6,(total/1e6)/stdTotal);
        delete h;
    }
    return 0;
}
