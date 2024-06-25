#include <iostream>
#include <cmath>
#include <vector>
#include <ff/parallel_for.hpp>
#include <hpc_helpers.hpp>

#define MAX_THREADS 40
using namespace ff;


void init_matrix(std::vector<double> &M, int N){
    for (int i=0;i<N;i++){
        M[i * N + i] = (i + 1) / static_cast<double>(N);
    }
}

std::vector<double> get_vector_m_k(const std::vector<double>& M, int N, int m, int k) {
    std::vector<double> v(k);
    for (int i = 0; i < k; ++i) {
        v[i] = M[m * N + (m + i)]; // tutti gli elementi precedenti sulla stessa riga
    }
    return v;
}

std::vector<double> get_vector_mk_k(const std::vector<double>& M, int N, int m, int k) {
    std::vector<double> v(k);
    for (int i = 0; i < k; ++i) {
        v[i] = M[(m + i + 1) * N + (m + k)]; // tutti gli elementi precedenti sulla stessa colonna
    }
    std::reverse(v.begin(), v.end());
    return v;
}

void print_matrix(const std::vector<double> &M, uint64_t N) {
    std::cout << "Matrice risultante:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << M[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_vector(const std::vector<double> v, int dim, bool vertical){
    for(int i=0; i<dim;++i){
        std::cout<<v[i];
        if(vertical)
            std::cout<<std::endl;
        else
            std::cout<<" ";
    }
    std::cout<<std::endl;
}

void wavefront(std::vector<double> &M, uint64_t N, int numThreads) {
    ParallelFor pf(numThreads);
    
    for (int i = 1; i < N; ++i) {
        pf.parallel_for(0, N-i, 1, [&, N, i](const int j) {
            std::vector<double> horizontal, vertical;
            horizontal=get_vector_m_k(M, N, j, i);
            // print_vector(horizontal, i, false);
            vertical=get_vector_mk_k(M, N, j, i);
            // print_vector(vertical, i, true);
            double acc=0;
            for (int z = 0; z < i; ++z) {
                acc += horizontal[z] * vertical[z];
            }
            M[j * N + (j+i)] = cbrt(acc);
        });
    }
}

int main(int argc, char *argv[]) {

	uint64_t N = 512;    // default size of the matrix (NxN)
    int numThreads = MAX_NUM_THREADS;
	
	if (argc != 1 && argc != 2 && argc != 3) {
		std::printf("use: %s N numThreads\n", argv[0]);
		std::printf("     N size of the square matrix\n");
        std::printf("     numThread number of thread\n");
		return -1;
	}
	if (argc > 1) {
		N = std::stol(argv[1]);
		if (argc > 2) {
			numThreads = std::stol(argv[2]);
		}
	}
	
    std::vector<double> M(N * N, 0.0);

    init_matrix(M, N);

	ffTime(START_TIME);
	wavefront(M, N, numThreads);
    ffTime(STOP_TIME);

    // print_matrix(M, N);
    std::cout << "# elapsed time (wavefront): " << ffTime(GET_TIME)/1000  << "s" << std::endl;
    // std::cout<<M[N-1];

    return 0;
}