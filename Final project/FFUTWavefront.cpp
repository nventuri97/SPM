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

void wavefront(std::vector<double> &M, uint64_t N, int numThreads) {
    ParallelFor pf(numThreads, true, true);
    
    for (int k = 1; k < N; ++k) {
        if(numThreads>N-k)
            numThreads--;
        pf.parallel_for(0, N-k, 1, [&, N, k](const int i) {
            double acc=0;
            for (int j = 0; j < k + 1; ++j) {
                acc += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
            }
            M[i * N + (i+k)] = cbrt(acc);
        }, numThreads);
    }
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