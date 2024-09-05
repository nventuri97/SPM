#include <iostream>
#include <cmath>
#include <vector>
#include <ff/parallel_for.hpp>
#include <hpc_helpers.hpp>

#define MAX_THREADS 32
using namespace ff;

void init_matrix(double *M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (j == i)
                M[i * N + j] = (i + 1) / static_cast<double>(N);
            else
                M[i * N + j] = 0.0;
        }
    }
}

void wavefront(double *M, uint64_t N, int numThreads) {
    ParallelFor pf(numThreads, true, true);
    
    for (uint64_t k = 1; k < N; ++k) {
        if((uint64_t) numThreads > N-k && numThreads>1)
            numThreads--;
        pf.parallel_for(0, N-k, 1, [&, N, k](const uint64_t i) {
            double dotProduct = 0.0;

            for (uint64_t j = 0; j < k + 1; ++j) {
                dotProduct += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
            }
            M[i * N + (i+k)] = cbrt(dotProduct);
        }, numThreads);
    }
}

void print_matrix(double *M, uint64_t N) {
    std::cout << "Matrice risultante:" << std::endl;
    for (uint64_t i = 0; i < N; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            std::cout << M[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {

	uint64_t N = 512;    // default size of the matrix (NxN)
    int numThreads = MAX_THREADS;
	
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
	
    double *M=new double[N*N];

    init_matrix(M, N);

	ffTime(START_TIME);
	wavefront(M, N, numThreads);
    ffTime(STOP_TIME);

    std::cout << "# elapsed time (wavefront): " << ffTime(GET_TIME)/1000  << "s" << std::endl;
    // print_matrix(M, N);
    std::cout << M[N-1] << std::endl;

    delete[] M;

    return 0;
}