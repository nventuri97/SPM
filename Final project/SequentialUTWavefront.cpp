#include <iostream>
#include <cmath>
#include <vector>
#include <hpc_helpers.hpp>

using namespace std;


void init_matrix(std::vector<double> &M, int N){
    for (int i=0;i<N;i++){
        M[i * N + i] = (i + 1) / static_cast<double>(N);
    }
}

void print_matrix(const std::vector<double> &M, int N) {
    printf("Matrice risultante:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", M[i * N + j]);
        }
        printf("\n");
    }
}

void wavefront(std::vector<double> &M, int N) {

    for (int k = 1; k < N; ++k) {
        for(int i=0; i<N-k; ++i) {
            double acc=0;
            for (int j = 0; j < k + 1; ++j) {
                acc += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
            }
            M[i * N + (i+k)]=cbrt(acc);
        }
    }
}

int main(int argc, char *argv[]) {

	uint64_t N = 512;    // default size of the matrix (NxN)
	
	if (argc != 1 && argc != 2) {
		std::printf("use: %s N numThreads\n", argv[0]);
		std::printf("     N size of the square matrix\n");
		return -1;
	}
	if (argc > 1) {
		N = std::stol(argv[1]);
		// if (argc > 2) {
		// 	numThreads = std::stol(argv[2]);
		// }
	}
	
    std::vector<double> M(N * N, 0.0);

    init_matrix(M, N);

	TIMERSTART(wavefront);
	wavefront(M, N); 
    TIMERSTOP(wavefront);

    // print_matrix(M, N);
    std::cout << M[N-1];

    return 0;
}