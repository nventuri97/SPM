#include <iostream>
#include <cmath>
#include <vector>
#include <hpc_helpers.hpp>

using namespace std;


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

void print_matrix(double *M, u_int64_t N) {
    printf("Matrice risultante:\n");
    for (u_int64_t i = 0; i < N; ++i) {
        for (u_int64_t j = 0; j < N; ++j) {
            printf("%f ", M[i * N + j]);
        }
        printf("\n");
    }
}

void wavefront(double *M, u_int64_t N) {

    for (u_int64_t k = 1; k < N; ++k) {
        for(u_int64_t i=0; i<N-k; ++i) {
            double dotProduct = 0.0;

            for (u_int64_t j = 1; j < k + 1; ++j) {
                dotProduct += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
            }
            M[i * N + (i+k)]=cbrt(dotProduct);
        }
    }
}

int main(int argc, char *argv[]) {

	uint64_t N = 512;    // default size of the matrix (NxN)
	
	if (argc != 1 && argc != 2) {
		std::printf("use: %s N\n", argv[0]);
		std::printf("     N size of the square matrix\n");
		return -1;
	}

	if (argc > 1) {
		N = std::stol(argv[1]);
	}
	
    double *M=new double[N*N];

    init_matrix(M, N);

	TIMERSTART(wavefront);
	wavefront(M, N); 
    TIMERSTOP(wavefront);

    // print_matrix(M, N);
    std::cout << M[N-1] << endl;

    return 0;
}