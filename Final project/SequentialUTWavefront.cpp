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

void print_matrix(const std::vector<double> &M, int N) {
    std::cout << "Matrice risultante:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << M[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

void wavefront(std::vector<double> &M, int N) {

    std::vector<double> horizontal, vertical;
    for (int i = 1; i < (N); ++i) {
        for(int j=0; j<N-i; j++) {
            horizontal=get_vector_m_k(M, N, j, i);
            vertical=get_vector_mk_k(M, N, j, i);
            double acc=0;
            for (int z = 0; z < i; ++z) {
                acc += horizontal[z] * vertical[z];
            }
            M[j * N + (j+i)]=cbrt(acc);
        };
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