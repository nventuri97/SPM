#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

void init_matrix(std::vector<double> &M, int N) {
    for (int i = 0; i < N; i++) {
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

void wavefront(std::vector<double> &M, int N, int numThreads) {
    omp_set_num_threads(numThreads);

    for (int k = 1; k < N; ++k) {
        #pragma omp parallel for
        for(int i=0; i<N-k; ++i) {
            double acc=0;
            for (int j = 0; j < k + 1; ++j) {
                acc += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
            }
            M[i * N + (i+k)]=cbrt(acc);
        };
    }
}

int main(int argc, char *argv[]) {
    uint64_t N = 512;    // default size of the matrix (NxN)
    int numThreads = 4;  // default number of threads

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
    // print_matrix(M, N);

    double start_time = omp_get_wtime();
    wavefront(M, N, numThreads);
    double end_time = omp_get_wtime();

    // print_matrix(M, N);
    std::cout << "Time: " << (end_time - start_time) * 1000 << " (ms)\n";

    std::cout <<M[N-1] << std::endl;

    return 0;
}
