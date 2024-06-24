#include <iostream>
#include <cmath>
#include <vector>
#include <hpc_helpers.hpp>

#include <mpi.h>
#define MAX_THREADS 40

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
        v[k-i-1] = M[(m + i + 1) * N + (m + k)]; // tutti gli elementi precedenti sulla stessa colonna
    }
    // std::reverse(v.begin(), v.end());
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

void wavefront(std::vector<double> &M, int N, int size) {

    std::vector<double> horizontal, vertical;
    for (int i = 1; i < N; ++i) {
        for(int j=0; j<N-i; j++) {
            horizontal=get_vector_m_k(M, N, j, i);
            vertical=get_vector_mk_k(M, N, j, i);

            double result;

            //Tag 99 is for the diagonal number
            MPI_Send(&i, 1, MPI_INT, j % size, 99, MPI_COMM_WORLD);
            // //Tag 100 is for horizontal vector
            MPI_Send(&horizontal[0], horizontal.size(), MPI_DOUBLE, (j % size), 100, MPI_COMM_WORLD);
            // //Tag 101 is for vertical vector
            MPI_Send(&vertical[0], vertical.size(), MPI_DOUBLE, (j % size), 101, MPI_COMM_WORLD);
            // MPI_Recv(&result, 10, MPI_DOUBLE, (j % size), 102, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            M[j * N + (j+i)]=result;
        };
    }
}

double dotProduct(vector<double> h, vector<double> v, int i){
    double acc=0;
    for (int z = 0; z < i; ++z) {
        acc += h[z] * v[z];
    }
    return cbrt(acc);
}

int main(int argc, char *argv[]) {
    int myrank;
	int size;
    MPI_Status status;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Datatype row_type;
	MPI_Datatype col_type;

	MPI_Type_contiguous(size, MPI_DOUBLE, &row_type);
	// count, blocklen, stride, oldtype, newtype 
	MPI_Type_vector(size, 1, size, MPI_DOUBLE, &col_type);
	MPI_Type_commit(&row_type);
	MPI_Type_commit(&col_type);

	uint64_t N = 512;    // default size of the matrix (NxN)
    int numThreads = MAX_THREADS;
	
    if(myrank==0){
        if (argc != 1 && argc != 2) {
		std::printf("use: %s N numThreads\n", argv[0]);
		std::printf("     N size of the square matrix\n");
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
        print_matrix(M, N);

        TIMERSTART(wavefront);
        wavefront(M, N, size); 
        TIMERSTOP(wavefront);

        // print_matrix(M, N);
        std::cout << M[N-1];
    } else {
        int k;
        MPI_Recv(&k, 1, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
        
        std::vector<double> h(k);
        std::vector<double> v(k);
        MPI_Recv(&h[0], k, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &status);
        MPI_Recv(&v[0], k, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD, &status);
        printf("My rank is %d and I received a k equal to %d\n", myrank, k);

        double result=dotProduct(h,v,k);
        MPI_Send(&result, 10, MPI_DOUBLE, 0, 103, MPI_COMM_WORLD);
        // printf("My rank is %d and the first element of h vector is %lf\n", myrank, h[0]);
        // printf("My rank is %d and the first element of v vector is %lf\n", myrank, v[0]);
    }
    // cout<<"My rank is "<<myrank<<endl;

    MPI_Finalize();
}