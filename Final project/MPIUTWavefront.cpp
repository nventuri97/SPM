#include <iostream>
#include <cmath>
#include <cstring> 
#include <vector>
#include <hpc_helpers.hpp>
#include <mpi.h>
#include <omp.h>

using namespace std;


void init_matrix(double *M, int N){
    for (int i=0;i<N;i++){
        for(int j=0; j<N; j++)
            if(j==i)
                M[i * N + j] = (i + 1) / static_cast<double>(N);
            else
                M[i * N + j] = 0.0;
    }
}

void print_matrix(double *M, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%lf ", M[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    // int provided;
    // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Init(&argc, &argv);

    int myRank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	uint64_t N = 512;       // default size of the matrix (NxN)
    int numThreads = 0;     //default number of thread if not specified
    
    if(myRank==0){
        if (argc != 1 && argc != 2 && argc != 3) {
		std::printf("use: %s N numThreads\n", argv[0]);
		std::printf("     N size of the square matrix\n");
		return -1;
        }
    }

	if (argc > 1) {
		N = std::stol(argv[1]);
		if (argc > 2) {
			numThreads = std::stol(argv[2]);
		}
	}
 
    double *M = nullptr;
    
    if(!myRank){
        //Allocation of memory space for matrix M of size N*N for all the process
        M=new double[N*N];
        init_matrix(M, N);
    }

	// Measure the current time
	double start = MPI_Wtime();

    // Distribute work across processes from k = 1 to N-1 (diagonals)
    for (int k = 1; k < N; ++k) {                             
        //If the number of elements to compute is less than the number of processes I decrease the number of processes
        if(N-k<size){
            size--;
        }

        if(myRank<size){
            // The computation is divided by rows
            int numberOfRows = (N-k)/size;
            int myRows = numberOfRows+k;

            // For the cases that 'rows' is not multiple of size
            if(myRank < (N-k)%size){
                myRows++;
            }

            // Arrays for the chunk of data to work
            double *myData = new  double[myRows*N];

            // The process 0 must specify how many rows are sent to each process   
            int *sendCounts = nullptr;
            int *displs = nullptr;

            if(!myRank){
                sendCounts = new int[size];
                displs = new int[size];

                displs[0] = 0;

                for(int i=0; i<size; i++){
                    if(i>0){
                        displs[i] = displs[i-1]+sendCounts[i-1]- k * N;
                    }

                    if(i < (N-k)%size){
                        sendCounts[i] = (numberOfRows+k+1)*N;
                    } else {
                        sendCounts[i] = (numberOfRows+k)*N;
                    }
                }
            }

            // Scatter the input matrix
            MPI_Scatterv(M, sendCounts, displs, MPI_DOUBLE, myData, myRows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            int shift = myRank * numberOfRows;
            if((N-k)%size!=0 && myRank!=0){
                shift+=myRank < (N-k) % size ? myRank : (N-k) % size;
            }

            //Each process computes its part of the diagonal
            for (int i = 0; i < myRows - k; ++i) {
                double result = 0.0;

                // #pragma omp parallel for num_threads(numThreads) reduction(+:result)
                for (int j = 1; j < k+1; ++j) {
                    result += myData[shift + i * N + (i + k - j)] * myData[shift + (i + j) * N + (i + k)];
                }
                myData[shift + i * N + (i + k)]=cbrt(result);  
            }
        
            MPI_Gatherv(myData, myRows*N, MPI_DOUBLE, M, sendCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);   

            delete[] myData;
            if(myRank==0){
                delete[] sendCounts;
                delete[] displs;
            }
            
        }
    }
    /*******************END OF NEW PART*********************************/
    double end = MPI_Wtime();

    if(!myRank){
        std::cout << "Time with " << size << " processes: " << end-start << " seconds" << std::endl;
        // printf("The final matrix is\n");
        // print_matrix(M,N);
        printf("%f\n", M[N-1]);
        delete[] M;
    }

    MPI_Finalize();
}