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
        // print_matrix(M, N);
    }

	// Measure the current time
	double start = MPI_Wtime();

    // Distribute work across processes
    for (int k = 1; k < N; ++k) {                               // k = 1 to N-1 (diagonals)
        if(N-k<size){
            size--;
        }

        if(myRank<size){
            // The computation is divided by rows
            int overlap = 1;                                    // number of overlapping rows
            int numberOfRows = (N-k)/size;
            printf("The number of ROWS per process are %d and rank is %d\n", numberOfRows, myRank);
            int myRows = numberOfRows+k;                        //this plus overlap is necessary because to compute the dot product a process needs at least of two row

            // For the cases that 'rows' is not multiple of size
            if(myRank < (N-k)%size){
                myRows++;
            }
            printf("Rank=%d, k=%d and myRows are %d\n", myRank, k, myRows);

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

                    printf("The sendCounts of %d is %d\n", i, sendCounts[i]);
                    printf("The displs of %d is %d\n", i, displs[i]);
                }
            }

            // Scatter the input matrix
            MPI_Scatterv(M, sendCounts, displs, MPI_DOUBLE, myData, myRows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            int shift = myRank * numberOfRows;
            if((N-k)%size!=0 && myRank!=0){
                shift+=(N-k)%size;
            }
            printf("MyRank is %d and my shift is %d\n", myRank, shift);

            //Each process computes its part of the diagonal
            for (int i = 0; i < myRows - k; ++i) {
                double result = 0.0;

                // #pragma omp parallel for num_threads(numThreads) reduction(+:result)
                for (int j = 1; j < k+1; ++j) {
                    // if(myRank==1){
                       printf("My rank is %d, calcolo l'elemento myData[%d]: %f*%f, from myData[%d], myData[%d]\n",myRank, shift + i * N + (i + k), myData[shift + i * N + (i + k - j)], myData[shift + (i + j) * N + (i + k)], shift + i * N + (i + k - j), shift + (i + j) * N + (i + k));
                    // }
                    result += myData[shift + i * N + (i + k - j)] * myData[shift + (i + j) * N + (i + k)];
                }
                // if(myRank==0)
                    printf("My rank is %d: %f inserted in myData[%d] \n",myRank, cbrt(result), shift+i * N + (i + k));
                myData[shift + i * N + (i + k)]=cbrt(result);  
            }
        
            MPI_Gatherv(myData, myRows*N, MPI_DOUBLE, M, sendCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);   

            delete[] myData;
            if(myRank==0){
                delete[] sendCounts;
                delete[] displs;
                printf("For k=%d, matrix is\n", k);
                print_matrix(M, N);
            }
            
        }
    }
    /*******************END OF NEW PART*********************************/
    double end = MPI_Wtime();

    if(!myRank){
        std::cout << "Time with " << size << " processes: " << end-start << " seconds" << std::endl;
        printf("The final matrix is\n");
        print_matrix(M,N);
        // printf("%f\n", M[N-1]);
        delete[] M;
    }

    MPI_Finalize();
}