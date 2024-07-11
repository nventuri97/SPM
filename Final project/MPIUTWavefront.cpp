#include <iostream>
#include <cmath>
#include <cstring> 
#include <vector>
#include <hpc_helpers.hpp>
#include <mpi.h>

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
            std::cout << M[i * N + j] << " ";
        }
        std::cout << std::endl;
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
    int myRank=0;
	int size=0;
	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	uint64_t N = 512;    // default size of the matrix (NxN)
    int numThreads = 0; //default number of thread if not specified
    
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

    //Allocation of memory space for matrix M of size N*N
    double *M = new double[N*N];
    
    if(!myRank){
        init_matrix(M, N);
        MPI_Bcast(M, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        print_matrix(M, N);
    }

    /*******************NEW PART TO MODIFY******************************/
	// Measure the current time
	double start = MPI_Wtime();

    // Distribute work across processes
    for (int k = 1; k < N; ++k) {                               // k = 1 to N-1 (diagonals)
    
        // The computation is divided by rows
        int overlap = 1;                                        // number of overlapping rows
        int numberOfRows = (N-k)/size;
        int myRows = numberOfRows+overlap;                      //this plus overlap is necessary because to compute the dot product a process needs at least of two row

        // For the cases that 'rows' is not multiple of size
        if(myRank < (N-k)%size){
            myRows++;
        }

        // Arrays for the chunk of data to work
        double *myData = new double[myRows*N];

        // The process 0 must specify how many rows are sent to each process
        int *sendCounts = nullptr;
        int *displs = nullptr;
        
        if(!myRank){
            sendCounts = new int[size];
            displs = new int[size];

            displs[0] = 0;

            for(int i=0; i<size; i++){
                if(i>0){
                    displs[i] = displs[i-1]+sendCounts[i-1]- overlap * N;
                }

                if(i < (N-k)%size){
                    sendCounts[i] = (numberOfRows+overlap+1)*N;
                } else {
                    sendCounts[i] = (numberOfRows+overlap)*N;
                }
            }
        }

        // Scatter the input matrix
        MPI_Scatterv(M, sendCounts, displs, MPI_DOUBLE, myData, myRows*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //Each process computes its part of the diagonal
        for (int i = 0; i < myRows; ++i) {
            double result = 0.0;

            // #pragma omp parallel for reduction(+:result)
            for (int j = 0; j < k+1; ++j) {
                // if(myRank==2){
                //     printf("%f*%f\n",myData[row-j], myData[row+N+1+j]);
                // }
                result += myData[myRank + i * N + (i + k - j)] * myData[myRank + (i + j) * N + (i + k)];
            }
            myData[myRank+i+k]=cbrt(result);  
        }
     
        MPI_Gatherv(myData, myRows*N, MPI_DOUBLE, M, sendCounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);   

        delete[] myData;
        if(myRank==0){
            delete[] sendCounts;
            delete[] displs;
        }
    }
    /*******************END OF NEW PART*********************************/
    if(!myRank){
        printf("The final matrix is\n");
        print_matrix(M,N);
    }  

    MPI_Finalize();
}