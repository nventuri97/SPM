#include <iostream>
#include <cmath>
#include <cstring>
#include <mpi.h>

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

void print_matrix(double *M, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%lf ", M[i * N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init(&argc, &argv);

    int myRank, size;
    MPI_Group world;
    MPI_Comm commChannel;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // MPI_Comm_group(MPI_COMM_WORLD, &world);
    // MPI_Comm_create(MPI_COMM_WORLD, world, &commChannel);

    uint64_t N = 512;  // default size of the matrix (NxN)
    int numThreads = 0;  // default number of threads if not specified

    if (myRank == 0) {
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

    // Allocation of memory space for matrix M of size N*N for all the processes
    double *M = new double[N * N];
    init_matrix(M, N);

    // Measure the current time
    double start = MPI_Wtime();
    int threads=0;

    // Distribute work across processes from k = 1 to N-1 (diagonals)
    for (int k = 1; k < N; ++k) {
        // if(N-k<size){
        //     size--;

        //     int exludedRank[1]={size};
        //     MPI_Group_excl(world, 1, exludedRank, &world);
            
        //     MPI_Comm_free(&commChannel);
        //     MPI_Comm_create(MPI_COMM_WORLD, world, &commChannel);
        // }

        if (myRank < size) {
            // The computation is divided by rows
            int numberOfRows = (N-k)/size;
            int numberOfElements = numberOfRows;
            //The + k is required because to compute an element
            int myRows = numberOfRows+k;

            // For the cases that 'rows' is not multiple of size
            if(myRank < (N-k)%size){
                myRows++;
                numberOfElements++;
            }

            double* values_to_send = new double[numberOfElements];               // Array to collect calculated values
            int* indices = new int[numberOfElements];                            // Array to store indices corresponding to the calculated values

            // for (uint64_t i = myRank; i < N - k; ++i) {
            //     double dotProduct = 0.0;

            //     for (uint64_t j = 1; j < k + 1; ++j) {
            //         dotProduct += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
            //     }
            //     M[i * N + (i + k)] = cbrt(dotProduct);

            //     // Store the calculated value and its index
            //     values_to_send[i]=M[i * N + (i + k)];
            //     indices[i]=i;
            // }


            int shift = myRank * numberOfRows;
            if((N-k)%size!=0 && myRank!=0){
                shift+=myRank < (N-k) % size ? myRank : (N-k) % size;
            }

            // //Each process computes its part of the diagonal
            // for (int i = shift; i < shift + myRows - k; ++i) {
            //     double dotProduct = 0.0;

            //     // if(numThreads>k)
            //     //     threads=k;
            //     // else
            //     //     threads=numThreads;
            //     printf("%d my shift is %d\n", myRank, shift);
            //     // #pragma omp parallel for num_threads(numThreads) reduction(+:dotProduct)
            //     for (int j = 1; j < k+1; ++j) {
            //         dotProduct += M[(i) * N + (i + k - j)] * M[((i + j)) * N + (i + k)];
            //     }
            //     M[(i) * N + (i + k)]=cbrt(dotProduct);  
            //     if(myRank==1)
            //         printf("index is %d\n", (i) * N + (i + k));
            //     // Store the calculated value and its index
            //     printf("I inserted the value %f in position %d\n",  M[(i) * N + (i + k)], i);
            //     values_to_send[i]=M[(i) * N + (i + k)];
            //     indices[i]=i;
            // }

            int counter = 0;
            for (int i = shift; i < shift + numberOfRows + (myRank < (N - k) % size ? 1 : 0); ++i) {
                double dotProduct = 0.0;

                for (int j = 1; j < k + 1; ++j) {
                    dotProduct += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
                }
                M[i * N + (i + k)] = cbrt(dotProduct);

                // if(myRank==1)
                //     printf("index is %d\n", (i) * N + (i + k));
                // Store the calculated value and its index
                // printf("I inserted the value %f in position %d\n",  M[(i) * N + (i + k)], i);

                values_to_send[counter] = M[i * N + (i + k)];
                indices[counter] = i;
                counter++;
            }

            // Gather all the calculated values from all processes
            int* recv_counts= new int[size];
            MPI_Allgather(&numberOfElements, 1, MPI_INT, recv_counts, 1, MPI_INT, MPI_COMM_WORLD);

            int* displs= new int[size];
            int total_elements = 0;
            for (int i = 0; i < size; ++i) {
                displs[i] = total_elements;
                total_elements += recv_counts[i];
            }

            double* gathered_values=new double[total_elements];
            int* gathered_indices=new int[total_elements];

            MPI_Allgatherv(values_to_send, numberOfElements, MPI_DOUBLE,
                        gathered_values, recv_counts, displs, MPI_DOUBLE,
                        MPI_COMM_WORLD);

            MPI_Allgatherv(indices, numberOfElements, MPI_INT,
                        gathered_indices, recv_counts, displs, MPI_INT,
                        MPI_COMM_WORLD);

            // Update the matrix with the gathered values
            for (int i = 0; i < total_elements; ++i) {
                M[gathered_indices[i]*N + gathered_indices[i] + k] = gathered_values[i];
            }
        }
    }

    double end = MPI_Wtime();

    if (myRank == 0) {
        std::cout << "# elapsed time (wavefront): " << end - start << "s" << std::endl;
        // Uncomment to print the final matrix
        // print_matrix(M, N);
        printf("%f\n", M[N - 1]);
    }

    delete[] M;

    // Ensure to only free the communicator if it's valid
    // MPI_Comm_free(&commChannel);
    // MPI_Group_free(&world);

    MPI_Finalize();
    return 0;
}
