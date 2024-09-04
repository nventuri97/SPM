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

    MPI_Init(&argc, &argv);

    int myRank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t N = 512;  // default size of the matrix (NxN)

    if (myRank == 0) {
        if (argc != 1 && argc != 2 && argc != 3) {
            std::printf("use: %s N\n", argv[0]);
            std::printf("     N size of the square matrix\n");
            return -1;
        }
    }

    if (argc > 1) {
        N = std::stol(argv[1]);
    }

    // Allocation of memory space for matrix M of size N*N for all the processes
    double *M = new double[N * N];
    init_matrix(M, N);

    // Measure the current time
    double start = MPI_Wtime();

    // Distribute work across processes from k = 1 to N-1 (diagonals)
    for (uint64_t k = 1; k < N; ++k) {
        // The computation is divided by rows
        int numberOfRows = (N-k)/size;
        int numberOfElements = numberOfRows;
        //The + k is required because to compute an element
        int myRows = numberOfRows+k;

        // For the cases that 'rows' is not multiple of size
        if(myRank < (int) (N-k)%size){
            myRows++;
            numberOfElements++;
        }

        // Array to collect calculated values
        double* values_to_send = new double[numberOfElements];
        // Array to store indices corresponding to the calculated values
        int* indices = new int[numberOfElements];                            

        int shift = myRank * numberOfRows;
        if((N-k)%size!=0 && myRank!=0){
            shift+=myRank < (int) (N-k) % size ? myRank : (N-k) % size;
        }

        int counter = 0;
        for (int i = shift; i < shift + numberOfRows + (myRank < (int) (N - k) % size ? 1 : 0); ++i) {
            double dotProduct = 0.0;

            for (uint64_t j = 1; j < k + 1; ++j) {
                dotProduct += M[i * N + (i + k - j)] * M[(i + j) * N + (i + k)];
            }
            M[i * N + (i + k)] = cbrt(dotProduct);

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

    double end = MPI_Wtime();

    if (myRank == 0) {
        std::cout << "# elapsed time (wavefront): " << end - start << "s" << std::endl;
        // Uncomment to print the final matrix
        // print_matrix(M, N);
        printf("%f\n", M[N - 1]);
    }

    delete[] M;

    MPI_Finalize();
    return 0;
}
