//
// Sequential code of the first SPM Assignment a.a. 23/24.
//
// compile:
// g++ -std=c++20 -O3 -march=native -I ../include/ StaticParallelUTWavefront.cpp -o StaticParallelUTW
//
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <cassert>
#include <hpc_helpers.hpp>

#include <barrier>

int random(const int &min, const int &max) {
	static std::mt19937 generator(117);
	std::uniform_int_distribution<int> distribution(min,max);
	return distribution(generator);
};		

// emulate some work
void work(std::chrono::microseconds w) {
	auto end = std::chrono::steady_clock::now() + w;
    while(std::chrono::steady_clock::now() < end);	
}

void sequentialWavefront(const std::vector<int> &M, const uint64_t &N) {
	for(uint64_t k = 0; k< N; ++k) {        // for each upper diagonal
		for(uint64_t i = 0; i< (N-k); ++i) {// for each elem. in the diagonal
			work(std::chrono::microseconds(M[i*N+(i+k)])); 
		}
	}
}

//Function that compute the wave front in a static parallel way based on a chunk of data
void staticParallelWavefront(const std::vector<int> &M, const uint64_t &N, const uint64_t &numThreads){
	//Barrier with which threads wait until the end of a raw
	std::barrier greatBarrier(numThreads);
	//Number of elements that a single thread has to skip to work on the next item
	const uint64_t chunkSize = N/numThreads;
	//Threads vector which will contain the threads that compute the elements
	std::vector<std::thread> threads(numThreads);

	auto staticParallelization = [&] (uint64_t threadId) -> void {
		for(uint64_t k = 0; k< N; k+=chunkSize) {        								// for each upper diagonal
			for(uint64_t i = 0; i<(N-k); ++i) {											// for each elem. in the diagonal
				work(std::chrono::microseconds(M[i*N+(i+k)])); 
			}
		}
	};

	for (uint64_t i = 0; i < numThreads; i++){
        threads.emplace_back(staticParallelization, i);
    }

    for (auto& thread : threads)
        thread.join();
}

void dynamicParallelWavefront(const std::vector<int> &M, const uint64_t &N, const uint64_t &numThreads){
	
}

int main(int argc, char *argv[]) {
	int min    = 0;      // default minimum time (in microseconds)
	int max    = 1000;   // default maximum time (in microseconds)
	uint64_t N = 512;    // default size of the matrix (NxN)
	uint64_t numThreads= 4;
	
	if (argc != 1 && argc != 2 && argc != 5) {
		std::printf("use: %s N [min max]\n", argv[0]);
		std::printf("     N size of the square matrix\n");
		std::printf("     min waiting time (us)\n");
		std::printf("     max waiting time (us)\n");		
		std::printf("	  numTheads number of threads to be used (optional)\n");
		return -1;
	}

	if (argc > 1) {
		N = std::stol(argv[1]);
		if (argc > 2) {
			min = std::stol(argv[2]);
			max = std::stol(argv[3]);
		}

		if (argc > 4){
			numThreads = std::stol(argv[4]);
		}
	}

	// allocate the matrix
	std::vector<int> M(N*N, -1);

	uint64_t expected_totaltime=0;
	// init function
	auto init=[&]() {
		for(uint64_t k = 0; k< N; ++k) {  
			for(uint64_t i = 0; i< (N-k); ++i) {  
				int t = random(min,max);
				M[i*N+(i+k)] = t;
				expected_totaltime +=t;				
			}
		}
	};
	
	init();

	std::printf("Estimated compute time ~ %f (ms)\n", expected_totaltime/1000.0);
	
	TIMERSTART(wavefront);
	staticParallelWavefront(M, N, numThreads);
    TIMERSTOP(wavefront);

    return 0;
}
