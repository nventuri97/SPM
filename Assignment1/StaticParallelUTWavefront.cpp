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

//Function that compute the wave front in a static parallel way, a thread compute the corresponding element % number of threads
void staticParallelWavefront(const std::vector<int> &M, const uint64_t &N, const int &numThreads){
	//Barrier with which threads wait until the end of a row
	std::barrier greatBarrier(numThreads);
	//Threads vector which will contain the threads that compute the elements
	std::vector<std::thread> threads;
	threads.reserve(numThreads);

	auto staticParallelization = [&] (uint64_t threadId) -> void {
		//Integer that specifies the starting point for each threads based on its thread ID
		uint64_t startPoint=threadId;

		for(uint64_t k = 0; k< N; ++k) { 			       								// for each upper diagonal
			for(uint64_t i = startPoint; i<(N-k); i+=numThreads) {						// for each elem. in the diagonal
				work(std::chrono::microseconds(M[i*N+(i+k)])); 
			}

			if(threadId+1 > N-k){
				greatBarrier.arrive_and_drop();
				break;
			}
			else{
				greatBarrier.arrive_and_wait();
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
	int numThreads= 4;
	
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
