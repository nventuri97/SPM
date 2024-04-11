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

void wavefront(const std::vector<int> &M, const uint64_t &N) {
	for(uint64_t k = 0; k< N; ++k) {        // for each upper diagonal
		for(uint64_t i = 0; i< (N-k); ++i) {// for each elem. in the diagonal
			work(std::chrono::microseconds(M[i*N+(i+k)])); 
		}
	}
}