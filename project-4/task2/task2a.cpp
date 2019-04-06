#include <stdio.h>
#include <chrono>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

int main(int argc, char* argv[])
{
	upcxx::init();
	int rankId    = upcxx::rank_me();
	int rankCount = upcxx::rank_n();

	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

	checkResults(0 /* This will FAIL */ ); // Make sure you check results!

  upcxx::finalize();
	return 0;
}
