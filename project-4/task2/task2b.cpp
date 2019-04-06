#include <stdio.h>
#include <chrono>
#include <queue>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"

#define NSAMPLES 240
#define NPARAMETERS 2

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;

int main(int argc, char* argv[])
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();
	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

  if (rankId == 0) printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);

  auto t0 = std::chrono::system_clock::now();

  // Solution should be placed here for good timing

	auto t1 = std::chrono::system_clock::now();

	if (rankId == 0)
	{
	 checkResults(0 /* This will FAIL */ ); // Make sure you check results!
	 double evalTime = std::chrono::duration<double>(t1-t0).count();
	 printf("Total Running Time: %.3fs\n", evalTime);
	}

  upcxx::finalize();
	return 0;
}
