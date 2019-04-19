#include <cstdio>
#include <chrono>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

int main()
{
	upcxx::init();

	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

	if (!upcxx::rank_me())
		printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);

	// for the time: guard for uneven-sized shards
	if (nSamples % upcxx::rank_n())
	{
		if (!upcxx::rank_me()) // only first rank prints
			printf("Error: NSAMPLES must be a multiple of team size.\n");
		exit(1);
	}

	const int n_local{nSamples / upcxx::rank_n()}; // works as we just made sure it does

	upcxx::dist_object<upcxx::global_ptr<double>> results_g(upcxx::new_array<double>(nSamples));
	double* samples= (double*) calloc (nSamples * nParameters, sizeof(double));

	if (!upcxx::rank_me()) // only first (master) rank initializes (and checks at the end)
		samples= initializeSampler(nSamples, nParameters); // initialize samples
	
	auto start = std::chrono::system_clock::now(); // start timer

	upcxx::broadcast(samples, nSamples * nParameters, 0).wait(); // not overly elegant: transfers unrequired data

	upcxx::global_ptr<double> master_results_gptr= results_g.fetch(0).wait(); // get the first rank's location

	const int start_index{upcxx::rank_me() * n_local}; // starting location for each rank

	upcxx::future<> fut_all= upcxx::make_future(); // to conjoin during the loop
	for (int i= start_index; i < start_index + n_local; ++i) // run over rank's part of samples
	{
		double result_i= evaluateSample(&samples[i * nParameters]); // evaluate sample
        upcxx::future<> fut= upcxx::rput(result_i, master_results_gptr + i); // place into master's results_g
		fut_all= upcxx::when_all(fut_all, fut); // conjoin futures
	}	
    fut_all.wait(); // wait for all conjoined futures to complete
	
	auto end= std::chrono::system_clock::now(); // end timer
	auto diff= std::chrono::duration<double>(end - start).count(); // rank time
	auto total_time= upcxx::reduce_all(diff, upcxx::op_fast_add).wait(); // sum up all rank times 
	auto max_time= upcxx::reduce_all(diff, upcxx::op_fast_max).wait(); // find slowest rank time
	// printf("I'm %i of %i, I took %.3fs\n", upcxx::rank_me(), upcxx::rank_n(), diff);

	if (!upcxx::rank_me()) // only first rank checks results
	{
		checkResults(results_g->local()); // check results
		auto avg_time= total_time / upcxx::rank_n(); // calculate average rank time
		printf("Total time:\t%.3fs\nAverage time:\t%.3fs\nMaximum time:\t%.3fs", total_time, avg_time, max_time);
		printf("\nMaximum time/avg time = %.3f\n", max_time / avg_time);
	}

	upcxx::finalize();
	return 0;
}
