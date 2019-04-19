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

	// as exercise specified 240 samples on 24 ranks: guard for uneven-sized shards just in case
	if (nSamples % upcxx::rank_n())
	{
		if (!upcxx::rank_me()) // only first rank prints
			printf("Error: NSAMPLES must be a multiple of number of ranks (exercise specs 240/24).\n");
		exit(1);
	}
	const int n_local{nSamples / upcxx::rank_n()}; // int division as we just made sure

	double* samples= (double*) calloc (nSamples * nParameters, sizeof(double)); // local samples array

	upcxx::dist_object<upcxx::global_ptr<double>> samples_g(upcxx::new_array<double>(nSamples*nParameters)); // global samples
	upcxx::dist_object<upcxx::global_ptr<double>> results_g(upcxx::new_array<double>(nSamples)); // global results

	upcxx::global_ptr<double> master_samples_gptr= samples_g.fetch(0).wait(); // get the first rank's location
	upcxx::global_ptr<double> master_results_gptr= results_g.fetch(0).wait(); // get the first rank's location

	auto start = std::chrono::system_clock::now(); // start timer (will also count initializeSampler, but not much difference)

    // would be more convenient to do broadcast, but that doesn't scale nicely:	
	if (!upcxx::rank_me()) // only first (master) rank initializes (and checks at the end)
	{
		samples= initializeSampler(nSamples, nParameters); // initialize samples
		auto f= upcxx::rput(samples, master_samples_gptr, nSamples*nParameters); // put into global address space
		f.wait(); // wait for completion of rput
	}

	upcxx::barrier(); // make sure sample data has been made global as wait() does not suffice

	const int start_index{upcxx::rank_me() * n_local}; // starting location for each rank

	auto fut= upcxx::rget(master_samples_gptr + start_index * nParameters, // only rget required sample data 
						  samples + start_index * nParameters,
						  n_local * nParameters);
	fut.wait(); // wait for completion of rget

	upcxx::future<> fut_all= upcxx::make_future(); // to conjoin during the loop
	for (int i= start_index; i < start_index + n_local; ++i) // run over rank's part of samples
	{
		double result_i= evaluateSample(&samples[i * nParameters]); // evaluate sample
        fut= upcxx::rput(result_i, master_results_gptr + i); // place into master's results_g
		fut_all= upcxx::when_all(fut_all, fut); // conjoin futures
	}	
    fut_all.wait(); // wait for all conjoined futures to complete
	
	auto end= std::chrono::system_clock::now(); // end timer
	auto diff= std::chrono::duration<double>(end - start).count(); // rank time
	auto total_time= upcxx::reduce_all(diff, upcxx::op_fast_add).wait(); // sum up all rank times 
	auto max_time= upcxx::reduce_all(diff, upcxx::op_fast_max).wait(); // find slowest rank time

	if (!upcxx::rank_me()) // only first rank checks results
	{
		checkResults(results_g->local()); // check results
		auto avg_time= total_time / upcxx::rank_n(); // calculate average rank time
		printf("Total time:\t%.3fs\nAverage time:\t%.3fs\nMaximum time:\t%.3fs", total_time, avg_time, max_time);
		printf("\nMaximum time/avg time = %.3f\n", max_time / avg_time);
		printf("Load imbalance ratio = %.3f\n", (max_time - avg_time) / max_time);
	}

	upcxx::finalize();
	return 0;
}
