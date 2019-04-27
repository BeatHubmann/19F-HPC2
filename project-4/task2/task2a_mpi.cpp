#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <mpi.h>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

int main()
{
	MPI::Init();

	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

	int size, my_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (!my_rank)
		printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples, nParameters);

	// as exercise specified 240 samples on 24 ranks: guard for uneven-sized shards just in case
	if (nSamples % size)
	{
		if (!my_rank) // only first rank prints
			printf("Error: NSAMPLES must be a multiple of number of ranks (exercise specs 240/24).\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	const int n_local{nSamples / size}; // int division as we just made sure

	double* samples;
	double* results;

	if (!my_rank)
	{
		samples= initializeSampler(nSamples, nParameters); // master has all samples..
		results= (double*) calloc(nSamples, sizeof(double)); // .. and results
	}
	else
	{
		samples= (double*) calloc(n_local*nParameters, sizeof(double)); // slaves will do local share
		results= (double*) calloc(n_local, sizeof(double)); // slaves will do local share
	}

	auto start = std::chrono::system_clock::now(); // start timer (will also count initializeSampler, but not much difference)
	// distribute data:
	if (!my_rank)
		MPI_Scatter(samples, n_local*nParameters, MPI_DOUBLE, 
		            MPI_IN_PLACE, n_local*nParameters, MPI_DOUBLE,
					0, MPI_COMM_WORLD);
	else
		MPI_Scatter(samples, n_local*nParameters, MPI_DOUBLE,
		            samples, n_local*nParameters, MPI_DOUBLE,
					0, MPI_COMM_WORLD);
	// calculate data:
	for (int i= 0; i < n_local; ++i) // run over rank's part of samples
		results[i]= evaluateSample(&samples[i * nParameters]); // evaluate sample
	// gather data:
	if (!my_rank)
		MPI_Gather(MPI_IN_PLACE, n_local, MPI_DOUBLE, // master gathers result data from..
		           results, n_local, MPI_DOUBLE,
				   0, MPI_COMM_WORLD);
	else
		MPI_Gather(results, n_local, MPI_DOUBLE, // master gathers result data from..
		           results, n_local, MPI_DOUBLE,
				   0, MPI_COMM_WORLD);

	auto end= std::chrono::system_clock::now(); // end timer
	auto diff= std::chrono::duration<double>(end - start).count(); // rank time
	double total_time, max_time;
	MPI_Reduce(&diff, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // sum up all rank times 
	MPI_Reduce(&diff, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); // find slowest rank time

	if (!my_rank) // only first rank checks results
	{
		checkResults(results); // check results
		auto avg_time= total_time / size; // calculate average rank time
		printf("Total time:\t%.3fs\nAverage time:\t%.3fs\nMaximum time:\t%.3fs", total_time, avg_time, max_time);
		printf("\nMaximum time/avg time = %.3f\n", max_time / avg_time);
		printf("Load imbalance ratio = %.3f\n", (max_time - avg_time) / max_time);
	}

	MPI::Finalize();
	return 0;
}
