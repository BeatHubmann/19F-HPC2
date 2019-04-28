#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <mpi.h>
#include "sampler/sampler.hpp"

#define NSAMPLES 240
#define NPARAMETERS 2

#define MSG_EXIT 1
#define MSG_READY 2
#define MSG_RESULT 3

size_t nSamples;
size_t nParameters;


void dispatch(size_t& now_at_sample, MPI_Status& status, MPI_Comm master_comm)
{
	double send_buffer[1 + nParameters]; // to send sample parameters with sample ID
	if (now_at_sample < nSamples)
	{
		getSample(now_at_sample, &send_buffer[1]);
		send_buffer[0]= (double) now_at_sample++;
		MPI_Send(send_buffer, 1+nParameters, MPI_DOUBLE, status.MPI_SOURCE, MSG_READY, master_comm);
	}
	else
		MPI_Send(send_buffer, 1+nParameters, MPI_DOUBLE, status.MPI_SOURCE, MSG_EXIT, master_comm);	
}


void producer(MPI_Comm master_comm)
{
	MPI_Status status;
	int size; 
	MPI_Comm_size(master_comm, &size); // size of master_comm
	int active_consumers{size - 1}; // size - me == amount of consumers 
	size_t now_at_sample{0}; // running index to keep track of dispatches
	std::vector<double>	time_counters(size); // to keep track of consumers' calculation time
	double recv_buffer[3]; // to receive result with sample ID and calculation time

	printf("Processing %ld Samples (24 initially available), each with %ld Parameter(s)...\n", nSamples, nParameters);

	initializeSampler(nSamples, nParameters); // initialize sampler

	while (active_consumers > 0) // while there are active consumers (duh)
	{	// receive report from consumer:
		MPI_Recv(recv_buffer, 3, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, master_comm, &status);
		// act depending on MSG received:
		switch (status.MPI_TAG)
		{
			case MSG_EXIT: // consumer confirms sign off (during wind-down)
				--active_consumers;
				break;			
			case MSG_READY: // consumer reports ready w/o result (during ramp-up)
				dispatch(now_at_sample, status, master_comm); // task consumer or send exit 
				break;
			case MSG_RESULT: // consumer reports ready w/ result
				updateEvaluation((int) recv_buffer[0], recv_buffer[1]);
				time_counters[status.MPI_SOURCE]+= recv_buffer[2]; // record calculation time from consumer
				dispatch(now_at_sample, status, master_comm);  // task consumer again or send exit 
				break;
			default: // catch-all for error handling
				printf("Error: Producer received unknown MSG from consumer.\n");
				MPI_Abort(master_comm, 1);
		}
	}
	
	const auto total_time= std::accumulate(time_counters.begin(), time_counters.end(), 0.0);
	const auto max_time= *std::max_element(time_counters.begin(), time_counters.end());

	checkResults(); // well
	const auto avg_time= total_time / (size-1); // calculate average rank time
	printf("Total time:\t%.3fs\nAverage time:\t%.3fs\nMaximum time:\t%.3fs", total_time, avg_time, max_time);
	printf("\nMaximum time/avg time = %.3f\n", max_time / avg_time);
	printf("Load imbalance ratio = %.3f\n", (max_time - avg_time) / max_time);
}


void consumer(MPI_Comm master_comm)
{	
	MPI_Status status;
	double recv_buffer[1 + nParameters];
	double send_buffer[3];
	std::chrono::system_clock::time_point start, end;

	MPI_Send(send_buffer, 3, MPI_DOUBLE, 0, MSG_READY, master_comm); // report ready with bogus send_buffer
	MPI_Recv(recv_buffer, 1+nParameters, MPI_DOUBLE, 0, MPI_ANY_TAG, master_comm, &status); // receive instructions
	while (status.MPI_TAG == MSG_READY) // while there's work to do
	{
		send_buffer[0]= recv_buffer[0]; // sample ID
		start= std::chrono::system_clock::now(); // time check
		send_buffer[1]= evaluateSample(&recv_buffer[1]); // evaluate
		end= std::chrono::system_clock::now(); // time check
		send_buffer[2]= std::chrono::duration<double>(end-start).count(); // calculation time
		MPI_Send(send_buffer, 3, MPI_DOUBLE, 0, MSG_RESULT, master_comm); // report result and include calculation time
		MPI_Recv(recv_buffer, 1+nParameters, MPI_DOUBLE, 0, MPI_ANY_TAG, master_comm, &status);	// receive next instructions
	}
	MPI_Send(send_buffer, 3, MPI_DOUBLE, 0, MSG_EXIT, master_comm); // sign off when nothing to do
}


int main(int argc, char* argv[])
{
	MPI::Init();

	nSamples = NSAMPLES; // fixed parameters
	nParameters = NPARAMETERS;

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	if (!my_rank)
		producer(MPI_COMM_WORLD);
	else
		consumer(MPI_COMM_WORLD);
	
	MPI_Finalize();
	return 0;
}