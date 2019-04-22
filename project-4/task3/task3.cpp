#include <stdio.h>
#include <chrono>
#include <queue>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"

#define NSAMPLES 240
#define NPARAMETERS 2

size_t nSamples;
size_t nParameters;

struct Consumer // to dress producer ranks in when building queue
{
	upcxx::intrank_t my_rank; // the actual rank of the consumer
	upcxx::future<double> my_result; // the future which will contain the result once ready
	size_t sample_no; // the sample number the consumer is dispatched with to evaluate
	std::chrono::system_clock::time_point start; // dispatch time stamp
	double calc_time; // running total of calculation time
};

// dispatch function to task ranks with samples to evaluate returning future results:
upcxx::future<double> dispatch(const upcxx::intrank_t rank, const size_t sample_idx)
{   // RPC with lambda function for evaluation
	double sample_params[nParameters]; // to save sequentially obtained parameters
	getSample(sample_idx, sample_params); // obtain next set of parameters
	return upcxx::rpc(rank,
                      [](upcxx::view<double> sample_params) -> double {
						  return evaluateSample(const_cast<double*>(&sample_params.front()));
				      }, upcxx::make_view(sample_params, sample_params+nParameters));
}


int main(int argc, char* argv[])
{
	upcxx::init(); // they see me rollin'.. again

	nSamples = NSAMPLES; // fixed parameters
	nParameters = NPARAMETERS;

	if (!upcxx::rank_me()) // first rank is master==producer, tasks slaves==consumers
	{   // master rank prints status message:
		printf("Processing %ld Samples (24 initially available), each with %ld Parameter(s)...\n", nSamples, nParameters);

		initializeSampler(nSamples, nParameters); // initialize sampler
		
		double total_time{0.0}; // to sum up consumers' calculation time for stats
		double max_time{0.0}; // to keep track of highest consumer calculation time

		size_t now_at_sample{0}; // running index to keep track of dispatches
		std::queue<Consumer> consumers; // main tool: Consumer queue

		// dress up all ranks except master as consumers and dispatch with initial sample:
		upcxx::intrank_t max_rank= std::min((int)upcxx::rank_n(), (int)nSamples + 1); // safeguard num slave ranks > nSamples
		for (upcxx::intrank_t rank= 1; rank < max_rank; ++rank)
		{
			auto fut_result= dispatch(rank, now_at_sample); // dispatch sample for evaluation
			consumers.emplace(Consumer{rank, fut_result, now_at_sample++,
			                           std::chrono::system_clock::now(), 0.0}); // dress rank as consumer and enqueue
		}

		// cycle queue checking for ready consumers: take result dispatch new while samples last:
		while (!consumers.empty()) // as long as there are dispatched consumers in the queue
		{
			while (!consumers.front().my_result.ready()) // iterate until a consumer is ready
			{
				Consumer consumer= consumers.front();
				consumers.pop();
				consumers.push(consumer);
				upcxx::progress(); // allow draining & processing consumers' RPC inbox
			}
			// now consumer with ready result at front of queue:
			auto split_time= std::chrono::system_clock::now(); // take time to update consumer's stats
			Consumer consumer= consumers.front(); // handle for consumer
			consumers.pop(); // remove ready consumer from queue
			updateEvaluation(consumer.sample_no, consumer.my_result.result()); // update sampling engine with result
			consumer.calc_time+= std::chrono::duration<double>(split_time-consumer.start).count(); // update time stats

			if (now_at_sample < nSamples) // if there are still samples to evaluate
			{	// dispatch consumer with next sample and enqueue again		
				consumer.my_result= dispatch(consumer.my_rank, now_at_sample);
				consumer.sample_no= now_at_sample++;
				consumer.start= std::chrono::system_clock::now();
				consumers.push(consumer);
			}
			else
			{
				total_time += consumer.calc_time; // sum up total calculation time
				max_time= std::max(max_time, consumer.calc_time); // find maximum calculation time for stats
			} 
		}
		checkResults(); // well
		auto avg_time= total_time / upcxx::rank_n(); // calculate average rank time
		printf("Total time:\t%.3fs\nAverage time:\t%.3fs\nMaximum time:\t%.3fs", total_time, avg_time, max_time);
		printf("\nMaximum time/avg time = %.3f\n", max_time / avg_time);
		printf("Load imbalance ratio = %.3f\n", (max_time - avg_time) / max_time);
	}
	upcxx::finalize(); // done
	return 0;
}
