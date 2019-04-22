#include "conduits/upcxx.h"
#include "solvers/base.h"
#include <queue>


extern Korali::Solver::Base*  _solver;
extern Korali::Problem::Base* _problem;

size_t nSamples;
size_t nParameters;

upcxx::global_ptr<double> sampleArrayPointer;

Korali::Conduit::UPCXX::UPCXX(Korali::Solver::Base* solver) : Base::Base(solver) {};


struct Consumer // to dress producer ranks in when building queue
{
	upcxx::intrank_t my_rank; // the actual rank of the consumer
	upcxx::future<double> my_result; // the future which will contain the result once ready
	size_t sample_no; // the sample number the consumer is Dispatched with to evaluate
};


std::queue<Consumer> consumers; // queue of consumers to keep track of dispatched samples
std::queue<upcxx::intrank_t> ranks_ready; // queue of available ranks for efficient dispatch
upcxx::intrank_t rank_to_commission{1}; // keeping track of already commissioned ranks
int sample_count{0}; // keeping track of processed samples to monitor generations


void Dispatch(const upcxx::intrank_t rank, const size_t sample_idx) //task ranks with samples to evaluate
{   // RPC with lambda function for evaluation:
	auto fut_result= upcxx::rpc(rank,
                                [](const size_t sampleId) -> double {
						            return _problem->evaluateSample(&sampleArrayPointer.local()[nParameters*sampleId]);
				                }, sample_idx);
	consumers.emplace(Consumer{rank, fut_result, sample_idx}); // place dispatched rank into consumer queue
	upcxx::progress(); // allow for upcxx::progress() for good measure
}


void ProcessConsumer() // processes result once a consumer reports ready
{
	Consumer consumer= consumers.front(); // grab ready consumer ..
	consumers.pop(); // .. and remove it from queue
	upcxx::intrank_t return_rank= consumer.my_rank; // extract consumer rank
	double return_fitness= consumer.my_result.result(); // extract consumer result
	size_t return_sample= consumer.sample_no; // extract sampleId result belongs to
	_solver->updateEvaluation(return_sample, return_fitness); // update solver with fitness
	ranks_ready.emplace(return_rank); // add now available again rank to queue of ready ranks
	upcxx::progress(); // allow for upcxx::progress() for good measure
	// printf("I've received %ld from %i w/ fitness %.3f\n", return_sample, return_rank, return_fitness);	
}


void CycleQueue() // cycle the queue by one position while advancing upcxx::progress()
{
	Consumer consumer= consumers.front(); // grab (presumably) busy consumer from front..
	consumers.pop(); // .. and remove it ..
	consumers.push(consumer); // .. to add it again at the back
	upcxx::progress(); // allow draining & processing all consumers' RPC inbox
}


upcxx::intrank_t AssignRank() // calculate next best rank to be dispatched
{
	upcxx::progress(); // allow upcxx::progress for good measure
	upcxx::intrank_t next_rank; // this will be the return value
	if (rank_to_commission < upcxx::rank_n()) // during ramp-up (typically Gen 1, 2):
	{
		next_rank= rank_to_commission++; // assign next previously uncommissioned rank
	}
	else // after ramp-up with all ranks in the game:
	{
		if (ranks_ready.size() != 0) // there's ranks ready with nothing to do:
		{
			next_rank= ranks_ready.front(); // take rank from front of queue..
			ranks_ready.pop(); // ..and remove it from queue
		}
		else // all ranks are busy with previous dispatches
		{
			next_rank= rank_to_commission++ % upcxx::rank_n(); // queue dispatch in round-robin sequence
		}
		
	}
	return next_rank;	
	// return rank_to_commission++ % upcxx::rank_n(); // naive round-robin: ~+30% runtime 
}


void Korali::Conduit::UPCXX::processSample(size_t sampleId)
{
	upcxx::intrank_t next_rank= AssignRank(); // get rank to be tasked
	// printf("I'm %i, sending %ld to %i\n", upcxx::rank_me(), sampleId, next_rank);	 
	Dispatch(next_rank, sampleId); // send sampleId to next best rank to evaluate
	++sample_count; // keep track of processed samples for end-of-generation processing

	if (consumers.front().my_result.ready()) // if there is a result ready right now ..
		ProcessConsumer(); // .. process ready result

	if (sample_count % nSamples == 0) // at end-of-generation: we have to clean out queue:
		while (consumers.size() != 0) // while there are samples being processed ..
		{
			while (!consumers.front().my_result.ready()) // .. until a consumer is ready ..
				CycleQueue(); // .. iterate the queue, then ..
			ProcessConsumer(); // .. process ready result
		}
}


void Korali::Conduit::UPCXX::run()
{
	upcxx::init(); // yep
	nSamples= _solver->_sampleCount; // population size
	nParameters= _solver->N; // parameter count

	if (!upcxx::rank_me()) 	// master rank 0 creates sample array in global shared memory:
		sampleArrayPointer= upcxx::new_array<double>(nSamples*nParameters);
	
	upcxx::broadcast(&sampleArrayPointer, 1, 0).wait(); // broadcast sample array master->all

	if (!upcxx::rank_me()) // only master rank runs solver and thus nSample times processSample()
		_solver->runSolver(); // do the deed

	upcxx::barrier(); // make sure everyone's done
	upcxx::finalize(); // close up shop
}
