#include "conduits/upcxx.h"
#include "solvers/base.h"
#include <queue>

extern Korali::Solver::Base*  _solver;
extern Korali::Problem::Base* _problem;

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;
upcxx::global_ptr<double> sampleArrayPointer;

Korali::Conduit::UPCXX::UPCXX(Korali::Solver::Base* solver) : Base::Base(solver) {};

void Korali::Conduit::UPCXX::processSample(size_t sampleId)
{
}

void Korali::Conduit::UPCXX::run()
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();
	nSamples = _solver->_sampleCount;
	nParameters = _solver->N;

	// Creating sample array in global shared memory
	if (rankId == 0) sampleArrayPointer  = upcxx::new_array<double>(nSamples*nParameters);
	upcxx::broadcast(&sampleArrayPointer,  1, 0).wait();

  upcxx::barrier();
  upcxx::finalize();
}


