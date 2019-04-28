#include "conduits/single.h"
#include "solvers/base.h"
#include <mpi.h>
#include <queue>

extern Korali::Solver::Base*  _solver;
extern Korali::Problem::Base* _problem;

#define MSG_EXIT 1
#define MSG_READY 2
#define MSG_RESULT 3

size_t _nSamples;
size_t _nParameters;
double* _sampleArrayPointer;

Korali::Conduit::Single::Single(Korali::Solver::Base* solver) : Base::Base(solver) {};

int sample_in{0}; // keeping track of processed samples to monitor generations
int my_rank, size;
MPI_Comm master_comm;
std::queue<int> consumer_stby; // queue of available ranks for efficient dispatch

void Dispatch(const size_t sampleId, const int destination)
{
	double send_buffer[1 + _nParameters]; // to send sample parameters with sample ID
	std::copy(&_sampleArrayPointer[sampleId*_nParameters],
			  &_sampleArrayPointer[(sampleId+1)*_nParameters],
			  &send_buffer[1]);
	send_buffer[0]= (double) sampleId;
	MPI_Send(send_buffer, 1+_nParameters, MPI_DOUBLE, destination, MSG_RESULT, master_comm);
}


void Shutdown(MPI_Comm master_comm)
{
	MPI_Status status;
	double send_buffer[1 + _nParameters];
	double recv_buffer[2];
	for (int rank= 1; rank < size; ++rank)
	{
		MPI_Send(send_buffer, 1+_nParameters, MPI_DOUBLE, rank, MSG_EXIT, master_comm);
		MPI_Recv(recv_buffer, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MSG_EXIT, master_comm, &status);
	}
}


void Korali::Conduit::Single::processSample(size_t sampleId)
{
	MPI_Status status;
	double recv_buffer[2]; // to receive result with sample ID

	if (sampleId == 0) // at the beginning of a generation..
		sample_in= 0; // ..reset counter

	if (consumer_stby.size() == 0) // during ramp-up with no consumers checked in
	{
		MPI_Recv(recv_buffer, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, master_comm, &status);
		// act depending on MSG received:
		switch (status.MPI_TAG)
		{
			case MSG_READY: // consumer reports ready w/o result (during ramp-up)
				Dispatch(sampleId, status.MPI_SOURCE); // task consumer
				break;
			case MSG_RESULT: // consumer reports ready w/ result
				_solver->updateEvaluation((int) recv_buffer[0], recv_buffer[1]);
				++sample_in; // increase received sample counter +1
				Dispatch(sampleId, status.MPI_SOURCE);  // task consumer again 
				break;
			default: // catch-all for error handling
				printf("Error: Producer received unknown MSG from consumer.\n");
				MPI_Abort(master_comm, 1);
		}
	}

	else // empty queue of consumers ready to go
	{
		Dispatch(sampleId, consumer_stby.front());
		consumer_stby.pop();
	}

	if (sampleId == _nSamples - 1) // at end of generation..
	{
		while (sample_in < _nSamples) // ..collect remaining results
		{
			MPI_Recv(recv_buffer, 2, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, master_comm, &status);
			switch (status.MPI_TAG)
			{			
				case MSG_READY: // consumer reports ready w/o result
					consumer_stby.emplace(status.MPI_SOURCE); // place consumer in stby queue
					break;
				case MSG_RESULT: // consumer reports ready w/ result
					_solver->updateEvaluation((int) recv_buffer[0], recv_buffer[1]);
					++sample_in; // increase received sample counter +1
					consumer_stby.emplace(status.MPI_SOURCE); // place consumer in stby queue
					break;	
				default: // catch-all for error handling
					printf("Error: Producer received unknown MSG from consumer.\n");
					MPI_Abort(master_comm, 1);
			}
		}	
	}
}


void Consumer(MPI_Comm master_comm)
{	
	MPI_Status status;
	double recv_buffer[1 + _nParameters];
	double send_buffer[2];

	MPI_Send(send_buffer, 2, MPI_DOUBLE, 0, MSG_READY, master_comm); // report ready with bogus send_buffer
	MPI_Recv(recv_buffer, 1+_nParameters, MPI_DOUBLE, 0, MPI_ANY_TAG, master_comm, &status); // receive instructions
	while (status.MPI_TAG != MSG_EXIT) // while there's work to do
	{
		if (status.MPI_TAG == MSG_RESULT) // ..or work on a sample result
		{
			send_buffer[0]= recv_buffer[0]; // sample ID
			send_buffer[1]= _problem->evaluateSample(&recv_buffer[1]); // evaluate
			MPI_Send(send_buffer, 2, MPI_DOUBLE, 0, MSG_RESULT, master_comm); // report result and include calculation time
		}
		else
		{
			printf("Error: Consumer received unknown MSG from producer.\n");
			MPI_Abort(master_comm, 1);
		}
		MPI_Recv(recv_buffer, 1+_nParameters, MPI_DOUBLE, 0, MPI_ANY_TAG, master_comm, &status);	// receive next instructions
	}
	MPI_Send(send_buffer, 2, MPI_DOUBLE, 0, MSG_EXIT, master_comm); // sign off when nothing to do
}


void Korali::Conduit::Single::run()
{
	MPI_Init(nullptr, nullptr);

	_nSamples= _solver->_sampleCount;
	_nParameters= _solver->N;
	
	master_comm= MPI_COMM_WORLD;

	MPI_Comm_rank(master_comm, &my_rank);
	MPI_Comm_size(master_comm, &size);
	
	if (!my_rank)
	{
		_sampleArrayPointer = (double*) calloc (_nSamples*_nParameters, sizeof(double));
		_solver->runSolver();
		Shutdown(master_comm);
		free(_sampleArrayPointer);		
	}
	else
		Consumer(master_comm);
	
	MPI_Finalize();
}
