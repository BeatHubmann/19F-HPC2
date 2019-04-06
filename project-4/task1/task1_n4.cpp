#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char* argv[])
{

	// Loading temperature measurement data
	FILE* dataFile = fopen("data_n4.in", "r");
	fscanf(dataFile, "%lu", &p.nPoints);

	p.xPos    = (double*) calloc (sizeof(double), p.nPoints);
	p.yPos    = (double*) calloc (sizeof(double), p.nPoints);
	p.refTemp = (double*) calloc (sizeof(double), p.nPoints);

	for (int i = 0; i < p.nPoints; i++)
	{
		fscanf(dataFile, "%le ", &p.xPos[i]);
		fscanf(dataFile, "%le ", &p.yPos[i]);
		fscanf(dataFile, "%le ", &p.refTemp[i]);
	}

	auto problem = Korali::Problem::Posterior(heat2DSolver);
	// 4-Candle Model x4 parameters/candle 
	p.nCandles = 4;
        
	/*
	TODO : fill in all parameter information required 
	*/


  	auto solver = Korali::Solver::CMAES(&problem);

    int Ng = 2000; // max generations for CMAES
	
    solver.setStopMinDeltaX(1e-6);
	solver.setPopulationSize(8); // ~4+3*log(N)
	solver.setMu(4);
	solver.setMaxGenerations(Ng);
	solver.run();

	return 0;
}
