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

	auto problem= Korali::Problem::Posterior(heat2DSolver);
	// 4-Candle Model x4 parameters/candle 
	p.nCandles= 4;
        
	Korali::Parameter::Uniform xpos_1("xpos_1", 0.0, 0.5);
	Korali::Parameter::Uniform ypos_1("ypos_1", 0.0, 1.0);

	Korali::Parameter::Uniform xpos_2("xpos_2", 0.0, 0.5);
	Korali::Parameter::Uniform ypos_2("ypos_2", 0.0, 1.0);
	
	Korali::Parameter::Uniform xpos_3("xpos_3", 0.5, 1.0);
	Korali::Parameter::Uniform ypos_3("ypos_3", 0.0, 1.0);

	Korali::Parameter::Uniform xpos_4("xpos_4", 0.5, 1.0);
	Korali::Parameter::Uniform ypos_4("ypos_4", 0.0, 1.0);

	Korali::Parameter::Uniform beamWidth_1("beamWidth_1", 0.04, 0.06);
	Korali::Parameter::Uniform beamWidth_2("beamWidth_2", 0.04, 0.06);
	Korali::Parameter::Uniform beamWidth_3("beamWidth_3", 0.04, 0.06);
	Korali::Parameter::Uniform beamWidth_4("beamWidth_4", 0.04, 0.06);

	Korali::Parameter::Uniform beamIntensity_1("beamIntensity_1", 0.4, 0.6);
	Korali::Parameter::Uniform beamIntensity_2("beamIntensity_2", 0.4, 0.6);
	Korali::Parameter::Uniform beamIntensity_3("beamIntensity_3", 0.4, 0.6);
	Korali::Parameter::Uniform beamIntensity_4("beamIntensity_4", 0.4, 0.6);
  
	problem.addParameter(&xpos_1);
    problem.addParameter(&ypos_1);
	problem.addParameter(&beamIntensity_1);
	problem.addParameter(&beamWidth_1);

	problem.addParameter(&xpos_2);
	problem.addParameter(&ypos_2);
	problem.addParameter(&beamIntensity_2);
	problem.addParameter(&beamWidth_2);
	
	problem.addParameter(&xpos_3);
    problem.addParameter(&ypos_3);
	problem.addParameter(&beamIntensity_3);
	problem.addParameter(&beamWidth_3);
 
    problem.addParameter(&ypos_4);
    problem.addParameter(&ypos_4);
	problem.addParameter(&beamIntensity_4);
	problem.addParameter(&beamWidth_4);
	
	problem.setReferenceData(p.nPoints, p.refTemp);

	auto solver= Korali::Solver::CMAES(&problem);

    const int maxGens= 2000; // max generations for CMAES
	const int popSize= 23; // instead of ~4+3*log(N) to make use of parallelism
	solver.setStopMinDeltaX(1e-6);
	solver.setMu(4);
	solver.setPopulationSize(popSize);
	solver.setMaxGenerations(maxGens);
	solver.run();

	return 0;
}
