#include "model/grass.hpp"
#include "korali.h"

// Grass Height at different spots, as measured by Herr Kueheli.
size_t  nSpots;
double* xPos;
double* yPos;
double* heights;





void likelihood_plant_data(double* x, double* fx)
{
    double pH= x[0];
    double rain_mm= x[1];

    for (int i= 0; i < nSpots; ++i)
        fx[i]= getGrassHeight(xPos[i], yPos[i], pH, rain_mm);
}




int main(int argc, char* argv[])
{
	// Loading grass height data

	FILE* dataFile = fopen("grass.in", "r");

	fscanf(dataFile, "%lu", &nSpots);
	xPos     = (double*) calloc (sizeof(double), nSpots);
	yPos     = (double*) calloc (sizeof(double), nSpots);
	heights  = (double*) calloc (sizeof(double), nSpots);

	for (int i = 0; i < nSpots; i++)
	{
		fscanf(dataFile, "%le ", &xPos[i]);
		fscanf(dataFile, "%le ", &yPos[i]);
		fscanf(dataFile, "%le ", &heights[i]);
	}

    auto problem= Korali::Problem::Likelihood(likelihood_plant_data);

    Korali::Parameter::Uniform pH("pH", 4.0, 9.0);
    Korali::Parameter::Gaussian rain_mm("rain_mm", 90.0, 20.0);

    rain_mm.setBounds(0.0, 190.0);

    problem.addParameter(&pH);
    problem.addParameter(&rain_mm);
    
    problem.setReferenceData(nSpots, heights);

    auto maximizer= Korali::Solver::CMAES(&problem);

    maximizer.setStopMinDeltaX(1e-11);
    maximizer.setPopulationSize(256);

    maximizer.run();

	return 0;
}
