#include "model/heat2d.hpp"
#include "korali.h"

void likelihood_heat2D(double* x, double* fx)
{
 heat2DSolver(x, fx);    
}

int main(int argc, char* argv[])
{
 // Loading temperature measurement data

 FILE* dataFile = fopen("data.in", "r");
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


 // How many candles will we simulate?
 // p.nCandles = 1; // 1-Candle Model - Requires 2 parameters (PosX, PosY)
 // p.nCandles = 2; // 2-Candle Model - Requires 4 parameters (PosX, PosY)x2
 // p.nCandles = 3; // 3-Candle Model - Requires 6 parameters (PosX, PosY)x3

 // Start configuring the Problem and the Korali Engine
 auto problem_n_2= Korali::Problem::Likelihood(likelihood_heat2D);
 auto problem_n_3= Korali::Problem::Likelihood(likelihood_heat2D);

 Korali::Parameter::Gaussian xpos_1("xpos_1", 0.25, 0.05);
 xpos_1.setBounds(0.0, 0.5);
 Korali::Parameter::Uniform ypos_1("ypos_1", 0.0, 1.0);

 Korali::Parameter::Gaussian xpos_2("xpos_2", 0.75, 0.05);
 xpos_2.setBounds(0.5, 1.0);
 Korali::Parameter::Uniform ypos_2("ypos_2", 0.0, 1.0);

 Korali::Parameter::Gaussian xpos_3("xpos_3", 0.75, 0.05);
 xpos_3.setBounds(0.5, 1.0);
 Korali::Parameter::Uniform ypos_3("ypos_3", 0.0, 1.0);

 problem_n_2.addParameter(&xpos_1);
 problem_n_2.addParameter(&ypos_1);
 problem_n_2.addParameter(&xpos_2);
 problem_n_2.addParameter(&ypos_2);

 problem_n_3.addParameter(&xpos_1);
 problem_n_3.addParameter(&ypos_1);
 problem_n_3.addParameter(&xpos_2);
 problem_n_3.addParameter(&ypos_2);
 problem_n_3.addParameter(&xpos_3);
 problem_n_3.addParameter(&ypos_3);

 problem_n_2.setReferenceData(p.nPoints, p.refTemp);
 problem_n_3.setReferenceData(p.nPoints, p.refTemp);

 auto maximizer_n_2= Korali::Solver::CMAES(&problem_n_2); 
 auto maximizer_n_3= Korali::Solver::CMAES(&problem_n_3); 

 auto sampler_n_2= Korali::Solver::TMCMC(&problem_n_2); 
 auto sampler_n_3= Korali::Solver::TMCMC(&problem_n_3); 

 maximizer_n_2.setStopMinDeltaX(1e-11);
 maximizer_n_3.setStopMinDeltaX(1e-11);

 sampler_n_2.setCovarianceScaling(0.2);
 sampler_n_3.setCovarianceScaling(0.2);

 maximizer_n_2.setPopulationSize(128);
 maximizer_n_3.setPopulationSize(128);

 maximizer_n_2.setMaxGenerations(1000);
 maximizer_n_3.setMaxGenerations(1000);

 sampler_n_2.setPopulationSize(10000);
 sampler_n_3.setPopulationSize(10000);

 p.nCandles= 2;
 maximizer_n_2.run();
 sampler_n_2.run();

 p.nCandles= 3;
 maximizer_n_3.run();
 sampler_n_3.run();

 return 0;
}
