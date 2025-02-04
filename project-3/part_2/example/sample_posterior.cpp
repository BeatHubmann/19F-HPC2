#include "korali.h"

// Computational Model
double ackley(const double a, const double b, const double c, const double x1, const double x2)
{
  double s1 = x1*x1 + x2*x2;
  double s2 = cos(c*x1) + cos(c*x2);

  return -a*exp(-b*sqrt(s1/2)) - exp(s2/2) + a + exp(1.);
}

// We store the value of the provided points in data.in as global variables, for convenience.
size_t  nPoints;
double* x1Pos;
double* x2Pos;
double* fxValues;
void likelihood_ackley(double* x, double* fx)
{
	// The goal of this code is to find the values of a, b, c that best reflect
	// the provided data points, so these are our parameters.
	double a = x[0];
	double b = x[1];
	double c = x[2];

	// Filling fx for Korali to calculate likelihood
	// No need to use the negative because this is not about maximizing or minimizing
	// the model, but the likelihood. Korali will take care of that.
	for (int i = 0; i < nPoints; i++)
	 fx[i] = ackley(a, b, c, x1Pos[i], x2Pos[i]);
}

int main(int argc, char* argv[])
{
	// Loading reference data: position and values of the ackley function
	// at several points, given a, b, c parameters that we don't currently know

	FILE* dataFile = fopen("data.in", "r");

	fscanf(dataFile, "%lu", &nPoints);
	x1Pos     = (double*) calloc (sizeof(double), nPoints);
	x2Pos     = (double*) calloc (sizeof(double), nPoints);
	fxValues  = (double*) calloc (sizeof(double), nPoints);

	for (int i = 0; i < nPoints; i++)
	{
		fscanf(dataFile, "%le ", &x1Pos[i]);
		fscanf(dataFile, "%le ", &x2Pos[i]);
		fscanf(dataFile, "%le ", &fxValues[i]);
	}

  // We want to maximize the posterior distribution of the parameters.
	// We do have some prior information now.
  auto problem = Korali::Problem::Posterior(likelihood_ackley);

  // Suppose we have some prior information about the distribution of the a and c
  // parameters. A is a Gamma distribution with rate 17.0 and shape 0.5.
  // C is a gaussian distribution with mean 1.0, sigma 0.3;
  Korali::Parameter::Gamma a("a", 17.0, 0.5);
  a.setBounds(0.0, +50.0); // Since gamma goes from -inf to +inf, we need to define its bounds.

  Korali::Parameter::Uniform b("b", 0.0, +1.0); // We still dont know anything about b.

  Korali::Parameter::Gaussian c("c", 1.0, 0.3);
  c.setBounds(0.0, +2.0*M_PI); // Since gaussian goes from -inf to +inf, we need to define its bounds.

  problem.addParameter(&a);
  problem.addParameter(&b);
  problem.addParameter(&c);

  // Very important: dont forget to give the reference data to Korali!
  problem.setReferenceData(nPoints, fxValues);

  // Use TMCMC to sample -ackley(x)
  auto sampler = Korali::Solver::TMCMC(&problem);

  // TMCMC-specific configuration.
  // Number of samples to represent the distribution.
  // The more samples, the more precise the representation will be
  // but may take more time to run per generation, and more generations
  // to find a perfectly annealing representation.
  sampler.setPopulationSize(10000);

  // Defines the 'sensitivity' of re-sampling. That is, how much the new
  // samples within a chain will scale during evaluation. A higher value
  // is better to explore a larger space, while a lower value will be
  // more precise for small magnitude parameters.
  sampler.setCovarianceScaling(0.2);

	// Run TMCMC to produce tmcmc.txt.
  // Use plotmatrix_hist to see the result of the sampling.
  sampler.run();

  return 0;
}
