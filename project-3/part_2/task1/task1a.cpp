#include "model/grass.hpp"
#include "korali.h"

#define MM 80.0
#define PH 6.0


double max_grass_height(double* x)
{
    return getGrassHeight(x[0], x[1], PH, MM);
}

int main(int argc, char* argv[])
{
    auto problem= Korali::Problem::Direct(max_grass_height);

    Korali::Parameter::Uniform x_coord("X", 0.0, 5.0);
    Korali::Parameter::Uniform y_coord("Y", 0.0, 5.0);
    problem.addParameter(&x_coord);
    problem.addParameter(&y_coord);
    auto maximizer= Korali::Solver::CMAES(&problem);
    maximizer.setStopMinDeltaX(1e-11);
    maximizer.setPopulationSize(64);
    maximizer.run();

	return 0;
}
