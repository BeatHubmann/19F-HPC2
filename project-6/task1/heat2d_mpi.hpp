#ifndef _HEAT2D_H_
#define _HEAT2D_H_

#include <stdlib.h>
#include <vector>
#include <mpi.h>

typedef struct pointsInfoStruct {
  std::vector<double> xPos;
  std::vector<double> yPos;
  std::vector<double> inten;
  std::vector<double> width;
  std::vector<double> refTemp;
  size_t nCandles;
} pointsInfo;

typedef struct gridLevelStruct {
 int N; // Number of points per dimension on grid level
 int NN; // Number of points on grid level 
 int n_x; // Number of points in x dimension on local grid level
 int n_x_limit; // Number of points used locally to cover original grid
 int n_y; // Number of points in y dimension on local grid level
 int n_y_limit; // Number of points used locally to cover original grid
 int nn; // Number of points on local grid level
 MPI_Datatype N_bound;
 MPI_Datatype E_bound;
 MPI_Datatype S_bound;
 MPI_Datatype W_bound;
 double h; // DeltaX = DeltaY, the distance between points in the discretized [0,1]x[0,1] domain
 double x_min; // local grid boundary positions, check if really needed
 double x_max;
 double y_min;
 double y_max;

 double* f; // Right hand side (external heat sources)
 double* U; // Main grid
 double* Un; // Previous Jacobi grid
 double* Res; // Residual Grid
//  double** f; // Right hand side (external heat sources)
//  double** U; // Main grid
//  double** Un; // Previous Jacobi grid
//  double** Res; // Residual Grid

 double L2NormLocal; // saving local L2 norm for debugging 
 double L2Norm; // L2 Norm of the residual
 double L2NormPrev; // Previous L2 Norm
 double L2NormDiff; // L2Norm Difference compared to previous step
} gridLevel;

// Helper Functions
gridLevel* generateInitialConditions(size_t N0, size_t gridCount);
void freeGrids(gridLevel* g, size_t gridCount);
void print(gridLevel* g, size_t l, double* matrix);

// Solver functions
void applyJacobi(gridLevel* g, size_t l, size_t relaxations);
void calculateResidual(gridLevel* g, size_t l);
void applyRestriction(gridLevel* g, size_t l);
void applyProlongation(gridLevel* g, size_t l);
void calculateL2Norm(gridLevel* g, size_t l);
void printTimings(size_t gridCount);

double* smoothingTime;
double* residualTime;
double* restrictionTime;
double* prolongTime;
double* L2NormTime;
double totalTime;


#endif // _HEAT2D_H_
