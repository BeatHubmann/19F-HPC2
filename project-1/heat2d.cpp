#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <cstring>
#include "auxiliar/auxiliar.hpp"

class GridLevel
{
public:
 size_t N; // Number of points per dimension in the grid level
 double h; // DeltaX = DeltaY, the distance between points in the discretized [0,1]x[0,1] domain
 double** f; // Right hand side (external heat sources)
 double** U; // Main grid for Jacobi
 double** Un; // Previous' step grid
 double** Res; // Residual Grid
};

void heat2DSolver(Heat2DSetup& s)
{
 // Multigrid parameters -- Find the best configuration! Now assigned by command line
//  s.setGridCount(s.gridCount);     // Number of Multigrid levels to use; needed to initialize values
 s.setGridCount(5);     // Number of Multigrid levels to use; needed to initialize values
 s.downRelaxations = 7; // Number of Relaxations before restriction
 s.upRelaxations   = 1;   // Number of Relaxations after prolongation

 // Allocating Grids -- Is there a better way to allocate these grids?
 GridLevel* g = (GridLevel*) calloc(sizeof(GridLevel), s.gridCount);
 for (int i = 0; i < s.gridCount; i++)
 {
  g[i].N = pow(2, s.N0-i) + 1;
  g[i].h = 1.0 / (g[i].N - 1);

  g[i].U   = (double**) calloc (sizeof(double*), g[i].N);
  for (int j = 0; j < g[i].N ; j++)
    g[i].U[j]   = (double*) calloc (sizeof(double), g[i].N);
  
  g[i].Un  = (double**) calloc (sizeof(double*), g[i].N);
  for (int j = 0; j < g[i].N ; j++)
    g[i].Un[j]  = (double*) calloc (sizeof(double), g[i].N);

  g[i].Res = (double**) calloc (sizeof(double*), g[i].N);
  for (int j = 0; j < g[i].N ; j++)
    g[i].Res[j] = (double*) calloc (sizeof(double), g[i].N);

  g[i].f   = (double**) calloc (sizeof(double*), g[i].N);
  for (int j = 0; j < g[i].N ; j++)
    g[i].f[j]   = (double*) calloc (sizeof(double), g[i].N);
 }

 // Setting up problem.
 #pragma vector aligned
 for (int i = 0; i < s.N; i++)
  #pragma ivdep
  for (int j = 0; j < s.N; j++)
  {
    g[0].U[i][j] = s.getInitial(i,j);
    g[0].f[i][j] = s.getRHS(i,j);
  }

 while (s.L2NormDiff > s.tolerance)  // Multigrid solver start
 {
  s.applyJacobi_(g, 0, s.downRelaxations); // Relaxing the finest grid first
  s.calculateResidual_(g, 0); // Calculating Initial Residual

  for (int grid = 1; grid < s.gridCount; grid++) // Going down the V-Cycle
  {
   s.applyRestriction_(g, grid); // Restricting the residual to the coarser grid's solution vector (f)
   s.applyJacobi_(g, grid, s.downRelaxations); // Smoothing coarser level
   s.calculateResidual_(g, grid); // Calculating Coarse Grid Residual
  }

  for (int grid = s.gridCount-1; grid > 0; grid--) // Going up the V-Cycle
  {
   s.applyProlongation_(g, grid); // Prolonging solution for coarser level up to finer level
   s.applyJacobi_(g, grid, s.upRelaxations); // Smoothing finer level
  }

  s.calculateL2Norm_(g, 0); // Calculating Residual L2 Norm
 }  // Multigrid solver end

 // Saving solution before returning
 for (int i = 0; i < g[0].N; i++) for (int j = 0; j < g[0].N; j++) s.saveSolution(i, j, g[0].U[i][j]);
}

void applyJacobi(GridLevel* g, int l, int relaxations)
{
 for (int r = 0; r < relaxations; r++)
 {
  std::swap(g[l].U, g[l].Un);

  const double gridhsquared{g[l].h * g[l].h};

  #pragma vector aligned
   for (int i = 1; i < g[l].N-1; i++)
   #pragma ivdep
    for (int j = 1; j < g[l].N-1; j++) // Perform a Jacobi Iteration
      g[l].U[i][j] = (g[l].Un[i-1][j] + g[l].Un[i+1][j] + g[l].Un[i][j-1] + g[l].Un[i][j+1] + g[l].f[i][j] * gridhsquared) * 0.25;
 }
}

void calculateResidual(GridLevel* g, int l)
{
  const double invgridhsquared{1 / (g[l].h * g[l].h)};
  #pragma vector aligned
  for (int i = 1; i < g[l].N-1; i++)
    #pragma ivdep
    for (int j = 1; j < g[l].N-1; j++)
      g[l].Res[i][j] = g[l].f[i][j] + (g[l].U[i-1][j] + g[l].U[i+1][j] - 4*g[l].U[i][j] + g[l].U[i][j-1] + g[l].U[i][j+1]) * invgridhsquared;
}

double calculateL2Norm(GridLevel* g, int l)
{
  double tmp = 0.0;
  #pragma vector aligned
  for (int i = 0; i < g[l].N; i++)
    #pragma ivdep
    for (int j = 0; j < g[l].N; j++)
      tmp += g[l].Res[i][j] * g[l].Res[i][j];

  return sqrt(tmp);
}

void applyRestriction(GridLevel* g, int l)
{
  #pragma vector aligned
  for (int i = 1; i < g[l].N-1; i++)
    #pragma ivdep
    for (int j = 1; j < g[l].N-1; j++)
      g[l].f[i][j] = ( 1.0*( g[l-1].Res[2*i-1][2*j-1] + g[l-1].Res[2*i-1][2*j+1] + g[l-1].Res[2*i+1][2*j-1]   + g[l-1].Res[2*i+1][2*j+1] ) +
                       2.0*( g[l-1].Res[2*i-1][2*j]   + g[l-1].Res[2*i][2*j-1]   + g[l-1].Res[2*i+1][2*j]     + g[l-1].Res[2*i][2*j+1]   ) +
                       4.0*( g[l-1].Res[2*i][2*j] ) ) * 0.0625;

#pragma vector aligned
for (int i = 0; i < g[l].N; i++)
  #pragma ivdep
  for (int j = 0; j < g[l].N; j++)
    g[l].U[i][j] = 0;
}

void applyProlongation(GridLevel* g, int l)
{
  // FIRST
  #pragma vector aligned
  for (int i = 1; i < g[l].N-1; i++)
    #pragma ivdep
    for (int j = 1; j < g[l].N-1; j++)
      g[l-1].Un[2*i][2*j] = g[l].U[i][j];

  // SECOND
  #pragma vector aligned
  for (int i = 1; i < g[l].N; i++)
    #pragma ivdep
    for (int j = 1; j < g[l].N-1; j++)
      g[l-1].Un[2*i-1][2*j] = ( g[l].U[i-1][j] + g[l].U[i][j] ) * 0.5;

  // THIRD
  #pragma vector aligned
  for (int i = 1; i < g[l].N-1; i++)
    #pragma ivdep
    for (int j = 1; j < g[l].N; j++)
      g[l-1].Un[2*i][2*j-1] = ( g[l].U[i][j-1] + g[l].U[i][j] ) * 0.5;

  // FOURTH
  #pragma vector aligned
  for (int i = 1; i < g[l].N; i++)
    #pragma ivdep
    for (int j = 1; j < g[l].N; j++)
      g[l-1].Un[2*i-1][2*j-1] = ( g[l].U[i-1][j-1] + g[l].U[i-1][j] + g[l].U[i][j-1] + g[l].U[i][j] ) * 0.25;

  #pragma vector aligned
  for (int i = 0; i < g[l-1].N; i++)
    #pragma ivdep
    for (int j = 0; j < g[l-1].N; j++)
      g[l-1].U[i][j] += g[l-1].Un[i][j];
}
