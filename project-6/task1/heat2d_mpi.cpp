/**********************************************************************/
// A now optimized Multigrid Solver for the Heat Equation             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin, Georgios Arampatzis                        //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <limits>
#include "heat2d_mpi.hpp"
// #include "string.h"
#include <cstring>
#include <chrono>
#include <mpi.h>
#include <cassert>
#include <utility>


#define GRIDCOUNT (1)
#define NPOINTSPOWER (8)


pointsInfo __p;

struct WorldStruct
{
    int my_rank;
    int num_procs;

    int dims_x;
    int dims_y;

    int coord_x;
    int coord_y;

    int cart_rank;
    int N_proc;
    int E_proc;
    int S_proc;
    int W_proc;
} world;

// MPI_Datatype N_bound, E_bound, S_bound, W_bound;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world.my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.num_procs);

    int dims[2]= {0, 0};
    MPI_Dims_create(world.num_procs, 2, dims);
    world.dims_x= dims[0];
    world.dims_y= dims[1];

    MPI_Comm cart_comm;
    int periods[2]= {false, false};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &cart_comm);
    MPI_Comm_rank(cart_comm, &world.cart_rank);
    MPI_Cart_shift(cart_comm, 0, 1, &world.W_proc, &world.E_proc);
    MPI_Cart_shift(cart_comm, 1, 1, &world.S_proc, &world.N_proc);

    int coords[2]= {0, 0};
    MPI_Cart_coords(cart_comm, world.cart_rank, 2, coords);
    world.coord_x= coords[0];
    world.coord_y= coords[1];

    if (!world.my_rank)
    {
        printf("num_procs: %i\n", world.num_procs);
        printf("dims_x: \%i\n", world.dims_x);
        printf("dims_y: \%i\n", world.dims_y); 
    }

    double tolerance = 1e-0; // L2 Difference Tolerance before reaching convergence.
    size_t N0 = NPOINTSPOWER;// 2^N0 + 1 elements per side

    // Multigrid parameters
    size_t gridCount = GRIDCOUNT;// Number of Multigrid levels to use
    assert(gridCount <= 4);      // according to task specification 1d
    size_t downRelaxations = 3;  // Number of Relaxations before restriction
    size_t upRelaxations = 3;    // Number of Relaxations after prolongation

    gridLevel* g = generateInitialConditions(N0, gridCount);

    MPI_Barrier(MPI_COMM_WORLD);

    auto startTime = std::chrono::system_clock::now();
    while (g[0].L2NormDiff > tolerance) // Multigrid solver start
    {
        applyJacobi(g, 0, downRelaxations); // Relaxing the finest grid first
        calculateResidual(g, 0);            // Calculating Initial Residual

        for (size_t grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
        {
            applyRestriction(g, grid);             // Restricting the residual to the coarser grid's solution vector (f)
            applyJacobi(g, grid, downRelaxations); // Smoothing coarser level
            calculateResidual(g, grid);            // Calculating Coarse Grid Residual
        }

        for (size_t grid = gridCount - 1; grid > 0; grid--) // Going up the V-Cycle
        {
            applyProlongation(g, grid);          // Prolonging solution for coarser level up to finer level
            applyJacobi(g, grid, upRelaxations); // Smoothing finer level
        }

        calculateL2Norm(g, 0); // Calculating Residual L2 Norm
    }                          // Multigrid solver end

    auto endTime = std::chrono::system_clock::now();
    totalTime = std::chrono::duration<double>(endTime - startTime).count();
    printTimings(gridCount);
    printf("L2Norm: %.4f\n", g[0].L2Norm);
    freeGrids(g, gridCount);
    MPI_Finalize();
    return 0;
}

void applyJacobi(gridLevel *g, size_t l, size_t relaxations)
{
    auto t0 = std::chrono::system_clock::now();

    
    double h1 = 0.25;
    double h2 = g[l].h * g[l].h;
    for (size_t r = 0; r < relaxations; r++)
    {
        // double **tmp = g[l].Un;
        // g[l].Un = g[l].U;
        // g[l].U = tmp;
        std::swap(g[l].U, g[l].Un);
        const int f_x= g[l].n_x + 2;
        const int f_y= g[l].n_y + 2;
        for (int i= 1; i < py - 1; ++i)
            for (int j= 0; j < g[l].N-1; ++j) // Perform a Jacobi Iteration
                g[l].U[i * g[l].N + j]= (g[l].Un[(i - 1) * g[l].N + j    ]
                                       + g[l].Un[(i + 1) * g[l].N + j    ]
                                       + g[l].Un[ i      * g[l].N + j - 1]
                                       + g[l].Un[ i      * g[l].N + j + 1] 
                                       + g[l].f  [i      * g[l].N + j    ] * h2)
                                       * h1;
    }


    auto t1 = std::chrono::system_clock::now();
    smoothingTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void calculateResidual(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    double h2 = 1.0 / pow(g[l].h, 2);

    for (int i= 1; i < g[l].N - 1; ++i)
        for (int j= 1; j < g[l].N - 1; ++j)
            g[l].Res[g[l].N * i + j]= g[l].f[i][j] + (g[l].U[i - 1][j] + g[l].U[i + 1][j] - 4 * g[l].U[i][j] + g[l].U[i][j - 1] + g[l].U[i][j + 1]) * h2;

    auto t1 = std::chrono::system_clock::now();
    residualTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void calculateL2Norm(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    double tmp = 0.0;

    for (size_t i = 0; i < g[l].N; i++)
        for (size_t j = 0; j < g[l].N; j++)
            g[l].Res[i][j] = g[l].Res[i][j] * g[l].Res[i][j];

    for (size_t i = 0; i < g[l].N; i++)
        for (size_t j = 0; j < g[l].N; j++)
            tmp += g[l].Res[i][j];

    g[l].L2Norm = sqrt(tmp);
    g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
    g[l].L2NormPrev = g[l].L2Norm;
    // printf("L2Norm: %.4f\n",  g[0].L2Norm);

    auto t1 = std::chrono::system_clock::now();
    L2NormTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void applyRestriction(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    for (size_t i = 1; i < g[l].N - 1; i++)
        for (size_t j = 1; j < g[l].N - 1; j++)
            g[l].f[i][j] = (1.0 * (g[l - 1].Res[2 * i - 1][2 * j - 1] + g[l - 1].Res[2 * i - 1][2 * j + 1] + g[l - 1].Res[2 * i + 1][2 * j - 1] + g[l - 1].Res[2 * i + 1][2 * j + 1]) +
                            2.0 * (g[l - 1].Res[2 * i - 1][2 * j] + g[l - 1].Res[2 * i][2 * j - 1] + g[l - 1].Res[2 * i + 1][2 * j] + g[l - 1].Res[2 * i][2 * j + 1]) +
                            4.0 * (g[l - 1].Res[2 * i][2 * j])) *
                           0.0625;

    for (size_t i = 0; i < g[l].N; i++)
        for (size_t j = 0; j < g[l].N; j++) // Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
            g[l].U[i][j] = 0;

    auto t1 = std::chrono::system_clock::now();
    restrictionTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void applyProlongation(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    for (size_t i = 1; i < g[l].N - 1; i++)
        for (size_t j = 1; j < g[l].N - 1; j++)
            g[l - 1].U[2 * i][2 * j] += g[l].U[i][j];

    for (size_t i = 1; i < g[l].N; i++)
        for (size_t j = 1; j < g[l].N - 1; j++)
            g[l - 1].U[2 * i - 1][2 * j] += (g[l].U[i - 1][j] + g[l].U[i][j]) * 0.5;

    for (size_t i = 1; i < g[l].N - 1; i++)
        for (size_t j = 1; j < g[l].N; j++)
            g[l - 1].U[2 * i][2 * j - 1] += (g[l].U[i][j - 1] + g[l].U[i][j]) * 0.5;

    for (size_t i = 1; i < g[l].N; i++)
        for (size_t j = 1; j < g[l].N; j++)
            g[l - 1].U[2 * i - 1][2 * j - 1] += (g[l].U[i - 1][j - 1] + g[l].U[i - 1][j] + g[l].U[i][j - 1] + g[l].U[i][j]) * 0.25;

    auto t1 = std::chrono::system_clock::now();
    prolongTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

gridLevel *generateInitialConditions(size_t N0, size_t gridCount)
{
    // Default values:
    __p.nCandles = 4;
    std::vector<double> pars;
    pars.push_back(0.228162);
    pars.push_back(0.226769);
    pars.push_back(0.437278);
    pars.push_back(0.0492324);
    pars.push_back(0.65915);
    pars.push_back(0.499616);
    pars.push_back(0.59006);
    pars.push_back(0.0566329);
    pars.push_back(0.0186672);
    pars.push_back(0.894063);
    pars.push_back(0.424229);
    pars.push_back(0.047725);
    pars.push_back(0.256743);
    pars.push_back(0.754483);
    pars.push_back(0.490461);
    pars.push_back(0.0485152);

    // Allocating Timers
    smoothingTime = (double *)calloc(gridCount, sizeof(double));
    residualTime = (double *)calloc(gridCount, sizeof(double));
    restrictionTime = (double *)calloc(gridCount, sizeof(double));
    prolongTime = (double *)calloc(gridCount, sizeof(double));
    L2NormTime = (double *)calloc(gridCount, sizeof(double));

    // Allocating Grids
    gridLevel *g = (gridLevel *)malloc(sizeof(gridLevel) * gridCount);
    for (size_t i = 0; i < gridCount; i++)
    {
        g[i].N = pow(2, N0 - i) + 1;
        g[i].NN = g[i].N * g[i].N;
        g[i].n_x= g[i].N / world.dims_x;
        g[i].n_y= g[i].N / world.dims_y;
        g[i].nn= g[i].n_x * g[i].n_y;
        g[i].h = 1.0 / (g[i].N - 1);
        g[i].x_min= 0.0        + world.coord_x * g[i].n_x    * g[i].h;
        g[i].x_max= g[i].x_min +                (g[i].n_x-1) * g[i].h;
        g[i].y_min= 0.0        + world.coord_y * g[i].n_y    * g[i].h;
        g[i].y_max= g[i].y_min +                (g[i].n_y-1) * g[i].h;

        MPI_Type_contiguous(g[i].n_x, MPI_DOUBLE, &(g[i].N_bound));
        MPI_Type_commit(&(g[i].N_bound));
        MPI_Type_contiguous(g[i].n_x, MPI_DOUBLE, &(g[i].S_bound));
        MPI_Type_commit(&(g[i].S_bound));
        MPI_Type_vector(g[i].n_y, 1, g[i].n_x + 2, MPI_DOUBLE, &(g[i].E_bound));
        MPI_Type_commit(&(g[i].E_bound));
        MPI_Type_vector(g[i].n_y, 1, g[i].n_x + 2, MPI_DOUBLE, &(g[i].W_bound));
        MPI_Type_commit(&(g[i].W_bound)); 

        g[i].U=  (double*)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));
        g[i].Un= (double*)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));
        g[i].Res=(double*)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));
        g[i].f=  (double*)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));

        // g[i].U = (double **)malloc(sizeof(double*) * g[i].N);
        // for (size_t j = 0; j < g[i].N; j++)
        //     g[i].U[j] = (double *)malloc(sizeof(double) * g[i].N);
        // g[i].Un = (double **)malloc(sizeof(double *) * g[i].N);
        // for (size_t j = 0; j < g[i].N; j++)
        //     g[i].Un[j] = (double *)malloc(sizeof(double) * g[i].N);
        // g[i].Res = (double **)malloc(sizeof(double *) * g[i].N);
        // for (size_t j = 0; j < g[i].N; j++)
        //     g[i].Res[j] = (double *)malloc(sizeof(double) * g[i].N);
        // g[i].f = (double **)malloc(sizeof(double *) * g[i].N);
        // for (size_t j = 0; j < g[i].N; j++)
        //     g[i].f[j] = (double *)malloc(sizeof(double) * g[i].N);

        g[i].L2Norm = 0.0;
        g[i].L2NormPrev = std::numeric_limits<double>::max();
        g[i].L2NormDiff = std::numeric_limits<double>::max();
    }

    const int f_x= g[0].n_x + 2;
    const int f_y= g[0].n_y + 2;

    // Initial Guess
    for (int i= 0; i < f_y; ++i)
        for (int j= 0; j < f_x; ++j)
            g[0].U[i * f_x + j]= 1.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     for (size_t j = 0; j < g[0].N; j++)
    //         g[0].U[i][j] = 1.0;

    // Boundary Conditions
    if (world.N_proc == MPI_PROC_NULL)
        for (int j= 0; j < f_x; ++j)
            g[0].U[(f_y - 1) * f_x + j]= 0.0;
    if (world.E_proc == MPI_PROC_NULL)
        for (int i= 0; i < f_y; ++i)
            g[0].U[i * f_x + (f_x - 1)]= 0.0;
    if (world.S_proc == MPI_PROC_NULL)
        for (int j= 0; j < f_x; ++j)
            g[0].U[j]= 0.0;
    if (world.W_proc == MPI_PROC_NULL)
        for (int i= 0; i < f_y; ++i)
            g[0].U[i * f_x]= 0.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[0][i] = 0.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[g[0].N - 1][i] = 0.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[i][0] = 0.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[i][g[0].N - 1] = 0.0;

    // F
    for (size_t i = 0; i < g[0].N; i++)
        for (size_t j = 0; j < g[0].N; j++)
        {
            double h = 1.0 / (g[0].N - 1);
            double x = i * h;
            double y = j * h;

            g[0].f[i][j] = 0.0;

            for (size_t c = 0; c < __p.nCandles; c++)
            {
                double c3 = pars[c * 4 + 0]; // x0
                double c4 = pars[c * 4 + 1]; // y0
                double c1 = pars[c * 4 + 2];
                c1 *= 100000; // intensity
                double c2 = pars[c * 4 + 3];
                c2 *= 0.01; // Width
                g[0].f[i][j] += c1 * exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);
            }
        }
    return g;
}

void freeGrids(gridLevel *g, size_t gridCount)
{
    for (size_t i = 0; i < gridCount; i++)
    {
        free(g[i].U);
        free(g[i].Un);
        free(g[i].Res);
        free(g[i].f);
        
        MPI_Type_free(&(g[i].N_bound));
        MPI_Type_free(&(g[i].E_bound));
        MPI_Type_free(&(g[i].S_bound));
        MPI_Type_free(&(g[i].W_bound)); 
    }
    free(g);
}

void printTimings(size_t gridCount)
{
    double *timePerGrid = (double *)calloc(sizeof(double), gridCount);
    double totalSmoothingTime = 0.0;
    double totalResidualTime = 0.0;
    double totalRestrictionTime = 0.0;
    double totalProlongTime = 0.0;
    double totalL2NormTime = 0.0;

    for (size_t i = 0; i < gridCount; i++)
        timePerGrid[i] = smoothingTime[i] + residualTime[i] + restrictionTime[i] + prolongTime[i] + L2NormTime[i];
    for (size_t i = 0; i < gridCount; i++)
        totalSmoothingTime += smoothingTime[i];
    for (size_t i = 0; i < gridCount; i++)
        totalResidualTime += residualTime[i];
    for (size_t i = 0; i < gridCount; i++)
        totalRestrictionTime += restrictionTime[i];
    for (size_t i = 0; i < gridCount; i++)
        totalProlongTime += prolongTime[i];
    for (size_t i = 0; i < gridCount; i++)
        totalL2NormTime += L2NormTime[i];

    double totalMeasured = totalSmoothingTime + totalResidualTime + totalRestrictionTime + totalProlongTime + totalL2NormTime;

    printf("   Time (s)    ");
    for (size_t i = 0; i < gridCount; i++)
        printf("Grid%lu   ", i);
    printf("   Total  \n");
    printf("-------------|-");
    for (size_t i = 0; i < gridCount; i++)
        printf("--------");
    printf("|---------\n");
    printf("Smoothing    | ");
    for (size_t i = 0; i < gridCount; i++)
        printf("%2.3f   ", smoothingTime[i]);
    printf("|  %2.3f  \n", totalSmoothingTime);
    printf("Residual     | ");
    for (size_t i = 0; i < gridCount; i++)
        printf("%2.3f   ", residualTime[i]);
    printf("|  %2.3f  \n", totalResidualTime);
    printf("Restriction  | ");
    for (size_t i = 0; i < gridCount; i++)
        printf("%2.3f   ", restrictionTime[i]);
    printf("|  %2.3f  \n", totalRestrictionTime);
    printf("Prolongation | ");
    for (size_t i = 0; i < gridCount; i++)
        printf("%2.3f   ", prolongTime[i]);
    printf("|  %2.3f  \n", totalProlongTime);
    printf("L2Norm       | ");
    for (size_t i = 0; i < gridCount; i++)
        printf("%2.3f   ", L2NormTime[i]);
    printf("|  %2.3f  \n", totalL2NormTime);
    printf("-------------|-");
    for (size_t i = 0; i < gridCount; i++)
        printf("--------");
    printf("|---------\n");
    printf("Total        | ");
    for (size_t i = 0; i < gridCount; i++)
        printf("%2.3f   ", timePerGrid[i]);
    printf("|  %2.3f  \n", totalMeasured);
    printf("-------------|-");
    for (size_t i = 0; i < gridCount; i++)
        printf("--------");
    printf("|---------\n");
    printf("\n");
    printf("Running Time      : %.3fs\n", totalTime);
}
