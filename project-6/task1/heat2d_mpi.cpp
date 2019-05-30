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
#include <algorithm>

#define GRIDCOUNT (2)
#define NPOINTSPOWER (3)

pointsInfo __p;

// to hold all relevant info for cartesian MPI grid members:
struct WorldStruct
{
    int my_rank;
    int num_procs;

    int dims_x;
    int dims_y;

    int coord_x;
    int coord_y;

    int cart_rank;
    MPI_Comm cart_comm;

    int N_proc;
    int E_proc;
    int S_proc;
    int W_proc;
} world;

void print(gridLevel *g, size_t l, double *matrix)
{
    const int f_x = g[l].n_x + 2;
    const int f_y = g[l].n_y + 2;
    for (int i = 0; i < f_y; ++i)
    {
        for (int j = 0; j < f_x; ++j)
        {
            printf("%f\t", matrix[i * f_x + j]);
        }
        printf("\n\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    // Setup MPI and create cartesian grid
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world.my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world.num_procs);

    int dims[2] = {0, 0};
    MPI_Dims_create(world.num_procs, 2, dims);
    world.dims_x = dims[0];
    world.dims_y = dims[1];

    int periods[2] = {false, false};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &world.cart_comm);
    MPI_Comm_rank(world.cart_comm, &world.cart_rank);
    MPI_Cart_shift(world.cart_comm, 0, 1, &world.W_proc, &world.E_proc);
    MPI_Cart_shift(world.cart_comm, 1, 1, &world.S_proc, &world.N_proc);

    int coords[2] = {0, 0};
    MPI_Cart_coords(world.cart_comm, world.cart_rank, 2, coords);
    world.coord_x = coords[0];
    world.coord_y = coords[1];
    // Print MPI cartesian grid partitioning:
    if (!world.my_rank)
    {
        printf("\nMPI num_procs: %i\n", world.num_procs);
        printf("MPI dims_x: \%i\n", world.dims_x);
        printf("MPI dims_y: \%i\n", world.dims_y);
    }

    double tolerance = 1e-0;  // L2 Difference Tolerance before reaching convergence.
    size_t N0 = NPOINTSPOWER; // 2^N0 + 1 elements per side

    // Multigrid parameters
    size_t gridCount = GRIDCOUNT; // Number of Multigrid levels to use
    assert(gridCount <= 4);       // according to task specification 1d
    size_t downRelaxations = 3;   // Number of Relaxations before restriction
    size_t upRelaxations = 3;     // Number of Relaxations after prolongation

    // Setup all grid levels including MPI partitioning information:
    gridLevel *g = generateInitialConditions(N0, gridCount);

    printf("\nRNK %d: x=%f...%f, y=%f...%f\n", world.my_rank, g[0].x_min, g[0].x_max, g[0].y_min, g[0].y_max);
    printf("n_x: %d\tn_y: %d\tn_x_limit: %d\tn_y_limit: %d\n", g[0].n_x, g[0].n_y, g[0].n_x_limit, g[0].n_y_limit);
    printf("     \tN: %d\n", world.N_proc);
    printf("W: %d\t**%d**\tE: %d\n", world.W_proc, world.my_rank, world.E_proc);
    printf("     \tS: %d\n", world.S_proc);

    MPI_Barrier(MPI_COMM_WORLD); // Make sure all ranks are ready

    // print(g, 0, g->U);
    // print(g, 0, g->f);

    // calculateResidual(g, 0); // Calculating Initial Residual
    // for (int i = 0; i < world.num_procs; ++i)
    // {
    //     if (i == world.my_rank)
    //     {
    //         printf("RNK %d STARTING U:\n", world.my_rank);
    //         print(g, 0, g->U);
    //         printf("RNK %d STARTING f:\n", world.my_rank);
    //         print(g, 0, g->f);
    //         printf("RNK %d STARTING residual:\n", world.my_rank);
    //         print(g, 0, g->Res);
    //     }
    //     MPI_Barrier(world.cart_comm);
    // }
    // calculateL2Norm(g, 0);

    auto startTime = std::chrono::system_clock::now();

    while (g[0].L2NormDiff > tolerance) // Multigrid solver start
    {
        applyJacobi(g, 0, downRelaxations); // Relaxing the finest grid first
        calculateResidual(g, 0);            // Calculating Initial Residual
        // for (int i= 0; i < world.num_procs; ++i)
        // {
        //     if (i == world.my_rank)
        //     {
        //         printf("RNK %d Residual:\n", world.my_rank);
        //         print(g, 0, g->Res);
        //     }
        //     MPI_Barrier(world.cart_comm);
        // }
        for (size_t grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
        {
            applyRestriction(g, grid);             // Restricting the residual to the coarser grid's solution vector (f)
            for (int i= 0; i < world.num_procs; ++i)
            {
                if (i == world.my_rank)
                {
                    printf("RNK %d Restriction:\n", world.my_rank);
                    print(g,grid, g[grid].f);
                }
                MPI_Barrier(world.cart_comm);
            }
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

    // Sum up individual ranks' timing for statistics:
    if (!world.my_rank)
    {
        MPI_Reduce(MPI_IN_PLACE, smoothingTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(MPI_IN_PLACE, residualTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(MPI_IN_PLACE, restrictionTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(MPI_IN_PLACE, prolongTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(MPI_IN_PLACE, L2NormTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
    }
    else
    {
        MPI_Reduce(smoothingTime, smoothingTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(residualTime, residualTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(restrictionTime, restrictionTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(prolongTime, prolongTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
        MPI_Reduce(L2NormTime, L2NormTime, gridCount, MPI_DOUBLE, MPI_SUM, 0, world.cart_comm);
    }

    if (!world.my_rank)
    {
        printTimings(gridCount);
        printf("MPI L2Norm: %.4f\n", g[0].L2Norm);
    }
    freeGrids(g, gridCount);
    MPI_Comm_free(&world.cart_comm);
    MPI_Finalize();
    return 0;
}

void applyJacobi(gridLevel *g, size_t l, size_t relaxations)
{
    auto t0 = std::chrono::system_clock::now();

    double h1 = 0.25;
    double h2 = g[l].h * g[l].h;

    const int f_x = g[l].n_x + 2;
    const int f_y = g[l].n_y + 2;

    const int x_start = (world.W_proc == MPI_PROC_NULL) ? 2 : 1;
    const int x_end = (world.E_proc == MPI_PROC_NULL) ? std::min(f_x - 2, g[l].n_x_limit) : f_x - 1;
    const int y_start = (world.S_proc == MPI_PROC_NULL) ? 2 : 1;
    const int y_end = (world.N_proc == MPI_PROC_NULL) ? std::min(f_y - 2, g[l].n_y_limit) : f_y - 1;

    for (size_t r = 0; r < relaxations; r++)
    {
         // double **tmp = g[l].Un;
        // g[l].Un = g[l].U;
        // g[l].U = tmp;
        std::swap(g[l].U, g[l].Un);

        for (int i = y_start; i < y_end; ++i)
            for (int j = x_start; j < x_end; ++j) // Perform a Jacobi Iteration
                g[l].U[i * f_x + j] = (g[l].Un[(i - 1) * f_x + j] + g[l].Un[(i + 1) * f_x + j] + g[l].Un[i * f_x + j - 1] + g[l].Un[i * f_x + j + 1] + g[l].f[i * f_x + j] * h2) * h1;

        MPI_Request request[8];
        int request_idx = 0;
        // Exchange boundaries in a hopefully coordinated way:
        // x-direction: W<->E:
        if (world.coord_x % 2 == 0) // even ranks: send W-recv W-send E-recv E
        {
            MPI_Isend(&g[l].U[f_x + 1], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Irecv(&g[l].U[f_x], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Isend(&g[l].U[f_x + g[l].n_x], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Irecv(&g[l].U[f_x + f_x - 1], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
        }
        else // odd ranks:  recv E-send E-recv W-send W
        {
            MPI_Irecv(&g[l].U[f_x + f_x - 1], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Isend(&g[l].U[f_x + g[l].n_x], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Irecv(&g[l].U[f_x], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Isend(&g[l].U[f_x + 1], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
        }
        // y-direction: N<->S:
        if (world.coord_y % 2 == 0) // even ranks: send N-recv N-send S-recv S
        {
            MPI_Isend(&g[l].U[f_x * g[l].n_y + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Irecv(&g[l].U[f_x * (f_y - 1) + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Isend(&g[l].U[f_x + 1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Irecv(&g[l].U[1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
        }
        else // odd ranks:  recv S-send S-recv N-send N
        {
            MPI_Irecv(&g[l].U[1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Isend(&g[l].U[f_x + 1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Irecv(&g[l].U[f_x * (f_y - 1) + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
            MPI_Isend(&g[l].U[f_x * g[l].n_y + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
        }
        MPI_Waitall(request_idx, request, MPI_STATUS_IGNORE); // Make sure all exchanges are done

        //         for (int i = 0; i < world.num_procs; ++i)
        // {
        //     if (i == world.my_rank)
        //     {
        //         printf("RNK %d Jacobi U:\n", world.my_rank);
        //         print(g, 0, g->U);
        //     }
        //     MPI_Barrier(world.cart_comm);
        // }

       
    }

    auto t1 = std::chrono::system_clock::now();
    smoothingTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void calculateResidual(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    const double h2 = 1.0 / pow(g[l].h, 2);

    const int f_x = g[l].n_x + 2;
    const int f_y = g[l].n_y + 2;

    const int x_start = (world.W_proc == MPI_PROC_NULL) ? 2 : 1;
    const int x_end = (world.E_proc == MPI_PROC_NULL) ? std::min(f_x - 2, g[l].n_x_limit) : f_x - 1;
    const int y_start = (world.S_proc == MPI_PROC_NULL) ? 2 : 1;
    const int y_end = (world.N_proc == MPI_PROC_NULL) ? std::min(f_y - 2, g[l].n_y_limit) : f_y - 1;

    // printf("RNK %d - i/y: %d...%d\tj/x: %d...%d\n", world.my_rank, y_start, y_end, x_start,y_end);
    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
        {
            g[l].Res[i * f_x + j] = g[l].f[i * f_x + j] + (g[l].U[(i - 1) * f_x + j] + g[l].U[(i + 1) * f_x + j] - 4 * g[l].U[i * f_x + j] + g[l].U[i * f_x + j - 1] + g[l].U[i * f_x + j + 1]) * h2;
            // printf("RNK %d LVL %zu Res [%d, %d] = %f <- %f\t%f\t%f\t%f\t%f\t%f\n", world.my_rank, l, i , j , g[l].Res[i * f_x + j],g[l].f[i * f_x + j], g[l].U[(i - 1) * f_x + j], g[l].U[(i + 1) * f_x + j], g[l].U[i * f_x + j], g[l].U[i * f_x + j - 1], g[l].U[i * f_x + j + 1]);
        }

    // Exchange residuals: In principle only needed for multiple grids:
    MPI_Request request[8];
    int request_idx = 0;
    // Exchange boundaries in a hopefully coordinated way:
    // x-direction: W<->E:
    if (world.coord_x % 2 == 0) // even ranks: send W-recv W-send E-recv E
    {
        MPI_Isend(&g[l].Res[f_x + 1], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l].Res[f_x], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l].Res[f_x + g[l].n_x], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l].Res[f_x + f_x - 1], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    else // odd ranks:  recv E-send E-recv W-send W
    {
        MPI_Irecv(&g[l].Res[f_x + f_x - 1], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l].Res[f_x + g[l].n_x], 1, g[l].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l].Res[f_x], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l].Res[f_x + 1], 1, g[l].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    // y-direction: N<->S:
    if (world.coord_y % 2 == 0) // even ranks: send N-recv N-send S-recv S
    {
        MPI_Isend(&g[l].Res[f_x * g[l].n_y + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l].Res[f_x * (f_y - 1) + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l].Res[f_x + 1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l].Res[1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    else // odd ranks:  recv S-send S-recv N-send N
    {
        MPI_Irecv(&g[l].Res[1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l].Res[f_x + 1], 1, g[l].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l].Res[f_x * (f_y - 1) + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l].Res[f_x * g[l].n_y + 1], 1, g[l].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    MPI_Waitall(request_idx, request, MPI_STATUS_IGNORE); // Make sure all exchanges are done

    auto t1 = std::chrono::system_clock::now();
    residualTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void calculateL2Norm(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    const int f_x = g[l].n_x + 2;
    const int f_y = g[l].n_y + 2;

    const int x_start = 1;
    const int x_end = std::min(f_x - 1, g[l].n_x_limit + 1);
    const int y_start = 1;
    const int y_end = std::min(f_y - 1, g[l].n_y_limit + 1);

    double tmp = 0.0;

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            g[l].Res[i * f_x + j] = g[l].Res[i * f_x + j] * g[l].Res[i * f_x + j];

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            tmp += g[l].Res[i * f_x + j];

    // for (size_t i = 0; i < g[l].N; i++)
    //     for (size_t j = 0; j < g[l].N; j++)
    //         g[l].Res[i][j] = g[l].Res[i][j] * g[l].Res[i][j];

    // for (size_t i = 0; i < g[l].N; i++)
    //     for (size_t j = 0; j < g[l].N; j++)
    //         tmp += g[l].Res[i][j];

    g[l].L2NormLocal = sqrt(tmp);
    MPI_Allreduce(&tmp, &g[l].L2Norm, 1, MPI_DOUBLE, MPI_SUM, world.cart_comm); // sum up squares of local L2Norms...
    g[l].L2Norm = sqrt(g[l].L2Norm);                                            // ...then take root of sum of squares to get overall main L2Norm
    g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
    g[l].L2NormPrev = g[l].L2Norm;
    if (!world.my_rank)
        printf("L2 Norm: %f\n", g[l].L2Norm);
    auto t1 = std::chrono::system_clock::now();
    L2NormTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void applyRestriction(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    const int f_x = g[l].n_x + 2;
    const int f_y = g[l].n_y + 2;
    const int l_x = g[l - 1].n_x + 2;

    int x_start = (world.W_proc == MPI_PROC_NULL) ? 2 : 1;
    int x_end = (world.E_proc == MPI_PROC_NULL) ? std::min(f_x - 2, g[l].n_x_limit) : f_x - 1;
    int y_start = (world.S_proc == MPI_PROC_NULL) ? 2 : 1;
    int y_end = (world.N_proc == MPI_PROC_NULL) ? std::min(f_y - 2, g[l].n_y_limit) : f_y - 1;

    // if (l > 0) printf("x: %d...%d\ty: %d...%d\n",x_start,x_end,y_start,y_end);

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
        {
            g[l].f[i * f_x + j] = (1.0 * (g[l - 1].Res[(2 * i - 1 - 1) * l_x + 2 * j - 1 - 1] + g[l - 1].Res[(2 * i - 1 - 1) * l_x + 2 * j + 1 - 1] + g[l - 1].Res[(2 * i + 1 - 1) * l_x + 2 * j - 1 - 1] + g[l - 1].Res[(2 * i + 1 - 1) * l_x + 2 * j + 1 - 1]) + 2.0 * (g[l - 1].Res[(2 * i - 1 - 1) * l_x + 2 * j - 1] + g[l - 1].Res[(2 * i - 1) * l_x + 2 * j - 1 - 1] + g[l - 1].Res[(2 * i + 1 - 1) * l_x + 2 * j - 1] + g[l - 1].Res[(2 * i - 1) * l_x + 2 * j + 1 - 1]) + 4.0 * (g[l - 1].Res[(2 * i - 1) * l_x + 2 * j - 1])) * 0.0625;
            // printf("f[%d, %d] = %f\n",i-1,j-1,g[l].f[i * f_x + j]);
        }
    // for (size_t i = 1; i < g[l].N - 1; i++)
    //     for (size_t j = 1; j < g[l].N - 1; j++)
    //         g[l].f[i][j] = (1.0 * (g[l - 1].Res[2 * i - 1][2 * j - 1] + g[l - 1].Res[2 * i - 1][2 * j + 1] + g[l - 1].Res[2 * i + 1][2 * j - 1] + g[l - 1].Res[2 * i + 1][2 * j + 1])
    //                        +2.0 * (g[l - 1].Res[2 * i - 1][2 * j] + g[l - 1].Res[2 * i][2 * j - 1] + g[l - 1].Res[2 * i + 1][2 * j] + g[l - 1].Res[2 * i][2 * j + 1])
    //                        +4.0 * (g[l - 1].Res[2 * i][2 * j])) *
    //                        0.0625;

    x_start = 0;
    x_end = f_x;
    y_start = 0;
    y_end = f_y;

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            g[l].U[i * f_x + j] = 0;

    // for (size_t i = 0; i < g[l].N; i++)
    //     for (size_t j = 0; j < g[l].N; j++) // Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
    //         g[l].U[i][j] = 0;

    auto t1 = std::chrono::system_clock::now();
    restrictionTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void applyProlongation(gridLevel *g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    const int f_x = g[l].n_x + 2;
    const int f_y = g[l].n_y + 2;
    const int l_x = g[l - 1].n_x + 2;
    const int l_y = g[l - 1].n_x + 2;

    int x_start = (world.W_proc == MPI_PROC_NULL) ? 2 : 1;
    int x_end = (world.E_proc == MPI_PROC_NULL) ? std::min(f_x - 2, g[l].n_x_limit) : f_x - 1;
    int y_start = (world.S_proc == MPI_PROC_NULL) ? 2 : 1;
    int y_end = (world.N_proc == MPI_PROC_NULL) ? std::min(f_y - 2, g[l].n_y_limit) : f_y - 1;

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            g[l - 1].U[(2 * i - 1) * l_x + 2 * j - 1] += g[l].U[i * f_x + j];

    y_end += 1;

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            g[l - 1].U[(2 * i - 1 - 1) * l_x + 2 * j - 1] += (g[l].U[(i - 1) * f_x + j] + g[l].U[i * f_x + j]) * 0.5;

    x_end += 1;
    y_end -= 1;

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            g[l - 1].U[(2 * i - 1) * l_x + 2 * j - 1 - 1] += (g[l].U[i * f_x + j - 1] + g[l].U[i * f_x + j]) * 0.5;

    y_end += 1;

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            g[l - 1].U[(2 * i - 1 - 1) * l_x + 2 * j - 1 - 1] += (g[l].U[(i - 1) * f_x + j - 1] + g[l].U[(i - 1) * f_x + j] + g[l].U[i * f_x + j - 1] + g[l].U[i * f_x + j]) * 0.25;

    // for (size_t i = 1; i < g[l].N - 1; i++)
    //     for (size_t j = 1; j < g[l].N - 1; j++)
    //         g[l - 1].U[2 * i][2 * j] += g[l].U[i][j];

    // for (size_t i = 1; i < g[l].N; i++)
    //     for (size_t j = 1; j < g[l].N - 1; j++)
    //         g[l - 1].U[2 * i - 1][2 * j] += (g[l].U[i - 1][j] + g[l].U[i][j]) * 0.5;

    // for (size_t i = 1; i < g[l].N - 1; i++)
    //     for (size_t j = 1; j < g[l].N; j++)
    //         g[l - 1].U[2 * i][2 * j - 1] += (g[l].U[i][j - 1] + g[l].U[i][j]) * 0.5;

    // for (size_t i = 1; i < g[l].N; i++)
    //     for (size_t j = 1; j < g[l].N; j++)
    //         g[l - 1].U[2 * i - 1][2 * j - 1] += (g[l].U[i - 1][j - 1] + g[l].U[i - 1][j] + g[l].U[i][j - 1] + g[l].U[i][j]) * 0.25;

    MPI_Request request[8];
    int request_idx = 0;
    // Exchange boundaries in a hopefully coordinated way:
    // x-direction: W<->E:
    if (world.coord_x % 2 == 0) // even ranks: send W-recv W-send E-recv E
    {
        MPI_Isend(&g[l - 1].U[l_x + 1], 1, g[l - 1].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l - 1].U[l_x], 1, g[l - 1].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l - 1].U[l_x + g[l - 1].n_x], 1, g[l - 1].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l - 1].U[l_x + l_x - 1], 1, g[l - 1].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    else // odd ranks:  recv E-send E-recv W-send W
    {
        MPI_Irecv(&g[l - 1].U[l_x + l_x - 1], 1, g[l - 1].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l - 1].U[l_x + g[l - 1].n_x], 1, g[l - 1].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l - 1].U[l_x], 1, g[l - 1].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l - 1].U[l_x + 1], 1, g[l - 1].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    // y-direction: N<->S:
    if (world.coord_y % 2 == 0) // even ranks: send N-recv N-send S-recv S
    {
        MPI_Isend(&g[l - 1].U[l_x * g[l - 1].n_y + 1], 1, g[l - 1].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l - 1].U[l_x * (l_y - 1) + 1], 1, g[l - 1].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l - 1].U[l_x + 1], 1, g[l - 1].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l - 1].U[1], 1, g[l - 1].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    else // odd ranks:  recv S-send S-recv N-send N
    {
        MPI_Irecv(&g[l - 1].U[1], 1, g[l - 1].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l - 1].U[l_x + 1], 1, g[l - 1].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Irecv(&g[l - 1].U[l_x * (l_y - 1) + 1], 1, g[l - 1].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
        MPI_Isend(&g[l - 1].U[l_x * g[l - 1].n_y + 1], 1, g[l - 1].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);
    }
    MPI_Waitall(request_idx, request, MPI_STATUS_IGNORE); // Make sure all exchanges are done

    auto t1 = std::chrono::system_clock::now();
    prolongTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

gridLevel *generateInitialConditions(size_t N0, size_t gridCount)
{
    // Default values:
    __p.nCandles = 4;
    std::vector<double> pars;
    pars.push_back(0.228162);  //x_0
    pars.push_back(0.226769);  //y_0
    pars.push_back(0.437278);  //intensity
    pars.push_back(0.0492324); //width

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
        g[i].n_x = ceil((double)g[i].N / (double)world.dims_x); // to make sure we're covering all points
        g[i].n_y = ceil((double)g[i].N / (double)world.dims_y);
        g[i].nn = g[i].n_x * g[i].n_y;
        g[i].h = 1.0 / (g[i].N - 1);
        // to limit local grid usage to not exceed original grid points for edge partitions:
        g[i].n_x_limit = (world.coord_x * g[i].n_x + g[i].n_x > g[i].N) ? g[i].N - world.coord_x * g[i].n_x : g[i].n_x;
        g[i].n_y_limit = (world.coord_y * g[i].n_y + g[i].n_y > g[i].N) ? g[i].N - world.coord_y * g[i].n_y : g[i].n_y;
        // printf("I'm %d level %zu - n_x_limit: %d n_y_limit: %d\n", world.my_rank, i, g[i].n_x_limit, g[i].n_y_limit);
        g[i].x_min = 0.0 + world.coord_x * g[i].n_x * g[i].h;
        g[i].x_max = g[i].x_min + (g[i].n_x_limit - 1) * g[i].h;
        g[i].y_min = 0.0 + world.coord_y * g[i].n_y * g[i].h;
        g[i].y_max = g[i].y_min + (g[i].n_y_limit - 1) * g[i].h;
        // Create named MPI_Types for to exchange halos on all four sides:
        MPI_Type_contiguous(g[i].n_x, MPI_DOUBLE, &(g[i].N_bound));
        MPI_Type_commit(&(g[i].N_bound));
        MPI_Type_contiguous(g[i].n_x, MPI_DOUBLE, &(g[i].S_bound));
        MPI_Type_commit(&(g[i].S_bound));
        MPI_Type_vector(g[i].n_y, 1, g[i].n_x + 2, MPI_DOUBLE, &(g[i].E_bound));
        MPI_Type_commit(&(g[i].E_bound));
        MPI_Type_vector(g[i].n_y, 1, g[i].n_x + 2, MPI_DOUBLE, &(g[i].W_bound));
        MPI_Type_commit(&(g[i].W_bound));
        // Assign memory, now no longer tedious pointers to pointers:
        g[i].U = (double *)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));
        g[i].Un = (double *)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));
        g[i].Res = (double *)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));
        g[i].f = (double *)calloc(sizeof(double), (g[i].n_x + 2) * (g[i].n_y + 2));

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

    const int f_x = g[0].n_x + 2;
    const int f_y = g[0].n_y + 2;

    int x_start = (world.W_proc == MPI_PROC_NULL) ? 2 : 0;
    int x_end = (world.E_proc == MPI_PROC_NULL) ? std::min(f_x - 2, g[0].n_x_limit) : f_x;
    int y_start = (world.S_proc == MPI_PROC_NULL) ? 2 : 0;
    int y_end = (world.N_proc == MPI_PROC_NULL) ? std::min(f_y - 2, g[0].n_y_limit) : f_y;

    // Initial guess (= 1.0) wherever not boundary/ghost cells
    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
            g[0].U[i * f_x + j] = 1.0;

    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[0][i] = 0.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[g[0].N - 1][i] = 0.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[i][0] = 0.0;
    // for (size_t i = 0; i < g[0].N; i++)
    //     g[0].U[i][g[0].N - 1] = 0.0;

    // F
    x_start = 1;
    x_end = std::min(f_x - 1, g[0].n_x_limit + 1);
    y_start = 1;
    y_end = std::min(f_y - 1, g[0].n_y_limit + 1);

    for (int i = y_start; i < y_end; ++i)
        for (int j = x_start; j < x_end; ++j)
        {
            double h = g[0].h;
            double x = g[0].x_min + (j - 1) * h;
            double y = g[0].y_min + (i - 1) * h;

            for (size_t c = 0; c < __p.nCandles; ++c)
            {
                double c3 = pars[c * 4 + 0]; // x0
                double c4 = pars[c * 4 + 1]; // y0
                double c1 = pars[c * 4 + 2];
                c1 *= 100000; // intensity
                double c2 = pars[c * 4 + 3];
                c2 *= 0.01;                                                               // Width
                g[0].f[i * f_x + j] += c1 * exp(-(pow(c4 - x, 2) + pow(c3 - y, 2)) / c2); // ATTN: swapped axis labelling when compared to CPU version
            }
        }
    // for (size_t i = 0; i < g[0].N; i++)
    //     for (size_t j = 0; j < g[0].N; j++)
    //     {
    //         double h = 1.0 / (g[0].N - 1);
    //         double x = i * h;
    //         double y = j * h;

    //         g[0].f[i][j] = 0.0;

    //         for (size_t c = 0; c < __p.nCandles; c++)
    //         {
    //             double c3 = pars[c * 4 + 0]; // x0
    //             double c4 = pars[c * 4 + 1]; // y0
    //             double c1 = pars[c * 4 + 2];
    //             c1 *= 100000; // intensity
    //             double c2 = pars[c * 4 + 3];
    //             c2 *= 0.01; // Width
    //             g[0].f[i][j] += c1 * exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);
    //         }
    //     }

    // Exchange initial F
    MPI_Request request[8];
    int request_idx = 0;

    MPI_Irecv(&g[0].f[f_x], 1, g[0].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
    MPI_Irecv(&g[0].f[f_x + f_x - 1], 1, g[0].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
    MPI_Irecv(&g[0].f[1], 1, g[0].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
    MPI_Irecv(&g[0].f[f_x * (f_y - 1) + 1], 1, g[0].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);

    MPI_Isend(&g[0].f[f_x + 1], 1, g[0].W_bound, world.W_proc, 1, world.cart_comm, &request[request_idx++]);
    MPI_Isend(&g[0].f[f_x + g[0].n_x], 1, g[0].E_bound, world.E_proc, 1, world.cart_comm, &request[request_idx++]);
    MPI_Isend(&g[0].f[f_x + 1], 1, g[0].S_bound, world.S_proc, 1, world.cart_comm, &request[request_idx++]);
    MPI_Isend(&g[0].f[f_x * g[0].n_y + 1], 1, g[0].N_bound, world.N_proc, 1, world.cart_comm, &request[request_idx++]);

    MPI_Waitall(request_idx, request, MPI_STATUS_IGNORE);

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

    free(smoothingTime);
    free(residualTime);
    free(restrictionTime);
    free(prolongTime);
    free(L2NormTime);

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
