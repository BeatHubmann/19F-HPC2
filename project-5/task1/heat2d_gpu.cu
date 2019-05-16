/**********************************************************************/
// A now optimized Multigrid Solver for the Heat Equation             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin, Georgios Arampatzis                        //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <limits>
#include "heat2d_gpu.hpp"
#include "string.h"
#include <chrono>


#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32



pointsInfo __p;

int main(int argc, char* argv[])
{
    double tolerance = 1e-0; // L2 Difference Tolerance before reaching convergence.
    size_t N0 = 10; // 2^N0 + 1 elements per side

    // Multigrid parameters -- Find the best configuration!
    size_t gridCount       = N0-1;     // Number of Multigrid levels to use
    size_t downRelaxations = 5; // Number of Relaxations before restriction
    size_t upRelaxations   = 0;   // Number of Relaxations after prolongation

    gridLevel* g = generateInitialConditions(N0, gridCount);
    cudaGridLevel* d_g= cudaInitialConditions(g, gridCount);

    auto startTime = std::chrono::system_clock::now();
    
    while (g[0].L2NormDiff > tolerance)  // Multigrid solver start
    {
        applyJacobi(g, 0, downRelaxations); // Relaxing the finest grid first
        calculateResidual(g, 0); // Calculating Initial Residual

        for (size_t grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
        {
            applyRestriction(g, grid); // Restricting the residual to the coarser grid's solution vector (f)
            applyJacobi(g, grid, downRelaxations); // Smoothing coarser level
            calculateResidual(g, grid); // Calculating Coarse Grid Residual
        }

        for (size_t grid = gridCount-1; grid > 0; grid--) // Going up the V-Cycle
        {
            applyProlongation(g, grid); // Prolonging solution for coarser level up to finer level
            applyJacobi(g, grid, upRelaxations); // Smoothing finer level
        }
        calculateL2Norm(g, 0); // Calculating Residual L2 Norm
    }  // Multigrid solver end
    
    auto endTime = std::chrono::system_clock::now();
    totalTime = std::chrono::duration<double>(endTime-startTime).count();



    auto cudaStartTime=  std::chrono::system_clock::now();
    while (d_g[0].L2NormDiff > tolerance)  
    {
        cudaApplyJacobi(d_g, 0, downRelaxations); // Relaxing the finest grid first
        cudaCalculateResidual(d_g, 0); // Calculating Initial Residual

        for (size_t grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
        {
            cudaApplyRestriction(d_g, grid); // Restricting the residual to the coarser grid's solution vector (f)
            cudaApplyJacobi(d_g, grid, downRelaxations); // Smoothing coarser level
            cudaCalculateResidual(d_g, grid); // Calculating Coarse Grid Residual
        }

        for (size_t grid = gridCount-1; grid > 0; grid--) // Going up the V-Cycle
        {
            cudaApplyProlongation(d_g, grid); // Prolonging solution for coarser level up to finer level
            cudaApplyJacobi(d_g, grid, upRelaxations); // Smoothing finer level
        }
        cudaCalculateL2Norm(d_g, 0); // Calculating Residual L2 Norm
    }
    cudaDeviceSynchronize();
    auto cudaEndTime = std::chrono::system_clock::now();
    cudaTotalTime = std::chrono::duration<double>(cudaEndTime-cudaStartTime).count();

 
    printTimings(gridCount);
    printf("L2Norm: %.4f\n",  g[0].L2Norm);
    cudaPrintTimings(gridCount);
    printf("L2Norm: %.4f\n",  d_g[0].L2Norm);
    freeGrids(g, gridCount);
    freeCudaGrids(d_g, gridCount);
    return 0;
}

void applyJacobi(gridLevel* g, size_t l, size_t relaxations)
{
    auto t0 = std::chrono::system_clock::now();

    double h1 = 0.25;
    double h2 = g[l].h*g[l].h;
    for (size_t r = 0; r < relaxations; r++)
    {
        double** tmp = g[l].Un; g[l].Un = g[l].U; g[l].U = tmp;
        for (size_t i = 1; i < g[l].N-1; i++)
            for (size_t j = 1; j < g[l].N-1; j++) // Perform a Jacobi Iteration
                g[l].U[i][j] = (g[l].Un[i-1][j] + g[l].Un[i+1][j] + g[l].Un[i][j-1] + g[l].Un[i][j+1] + g[l].f[i][j]*h2)*h1;
    }

    auto t1 = std::chrono::system_clock::now();
    smoothingTime[l] += std::chrono::duration<double>(t1-t0).count();
}

void cudaApplyJacobi(cudaGridLevel* g, size_t l, size_t relaxations)
{
    auto t0 = std::chrono::system_clock::now();

    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 dimGrid(iDivUp(g[l].N, BLOCKSIZE_X), iDivUp(g[l].N, BLOCKSIZE_Y));
   
    for (size_t r = 0; r < relaxations; r++)
    {
        double* tmp= g[l].Un; g[l].Un= g[l].U; g[l].U= tmp;
        cudaJacobiKernel<<<dimGrid, dimBlock>>>(g, l);
    }
    auto t1 = std::chrono::system_clock::now();
    cudaSmoothingTime[l] += std::chrono::duration<double>(t1-t0).count();
}

__global__ void cudaJacobiKernel(cudaGridLevel* g, size_t l)
{
    int i= threadIdx.x + blockIdx.x * blockDim.x;
    int j= threadIdx.y + blockIdx.y * blockDim.y;
    int NXY= g[l].N;
    double h1= 0.25;
    double h2= g[l].h * g[l].h;

    int C=  j    * NXY + i  ;
    int N= (j+1) * NXY + i  ;
    int E=  j    * NXY + i+1;
    int S= (j-1) * NXY + i  ;
    int W=  j    * NXY + i-1;
    if (i>0 && i<(NXY-1) && j>0 && j<(NXY-1))
        g[l].U[C]= (g[l].Un[S] + g[l].Un[N] + g[l].Un[W] + g[l].Un[E] + g[l].f[C]*h2) * h1;
    __syncthreads();
}


void calculateResidual(gridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    double h2 = 1.0 / pow(g[l].h,2);

    for (size_t i = 1; i < g[l].N-1; i++)
        for (size_t j = 1; j < g[l].N-1; j++)
            g[l].Res[i][j] = g[l].f[i][j] + (g[l].U[i-1][j] + g[l].U[i+1][j] - 4*g[l].U[i][j] + g[l].U[i][j-1] + g[l].U[i][j+1]) * h2;

    auto t1 = std::chrono::system_clock::now();
    residualTime[l] += std::chrono::duration<double>(t1-t0).count();
}


void cudaCalculateResidual(cudaGridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();
    
    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 dimGrid(iDivUp(g[l].N, BLOCKSIZE_X), iDivUp(g[l].N, BLOCKSIZE_Y));
    
    cudaResidualKernel<<<dimGrid, dimBlock>>>(g, l);

    auto t1 = std::chrono::system_clock::now();
    cudaResidualTime[l] += std::chrono::duration<double>(t1-t0).count();
}


__global__ void cudaResidualKernel(cudaGridLevel* g, size_t l)
{
    double h2= 1.0 / pow(g[l].h, 2);

    int i= threadIdx.x + blockIdx.x * blockDim.x;
    int j= threadIdx.y + blockIdx.y * blockDim.y;
    int NXY= g[l].N;

    int C=  j    * NXY + i  ;
    int N= (j+1) * NXY + i  ;
    int E=  j    * NXY + i+1;
    int S= (j-1) * NXY + i  ;
    int W=  j    * NXY + i-1;
    if (i>0 && i<(NXY-1) && j>0 && j<(NXY-1))
        g[l].Res[C]= g[l].f[C] +  (g[l].U[S] + g[l].U[N] - 4*g[l].U[C] + g[l].U[W] + g[l].U[E]) * h2;

    __syncthreads();
}


void calculateL2Norm(gridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    double tmp = 0.0;

    for (size_t i = 0; i < g[l].N; i++)
        for (size_t j = 0; j < g[l].N; j++)
            g[l].Res[i][j] = g[l].Res[i][j]*g[l].Res[i][j];

    for (size_t i = 0; i < g[l].N; i++)
        for (size_t j = 0; j < g[l].N; j++)
            tmp += g[l].Res[i][j];

    g[l].L2Norm = sqrt(tmp);
    g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
    g[l].L2NormPrev = g[l].L2Norm;
    // printf("L2Norm: %.4f\n",  g[0].L2Norm);

    auto t1 = std::chrono::system_clock::now();
    L2NormTime[l] += std::chrono::duration<double>(t1-t0).count();
}

void cudaCalculateL2Norm(cudaGridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();
    
    // dim3 dimBlock(BLOCKSIZE_X);
    // dim3 dimGrid(iDivUp(g[l].N, BLOCKSIZE_X));

    // cudaL2NormKernel<<<dimGrid, dimBlock>>>(g, l);
    printf("calc L2Norm on level %zu\n",l);
    cudaL2NormKernel<<<1, 1>>>(g, l); checkCUDAError("Failed cudaL2NormKernel");
    printf("L2Norm: %.4f\n",  g[l].L2Norm);
    double* result= new double(999);
    cudaMemcpy(result, g[l].U, sizeof(double), cudaMemcpyDeviceToHost); checkCUDAError("Failed Memcpy L2Norm result");
    printf("L2 debug result %f\n",*result);
    g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
    g[l].L2NormPrev = g[l].L2Norm;
    auto t1 = std::chrono::system_clock::now();
    cudaL2NormTime[l] += std::chrono::duration<double>(t1-t0).count();
}


__global__ void cudaL2NormKernel(cudaGridLevel* g, size_t l)
{
    int threadId= blockIdx.x * blockDim.x + threadIdx.x;
    printf("thread %d out before calc: \n", threadId );
    int NXY= g[l].N;
    *g[l].d_L2Norm= norm(NXY*NXY, g[l].Res);
    __syncthreads();
}


void applyRestriction(gridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    for (size_t i = 1; i < g[l].N-1; i++)
        for (size_t j = 1; j < g[l].N-1; j++)
            g[l].f[i][j] = ( 1.0*( g[l-1].Res[2*i-1][2*j-1] + g[l-1].Res[2*i-1][2*j+1] + g[l-1].Res[2*i+1][2*j-1]   + g[l-1].Res[2*i+1][2*j+1] )   +
                             2.0*( g[l-1].Res[2*i-1][2*j]   + g[l-1].Res[2*i][2*j-1]   + g[l-1].Res[2*i+1][2*j]     + g[l-1].Res[2*i][2*j+1] ) +
                             4.0*( g[l-1].Res[2*i][2*j] ) ) * 0.0625;

    for (size_t i = 0; i < g[l].N; i++)
        for (size_t j = 0; j < g[l].N; j++) // Resetting U vector for the coarser level before smoothing -- Find out if this is really necessary.
            g[l].U[i][j] = 0;

    auto t1 = std::chrono::system_clock::now();
    restrictionTime[l] += std::chrono::duration<double>(t1-t0).count();
}


void cudaApplyRestriction(cudaGridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 dimGrid(iDivUp(g[l].N, BLOCKSIZE_X), iDivUp(g[l].N, BLOCKSIZE_Y));
    
    cudaRestrictionKernel<<<dimGrid, dimBlock>>>(g, l);
    
    auto t1 = std::chrono::system_clock::now();
    cudaRestrictionTime[l] += std::chrono::duration<double>(t1-t0).count();
}


__global__ void cudaRestrictionKernel(cudaGridLevel* g, size_t l)
{
    int i= threadIdx.x + blockIdx.x * blockDim.x;
    int j= threadIdx.y + blockIdx.y * blockDim.y;
    int NXY= g[l].N;

    if (i>0 && i<(NXY-1) && j>0 && j<(NXY-1))
        g[l].f[i * NXY + j]= (1.0*( g[l-1].Res[(2*i-1) * NXY + 2*j-1] + g[l-1].Res[(2*i-1) * NXY + 2*j+1] + g[l-1].Res[(2*i+1) * NXY + 2*j-1] + g[l-1].Res[(2*i+1) * NXY + 2*j+1]) +
                       2.0*( g[l-1].Res[(2*i-1) * NXY + 2*j  ] + g[l-1].Res[(2*i)   * NXY + 2*j-1] + g[l-1].Res[(2*i+1) * NXY + 2*j  ] + g[l-1].Res[(2*i)   * NXY + 2*j+1]) +
                       4.0*( g[l-1].Res[(2*i)   * NXY + 2*j  ])) * 0.0625;
    g[l].U[i * NXY + j]= 0;
    __syncthreads();
}





void applyProlongation(gridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    for (size_t i = 1; i < g[l].N-1; i++)
        for (size_t j = 1; j < g[l].N-1; j++)
            g[l-1].U[2*i][2*j] += g[l].U[i][j];

    for (size_t i = 1; i < g[l].N; i++)
        for (size_t j = 1; j < g[l].N-1; j++)
            g[l-1].U[2*i-1][2*j] += ( g[l].U[i-1][j] + g[l].U[i][j] ) *0.5;

    for (size_t i = 1; i < g[l].N-1; i++)
        for (size_t j = 1; j < g[l].N; j++)
            g[l-1].U[2*i][2*j-1] += ( g[l].U[i][j-1] + g[l].U[i][j] ) *0.5;

    for (size_t i = 1; i < g[l].N; i++)
        for (size_t j = 1; j < g[l].N; j++)
            g[l-1].U[2*i-1][2*j-1] += ( g[l].U[i-1][j-1] + g[l].U[i-1][j] + g[l].U[i][j-1] + g[l].U[i][j] ) *0.25;

    auto t1 = std::chrono::system_clock::now();
    prolongTime[l] += std::chrono::duration<double>(t1-t0).count();
}

void cudaApplyProlongation(cudaGridLevel* g, size_t l)
{
    auto t0 = std::chrono::system_clock::now();

    dim3 dimBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 dimGrid(iDivUp(g[l].N, BLOCKSIZE_X), iDivUp(g[l].N, BLOCKSIZE_Y));
    
    cudaProlongationKernel<<<dimGrid, dimBlock>>>(g, l);
    
    auto t1 = std::chrono::system_clock::now();
    cudaProlongTime[l] += std::chrono::duration<double>(t1-t0).count();
}

__global__ void cudaProlongationKernel(cudaGridLevel* g, size_t l)
{
    int i= threadIdx.x + blockIdx.x * blockDim.x;
    int j= threadIdx.y + blockIdx.y * blockDim.y;
    int NXY= g[l].N;

    if (i>0 && i<(NXY-1) && j>0 && j<(NXY-1))
        g[l-1].U[(2*i) * NXY + 2*j]+= g[l].U[i * NXY + j];

    if (i>0 && i<NXY     && j>0 && j<(NXY-1))
        g[l-1].U[(2*i-1) * NXY + 2*j]+= (g[l].U[(i-1) * NXY + j] + g[l].U[i * NXY + j]) * 0.5;

    if (i>0 && i<(NXY-1) && j>0 && j<(NXY  ))
        g[l-1].U[(2*i) * NXY + 2*j-1]+= (g[l].U[i * NXY + j-1] + g[l].U[i * NXY + j]) * 0.5;

    if (i>0 && i<NXY     && j>0 && j<NXY    )
        g[l-1].U[(2*i-1) * NXY + 2*j-1]+= (g[l].U[(i-1) * NXY + j-1] + g[l].U[(i-1) * NXY + j] + g[l].U[i * NXY + j-1] + g[l].U[i * NXY + j]) * 0.25;
    __syncthreads();
}



gridLevel* generateInitialConditions(size_t N0, size_t gridCount)
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
    smoothingTime = (double*) calloc (gridCount, sizeof(double));
    residualTime = (double*) calloc (gridCount, sizeof(double));
    restrictionTime = (double*) calloc (gridCount, sizeof(double));
    prolongTime = (double*) calloc (gridCount, sizeof(double));
    L2NormTime = (double*) calloc (gridCount, sizeof(double));

    // Allocating Grids
    gridLevel* g = (gridLevel*) malloc(sizeof(gridLevel) * gridCount);
    for (size_t i = 0; i < gridCount; i++)
    {
        g[i].N = pow(2, N0-i) + 1;
        g[i].h = 1.0/(g[i].N-1);

        g[i].U   = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].U[j]   = (double*) malloc(sizeof(double) * g[i].N);
        g[i].Un  = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].Un[j]  = (double*) malloc(sizeof(double) * g[i].N);
        g[i].Res = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].Res[j] = (double*) malloc(sizeof(double) * g[i].N);
        g[i].f   = (double**) malloc(sizeof(double*) * g[i].N); for (size_t j = 0; j < g[i].N ; j++) g[i].f[j]   = (double*) malloc(sizeof(double) * g[i].N);

        g[i].L2Norm = 0.0;
        g[i].L2NormPrev = std::numeric_limits<double>::max();
        g[i].L2NormDiff = std::numeric_limits<double>::max();
    }

    // Initial Guess
    for (size_t i = 0; i < g[0].N; i++) for (size_t j = 0; j < g[0].N; j++) g[0].U[i][j] = 1.0;

    // Boundary Conditions
    for (size_t i = 0; i < g[0].N; i++) g[0].U[0][i]        = 0.0;
    for (size_t i = 0; i < g[0].N; i++) g[0].U[g[0].N-1][i] = 0.0;
    for (size_t i = 0; i < g[0].N; i++) g[0].U[i][0]        = 0.0;
    for (size_t i = 0; i < g[0].N; i++) g[0].U[i][g[0].N-1] = 0.0;

    // F
    for (size_t i = 0; i < g[0].N; i++)
    for (size_t j = 0; j < g[0].N; j++)
    {
        double h = 1.0/(g[0].N-1);
        double x = i*h;
        double y = j*h;

        g[0].f[i][j] = 0.0;

        for (size_t c = 0; c < __p.nCandles; c++)
        {
            double c3 = pars[c*4  + 0]; // x0
            double c4 = pars[c*4  + 1]; // y0
            double c1 = pars[c*4  + 2]; c1 *= 100000;// intensity
            double c2 = pars[c*4  + 3]; c2 *= 0.01;// Width
            g[0].f[i][j] += c1*exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);
        }
    }

    return g;
}


cudaGridLevel* cudaInitialConditions(gridLevel* g, size_t gridCount)
{
    cudaSmoothingTime = (double*) calloc (gridCount, sizeof(double));
    cudaResidualTime = (double*) calloc (gridCount, sizeof(double));
    cudaRestrictionTime = (double*) calloc (gridCount, sizeof(double));
    cudaProlongTime = (double*) calloc (gridCount, sizeof(double));
    cudaL2NormTime = (double*) calloc (gridCount, sizeof(double));

    cudaGridLevel* d_g= (cudaGridLevel*) malloc(sizeof(cudaGridLevel) * gridCount);

    for (size_t i= 0; i < gridCount; i++)
    {
        d_g[i].N= g[i].N;
        d_g[i].h= g[i].h;
        
        cudaMalloc((void**) &d_g[i].U, sizeof(double) * d_g[i].N * d_g[i].N); checkCUDAError("Unable to allocate storage on the device");
        for (int j= 0; j < d_g[i].N; j++) cudaMemcpy(&d_g[i].U[j * d_g[i].N], g[i].U[j], d_g[i].N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy"); 
        cudaMalloc((void**) &d_g[i].Un, sizeof(double) * d_g[i].N * d_g[i].N); checkCUDAError("Unable to allocate storage on the device");
        for (int j= 0; j < d_g[i].N; j++) cudaMemcpy(&d_g[i].Un[j * d_g[i].N], g[i].Un[j], d_g[i].N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy"); 
        cudaMalloc((void**) &d_g[i].Res, sizeof(double) * d_g[i].N * d_g[i].N); checkCUDAError("Unable to allocate storage on the device");
        for (int j= 0; j < d_g[i].N; j++) cudaMemcpy(&d_g[i].Res[j * d_g[i].N], g[i].Res[j], d_g[i].N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy"); 
        cudaMalloc((void**) &d_g[i].ResOut, sizeof(double) * d_g[i].N * d_g[i].N); checkCUDAError("Unable to allocate storage on the device");
        for (int j= 0; j < d_g[i].N; j++) cudaMemcpy(&d_g[i].ResOut[j * d_g[i].N], g[i].Res[j], d_g[i].N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy"); 
        cudaMalloc((void**) &d_g[i].f, sizeof(double) * d_g[i].N * d_g[i].N); checkCUDAError("Unable to allocate storage on the device");
        for (int j= 0; j < d_g[i].N; j++) cudaMemcpy(&d_g[i].f[j * d_g[i].N], g[i].f[j], d_g[i].N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy"); 
        cudaMalloc((void**) &d_g[i].d_L2Norm, sizeof(double)); checkCUDAError("Unable to allocate storage on the device");

        d_g[i].L2Norm=  g[i].L2Norm;
        d_g[i].L2NormPrev= g[i].L2NormPrev;
        d_g[i].L2NormDiff= g[i].L2NormDiff;

    }
    return d_g;
}



void freeGrids(gridLevel* g, size_t gridCount)
{
    for (size_t i = 0; i < gridCount; i++)
    {
        for (size_t j = 0; j < g[i].N ; j++) free(g[i].U[j]);
        for (size_t j = 0; j < g[i].N ; j++) free(g[i].Un[j]);
        for (size_t j = 0; j < g[i].N ; j++) free(g[i].f[j]);
        for (size_t j = 0; j < g[i].N ; j++) free(g[i].Res[j]);
        free(g[i].U);
        free(g[i].Un);
        free(g[i].f);
        free(g[i].Res);
    }
    free(g);
}



void freeCudaGrids(cudaGridLevel* g, size_t gridCount)
{
    for (size_t i = 0; i < gridCount; i++)
    {
        cudaFree(g[i].U);
        cudaFree(g[i].Un);
        cudaFree(g[i].f);
        cudaFree(g[i].Res);
        
        cudaFree(g[i].ResOut);
        cudaFree(g[i].d_L2Norm);
    }
    free(g);
}


void printTimings(size_t gridCount)
{
    double* timePerGrid = (double*) calloc (sizeof(double), gridCount);
    double totalSmoothingTime = 0.0;
    double totalResidualTime = 0.0;
    double totalRestrictionTime = 0.0;
    double totalProlongTime = 0.0;
    double totalL2NormTime = 0.0;

    for (size_t i = 0; i < gridCount; i++) timePerGrid[i] = smoothingTime[i] + residualTime[i] + restrictionTime[i] + prolongTime[i] + L2NormTime[i];
    for (size_t i = 0; i < gridCount; i++) totalSmoothingTime += smoothingTime[i];
    for (size_t i = 0; i < gridCount; i++) totalResidualTime += residualTime[i];
    for (size_t i = 0; i < gridCount; i++) totalRestrictionTime += restrictionTime[i];
    for (size_t i = 0; i < gridCount; i++) totalProlongTime += prolongTime[i];
    for (size_t i = 0; i < gridCount; i++) totalL2NormTime += L2NormTime[i];

    double totalMeasured = totalSmoothingTime + totalResidualTime + totalRestrictionTime + totalProlongTime + totalL2NormTime;

    printf("\n==== CPU ====\n");
    printf("   Time (s)    "); for (size_t i = 0; i < gridCount; i++) printf("Grid%zu   ", i);                    printf("   Total  \n");
    printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
    printf("Smoothing    | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", smoothingTime[i]);    printf("|  %2.3f  \n", totalSmoothingTime);
    printf("Residual     | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", residualTime[i]);     printf("|  %2.3f  \n", totalResidualTime);
    printf("Restriction  | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", restrictionTime[i]);  printf("|  %2.3f  \n", totalRestrictionTime);
    printf("Prolongation | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", prolongTime[i]);      printf("|  %2.3f  \n", totalProlongTime);
    printf("L2Norm       | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", L2NormTime[i]);       printf("|  %2.3f  \n", totalL2NormTime);
    printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
    printf("Total        | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", timePerGrid[i]); printf("|  %2.3f  \n", totalMeasured);
    printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
    printf("\n");
    printf("Running Time      : %.3fs\n", totalTime);
}

void cudaPrintTimings(size_t gridCount)
{
    double* timePerGrid = (double*) calloc (sizeof(double), gridCount);
    double totalSmoothingTime = 0.0;
    double totalResidualTime = 0.0;
    double totalRestrictionTime = 0.0;
    double totalProlongTime = 0.0;
    double totalL2NormTime = 0.0;

    for (size_t i = 0; i < gridCount; i++) timePerGrid[i] = cudaSmoothingTime[i] + cudaResidualTime[i] + cudaRestrictionTime[i] + cudaProlongTime[i] + cudaL2NormTime[i];
    for (size_t i = 0; i < gridCount; i++) totalSmoothingTime += cudaSmoothingTime[i];
    for (size_t i = 0; i < gridCount; i++) totalResidualTime += cudaResidualTime[i];
    for (size_t i = 0; i < gridCount; i++) totalRestrictionTime += cudaRestrictionTime[i];
    for (size_t i = 0; i < gridCount; i++) totalProlongTime += cudaProlongTime[i];
    for (size_t i = 0; i < gridCount; i++) totalL2NormTime += cudaL2NormTime[i];

    double totalMeasured = totalSmoothingTime + totalResidualTime + totalRestrictionTime + totalProlongTime + totalL2NormTime;

    printf("\n==== GPU ====\n");
    printf("   Time (s)    "); for (size_t i = 0; i < gridCount; i++) printf("Grid%zu   ", i);                    printf("   Total  \n");
    printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
    printf("Smoothing    | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", cudaSmoothingTime[i]);    printf("|  %2.3f  \n", totalSmoothingTime);
    printf("Residual     | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", cudaResidualTime[i]);     printf("|  %2.3f  \n", totalResidualTime);
    printf("Restriction  | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", cudaRestrictionTime[i]);  printf("|  %2.3f  \n", totalRestrictionTime);
    printf("Prolongation | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", cudaProlongTime[i]);      printf("|  %2.3f  \n", totalProlongTime);
    printf("L2Norm       | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", cudaL2NormTime[i]);       printf("|  %2.3f  \n", totalL2NormTime);
    printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
    printf("Total        | "); for (size_t i = 0; i < gridCount; i++) printf("%2.3f   ", timePerGrid[i]); printf("|  %2.3f  \n", totalMeasured);
    printf("-------------|-"); for (size_t i = 0; i < gridCount; i++) printf("--------"); printf("|---------\n");
    printf("\n");
    printf("Running Time      : %.3fs\n", cudaTotalTime);
}






void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE);
    }
}

__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }