/**********************************************************************/
// An unoptimized Naive N-Body solver for Gravity Simulations         //
// G is assumed to be 1.0                                             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin                                             //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include <stdio.h>
#include <math.h>
#include "string.h"
#include <chrono>


void checkCUDAError(const char *msg);

__global__ void forceKernel(double* xPos, double* yPos, double* zPos, double* mass, double* xFor, double* yFor, double* zFor, size_t N)
{
    size_t m = blockIdx.x*blockDim.x+threadIdx.x;

    for (size_t i = 0; i < N; i++) if (i != m)
    {
        double xDist = xPos[m] - xPos[i];
        double yDist = yPos[m] - yPos[i];
        double zDist = zPos[m] - zPos[i];
        double r     = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);
        xFor[m] += xDist*mass[m]*mass[i] / (r*r*r);
        yFor[m] += yDist*mass[m]*mass[i] / (r*r*r);
        zFor[m] += zDist*mass[m]*mass[i] / (r*r*r);
    }
}


__global__ void OptForceKernel(double* xPos, double* yPos, double* zPos,
                               double* mass,
                               double* xFor, double* yFor, double* zFor,
                               size_t N)
{
    const double EPS= 1e-9; // use to avoid slingshots and division by zero
    size_t threadId= blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < N)
    {
        size_t laneId= threadIdx.x & 0x1f; // Id within warp
        double xFor_tmp= 0.0; // to accumulate interaction loop results
        double yFor_tmp= 0.0;
        double zFor_tmp= 0.0;

        double xPosRef_reg= xPos[threadId]; // load interaction ref into registers
        double yPosRef_reg= yPos[threadId];
        double zPosRef_reg= zPos[threadId];
        double massRef_reg= mass[threadId];

        for (size_t i= 0; i < N; i+= 32) // warp-tiled interaction with all bodies
        {
            double x= xPos[i + laneId]; // load interaction object
            double y= yPos[i + laneId];
            double z= zPos[i + laneId];
            double m= mass[i + laneId];

            #pragma unroll 32
            for (size_t j= 0; j < 32; ++j) // within warp
            { // use shuffle as it beats shared memory:
                double deltaX= xPosRef_reg - __shfl_sync(0xFFFFFFFF, x, j); 
                double deltaY= yPosRef_reg - __shfl_sync(0xFFFFFFFF, y, j);
                double deltaZ= zPosRef_reg - __shfl_sync(0xFFFFFFFF, z, j);

                double distanceSquared= deltaX * deltaX
                                      + deltaY * deltaY
                                      + deltaZ * deltaZ
                                      + EPS; // avoid slingshots, division by zero
                
                double invDistance= rsqrt(distanceSquared); // use built-in arithmetic
                double invDistanceCubed= invDistance * invDistance * invDistance;
                
                double scalarForce= massRef_reg 
                                  * __shfl_sync(0xFFFFFFFF, m, j) // mass
                                  * invDistanceCubed;
                
                xFor_tmp+= scalarForce * deltaX; // add up xForce vector component
                yFor_tmp+= scalarForce * deltaY; // add up yForce vector component 
                zFor_tmp+= scalarForce * deltaZ; // add up zForce vector component
            }
            __syncthreads();
        }
        xFor[threadId]= xFor_tmp; // write results back
        yFor[threadId]= yFor_tmp;
        zFor[threadId]= zFor_tmp;
    }
}


int main(int argc, char* argv[])
{
    size_t N0 = 80;
    size_t N  = N0*N0*N0;

    // Initializing N-Body Problem

    double* xPos   = (double*) calloc (N, sizeof(double));
    double* yPos   = (double*) calloc (N, sizeof(double));
    double* zPos   = (double*) calloc (N, sizeof(double));
    double* xFor   = (double*) calloc (N, sizeof(double));
    double* yFor   = (double*) calloc (N, sizeof(double));
    double* zFor   = (double*) calloc (N, sizeof(double));
    double* mass   = (double*) calloc (N, sizeof(double));

    size_t current = 0;
    for (size_t i = 0; i < N0; i++)
    for (size_t j = 0; j < N0; j++)
    for (size_t k = 0; k < N0; k++)
    {
        xPos[current] = i;
        yPos[current] = j;
        zPos[current] = k;
        mass[current] = 1.0;
        xFor[current] = 0.0;
        yFor[current] = 0.0;
        zFor[current] = 0.0;
        current++;
    }

    // Allocating and initializing GPU memory

    double* d_xPos; cudaMalloc((void **) &d_xPos,  sizeof(double) * N); checkCUDAError("Unable to allocate storage on the device");
    double* d_yPos; cudaMalloc((void **) &d_yPos,  sizeof(double) * N); checkCUDAError("Unable to allocate storage on the device");
    double* d_zPos; cudaMalloc((void **) &d_zPos,  sizeof(double) * N); checkCUDAError("Unable to allocate storage on the device");
    double* d_xFor; cudaMalloc((void **) &d_xFor,  sizeof(double) * N); checkCUDAError("Unable to allocate storage on the device");
    double* d_yFor; cudaMalloc((void **) &d_yFor,  sizeof(double) * N); checkCUDAError("Unable to allocate storage on the device");
    double* d_zFor; cudaMalloc((void **) &d_zFor,  sizeof(double) * N); checkCUDAError("Unable to allocate storage on the device");
    double* d_mass; cudaMalloc((void **) &d_mass,  sizeof(double) * N); checkCUDAError("Unable to allocate storage on the device");

    cudaMemcpy(d_xPos, xPos, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy");
    cudaMemcpy(d_yPos, yPos, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy");
    cudaMemcpy(d_zPos, zPos, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy");
    cudaMemcpy(d_xFor, xFor, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy");
    cudaMemcpy(d_yFor, yFor, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy");
    cudaMemcpy(d_zFor, zFor, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy");
    cudaMemcpy(d_mass, mass, sizeof(double) * N, cudaMemcpyHostToDevice); checkCUDAError("Failed Initial Conditions Memcpy");

    // Calculating Kernel Geometry
    size_t threadsPerBlock  = 512;
    size_t blocksPerGrid    = ceil(double (((double)N) / ((double)threadsPerBlock)));

    // Running Force-calculation kernel
    auto startTime = std::chrono::system_clock::now();
    OptForceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_xPos, d_yPos, d_zPos,
                                                       d_mass,
                                                       d_xFor, d_yFor, d_zFor, N);
    checkCUDAError("Failed Force Kernel");
    cudaDeviceSynchronize();
    auto endTime = std::chrono::system_clock::now();

    cudaMemcpy(xFor, d_xFor, sizeof(double) * N, cudaMemcpyDeviceToHost); checkCUDAError("Failed Final Conditions Memcpy");
    cudaMemcpy(yFor, d_yFor, sizeof(double) * N, cudaMemcpyDeviceToHost); checkCUDAError("Failed Final Conditions Memcpy");
    cudaMemcpy(zFor, d_zFor, sizeof(double) * N, cudaMemcpyDeviceToHost); checkCUDAError("Failed Final Conditions Memcpy");

    cudaFree(d_xPos);
    cudaFree(d_yPos);
    cudaFree(d_zPos);
    cudaFree(d_xFor);
    cudaFree(d_yFor);
    cudaFree(d_zFor);
    cudaFree(d_mass);

    double netForce = 0.0;
    double absForce = 0.0;
    for (size_t i = 0; i < N; i++) netForce += xFor[i] + yFor[i] + zFor[i];
    for (size_t i = 0; i < N; i++) absForce += abs(xFor[i] + yFor[i] + zFor[i]);

    printf("     Net Force: %.6f\n", netForce);
    printf("Absolute Force: %.6f\n", absForce);

    if (isfinite(netForce) == false)      { printf("Verification Failed: Net force is not a finite value!\n"); exit(-1); }
    if (fabs(netForce) > 0.00001)         { printf("Verification Failed: Force equilibrium not conserved!\n"); exit(-1); }
    if (isfinite(absForce) == false)      { printf("Verification Failed: Absolute Force is not a finite value!\n"); exit(-1); }

    printf("Time: %.8fs\n", std::chrono::duration<double>(endTime-startTime).count());
    return 0;
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
