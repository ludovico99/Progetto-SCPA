// Copyright 2014, Cranfield University
// All rights reserved
// Author: Michał Czapiński (mczapinski@gmail.com)
//
// A program demonstrating the impact of warp divergence on the overall performance.
// It can only be observed when enough blocks are run (typicaly more than 10000).

#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>

// Size of each block of threads on the GPU.
const int BLOCK_DIM = 32;

// Number of blocks to run.
const dim3 GRID_DIM(1024, 1024);

__device__ float result[BLOCK_DIM];

/*! GPU kernel executing the given number of divergent threads -
 *  (divergentThreads + 1) execution paths.
 *
 * 	\param divergentThreads number of threads diverging from the main execution path.
 */
__global__ void WarpDivergence(int divergentThreads) {
  // TODO Try to implement a kernel which executes divergentThreads different execution paths,
  // effectively introducing warp divergence and slower execution.
}

int main(int argc, char** argv) {
  checkCudaErrors(cudaSetDevice(gpuGetMaxGflopsDeviceId()));

  StopWatchInterface* timer = NULL;
  sdkCreateTimer(&timer);

  // This is to "wake up" the GPU and avoid spurious overhead on the first time measurement.
  checkCudaErrors(cudaDeviceSynchronize());

  for (int i = 0; i < BLOCK_DIM; ++i) {
    timer->reset();
    timer->start();

    WarpDivergence<<< GRID_DIM, BLOCK_DIM >>>(i);
    checkCudaErrors(cudaDeviceSynchronize());

    timer->stop();
    std::cout << (i + 1) << " execution path(s): " << timer->getTime() << " ms" << std::endl;
  }

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;
  return 0;
}
