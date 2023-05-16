// Copyright 2014, Cranfield University
// All rights reserved
// Author: Michał Czapiński (mczapinski@gmail.com)
//
// Adds two matrices on the GPU. Matrices are stored in linear memory in row-major order,
// i.e. A[i, j] is stored in i * COLS + j element of the vector.

#include <iostream>

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

// Matrix dimensions. Can you make these input arguments?
const int ROWS = 4096;
const int COLS = 4096;

// TODO(later) Play a bit with the block size. Is 16x16 setup the fastest possible?
// Note: For meaningful time measurements you need sufficiently large matrix.
const dim3 BLOCK_DIM(16, 16);

// Simple CPU implementation of matrix addition.
void CpuMatrixAdd(int rows, int cols, const float* A, const float* B, float* C) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      int idx = row * cols + col;
      C[idx] = A[idx] + B[idx];
    }
  }
}

// GPU implementation of matrix add using one CUDA thread per vector element.
__global__ void GpuMatrixAdd(int rows, int cols, const float* A, const float* B, float* C) {
  // TODO Calculate indices of matrix elements added by this thread. Assume 2D grid of blocks.
  int col = 0;
  int row = 0;
  // TODO(later) Does it matter if you index rows with x or y dimension of threadIdx and blockIdx?

  // TODO Calculate the element index in the global memory and add the values.
  // TODO Make sure that no threads access memory outside the allocated area.
}

int main(int argc, char** argv) {

// ----------------------- Host memory initialisation ----------------------- //

  float* h_A = new float[ROWS * COLS];
  float* h_B = new float[ROWS * COLS];
  float* h_C = new float[ROWS * COLS];

  srand(time(0));
  for (int row = 0; row < ROWS; ++row) {
    for (int col = 0; col < COLS; ++col) {
      int idx = row * COLS + col;
      h_A[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
      h_B[idx] = 100.0f * static_cast<float>(rand()) / RAND_MAX;
    }
  }

// ---------------------- Device memory initialisation ---------------------- //

  // TODO Allocate global memory on the GPU.
  float *d_A, *d_B, *d_C;

  // TODO Copy matrices from the host (CPU) to the device (GPU).

// ------------------------ Calculations on the CPU ------------------------- //

  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  timer->start();
  CpuMatrixAdd(ROWS, COLS, h_A, h_B, h_C);

  timer->stop();
  std::cout << "CPU time: " << timer->getTime() << " ms." << std::endl;

// ------------------------ Calculations on the GPU ------------------------- //

  // TODO Calculate the dimension of the grid of blocks (2D).
  const dim3 GRID_DIM;

  timer->reset();
  timer->start();
  GpuMatrixAdd<<<GRID_DIM, BLOCK_DIM>>>(ROWS, COLS, d_A, d_B, d_C);
  checkCudaErrors(cudaDeviceSynchronize());

  timer->stop();
  std::cout << "GPU time: " << timer->getTime() << " ms." << std::endl;

  // TODO Download the resulting matrix d_C from the device and store it in h_A.

  // Now let's check if the results are the same.
  float diff = 0.0f;
  for (int row = 0; row < ROWS; ++row) {
    for (int col = 0; col < COLS; ++col) {
      int idx = row * COLS + col;
      diff = std::max(diff, std::abs(h_A[idx] - h_C[idx]));
    }
  }
  std::cout << "Max diff = " << diff << std::endl;  // Should be (very close to) zero.

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return 0;
}
