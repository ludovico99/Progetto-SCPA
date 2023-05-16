// Copyright 2014, Cranfield University
// All rights reserved
// Author: Michał Czapiński (mczapinski@gmail.com)
//
// Demonstrates the use of page-locked host memory (to speed up CPU-GPU transfers).
// Also demonstrates handling of 2D arrays: allocating row-aligned arrays (to meet the
// global memory coalescing requirements), and the use of cudaMemcpy2D routine.

#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>

// Matrix dimensions.
const int COLS = 4000;
const int ROWS = 4096;

// TODO(later) Play with different COLS values.
// What happens if it naturaly meets the alignment requirements (multiple of 128)?
// What happens if it does not? What happens if COLS becomes small? Why?

int main(int argc, char** argv) {

// ----------------------- Host memory initialisation ----------------------- //

  // TODO Allocate h_A in page-locked memory.
  float* h_A = 0;
  // checkCudaErrors(cudaHostAlloc(...));

  // h_B is allocated in pageable memory.
  float* h_B = new float[ROWS * COLS];

  // TODO Uncomment when h_A is properly allocated.
//  srand(time(0));
//  for (int row = 0; row < ROWS; ++row) {
//    for (int col = 0; col < COLS; ++col) {
//      int idx = row * COLS + col;
//      h_A[idx] = static_cast<float>(rand()) / RAND_MAX;
//      h_B[idx] = static_cast<float>(rand()) / RAND_MAX;
//    }
//  }

// ---------------- Device memory allocation (no alignment) ----------------- //

  checkCudaErrors(cudaSetDevice(gpuGetMaxGflopsDeviceId()));

  // Allocate global memory on the GPU.
  float *d_A, *d_B;

  const size_t BYTES = ROWS * COLS * sizeof(float);
  checkCudaErrors(cudaMalloc((void**) &d_A, BYTES));
  checkCudaErrors(cudaMalloc((void**) &d_B, BYTES));

// ----------------- Upload data to the GPU (no alignment) ------------------ //

  std::cout << "Uploading linear data to the GPU (cudaMemcpy)" << std::endl;

  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  timer->start();
  checkCudaErrors(cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice));

  timer->stop();
  std::cout << "Page-locked memory: " << timer->getTime() << " ms ("
      << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl;

  timer->reset();
  timer->start();
  checkCudaErrors(cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice));

  timer->stop();
  std::cout << "Pageable memory:    " << timer->getTime() << " ms ("
      << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl << std::endl;

// --------------- Download data from the GPU (no alignment) ---------------- //

  std::cout << "Downloading linear data from the GPU (cudaMemcpy)" << std::endl;

  timer->reset();
  timer->start();
  checkCudaErrors(cudaMemcpy(h_A, d_A, BYTES, cudaMemcpyDeviceToHost));
  timer->stop();

  std::cout << "Page-locked memory: " << timer->getTime() << " ms ("
        << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl;

  timer->reset();
  timer->start();
  checkCudaErrors(cudaMemcpy(h_B, d_B, BYTES, cudaMemcpyDeviceToHost));
  timer->stop();

  std::cout << "Pageable memory:    " << timer->getTime() << " ms ("
      << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl << std::endl;

// --------------- Device memory allocation (with alignment) ---------------- //

  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));

  // TODO Allocate d_A and d_B with correct row allignment.
  // checkCudaErrors(cudaMallocPitch(...));

// ---------------- Upload data to the GPU (with alignment) ----------------- //

  std::cout << "Uploading row-aligned data to the GPU (cudaMemcpy2D)" << std::endl;

  timer->reset();
  timer->start();
  // TODO Upload h_A to the GPU using cudaMemcpy2D.
  // checkCudaErrors(cudaMemcpy2D(...));

  timer->stop();
  std::cout << "Page-locked memory: " << timer->getTime() << " ms ("
      << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl;

  timer->reset();
  timer->start();
  // TODO Upload h_B to the GPU using cudaMemcpy2D.
  // checkCudaErrors(cudaMemcpy2D(...));

  timer->stop();
  std::cout << "Pageable memory:    " << timer->getTime() << " ms ("
      << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl << std::endl;

// -------------- Download data from the GPU (with alignment) --------------- //

  std::cout << "Downloading row-aligned data from the GPU (cudaMemcpy2D)" << std::endl;

  timer->reset();
  timer->start();
  // TODO Download d_A from the GPU using cudaMemcpy2D.
  // checkCudaErrors(cudaMemcpy2D(...));

  timer->stop();
  std::cout << "Page-locked memory: " << timer->getTime() << " ms ("
      << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl;

  timer->reset();
  timer->start();
  // TODO Download d_B from the GPU using cudaMemcpy2D.
  // checkCudaErrors(cudaMemcpy2D(...));

  timer->stop();
  std::cout << "Pageable memory:    " << timer->getTime() << " ms ("
      << 1e-6 * BYTES / timer->getTime() << " GB/s)" << std::endl;

// ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  // TODO Deallocate d_A, d_B, and h_A correctly

  delete[] h_B;

  // Required for cuda-memcheck to detect leaks correctly.
  checkCudaErrors(cudaDeviceReset());
  return 0;
}
