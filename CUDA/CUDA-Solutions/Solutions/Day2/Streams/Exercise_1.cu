#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define REPEAT 1

__global__ void arrayFunc(float* d_idata, float* d_jdata, float* d_odata, int size)
{
  int tid =  blockDim.x * blockIdx.x + threadIdx.x; 
  if (tid < size) {
    for(int i=0; i < REPEAT; i++)
       d_odata[tid] = d_idata[tid] * __expf(d_jdata[tid]);
  }
}

void initArrayData(float * array, float alpha, int size);
void arrayFuncCPU(const float* h_idata, const float* h_jdata, float* h_odata, int size);

#define NSIZE 1048576

int 
main (void) {
  
  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;
  
  int nsize = NSIZE;
  int nThreads = 256;
  int nBlocks;

  cudaEvent_t start, end;
  float eventEtime;

  // calculate block number
  nBlocks = (nsize-1) / nThreads + 1;
  printf("Number of elements: %d\n", nsize);
  printf("GPU execution with %d blocks each one of %d threads\n", nBlocks, nThreads);

  // allocation and initialization of host buffers
  h_a = (float*) malloc (nsize * sizeof(float));
  h_b = (float*) malloc (nsize * sizeof(float));
  h_c = (float*) malloc (nsize * sizeof(float));

  initArrayData(h_a, 1.0f, nsize);
  initArrayData(h_b, 10.0f, nsize);

//-- insert CUDE code ----------------
  // allocation of device buffers
  cudaMalloc((void**)&d_a, nsize * sizeof(float));
  cudaMalloc((void**)&d_b, nsize * sizeof(float));
  cudaMalloc((void**)&d_c, nsize * sizeof(float));
//------------------------------------

  // creation of cuda events: start, end
  cudaEventCreate(&start);  
  cudaEventCreate(&end);

  printf ("\nGPU computation ... ");

  cudaEventRecord(start,0);  

//-- insert CUDA code ----------------
  // host to device buffer copies
  cudaMemcpy(d_a, h_a, nsize * sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_b, h_b, nsize * sizeof(float), cudaMemcpyHostToDevice); 
//------------------------------------

//-- insert CUDA code ----------------
  // arrayFunc kernel launch
  arrayFunc<<<nBlocks,nThreads>>>(d_a, d_b, d_c, nsize);
//------------------------------------

//-- insert CUDA code ----------------
  // copy back of results from device
  cudaMemcpy(h_c, d_c, nsize*sizeof(float), cudaMemcpyDeviceToHost);
//------------------------------------

  cudaEventRecord(end,0);  
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&eventEtime, start, end);

  printf ("ok\n");

  printf("Elapsed time on GPU: %.2f ms\n", eventEtime);

  // host computation
  printf("\nCPU computation ... ");
  float *cpuResult;
  float eventTimeCPU;
  cudaMallocHost((void**)&cpuResult, nsize * sizeof(float));
  cudaEventRecord(start,0);

  arrayFuncCPU(h_a, h_b, cpuResult, nsize);

  cudaEventRecord(end,0);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&eventTimeCPU, start, end);
  printf ("ok\n");
  printf("Elapsed time on CPU: %.2f ms\n", eventTimeCPU);
  printf("\nSpeed UP CPU/GPU %.1fx\n", eventTimeCPU/eventEtime);

  printf("\nCheck results:\n");
  printf ("h_c[0]       = %f\n", h_c[0]);
  printf ("cpuResult[0] = %f\n", cpuResult[0]);

  // free resources on device
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  // free resources on host
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}


void 
initArrayData(float * array, float alpha, int size)
{ 
  int i;
  for (i=0; i< size; i++) 
    array[i] = alpha * (float) rand() / (float) RAND_MAX;
}

void arrayFuncCPU(const float* h_idata, const float* h_jdata, float* h_odata, int size)
{
   int i, j;
   for (i=0; i<size; i++)
     for (j=0; j<REPEAT; j++)
        h_odata[i] = h_idata[i] * expf(h_jdata[i]);
}
