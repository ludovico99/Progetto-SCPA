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
void getChunkInfo(int gpuID, int i, int *d_offset, int *chunk_size, int *h_offset, int *chunk_stream, int nSize, int chunk_size_max, int num_chunk, int num_streams, int gpu_number);

#define NSIZE 2097152
#define CHUNKSIZEMAX 65536
#define NUMSTREAMS 8

#define NGPU 2

int 
main (void) {
  
  float *h_a, *h_b, *h_c;
//-- insert CUDA code ----------------
  // buffers declarations
  float *d_a[NGPU], *d_b[NGPU], *d_c[NGPU];
//------------------------------------
  
  int nsize = NSIZE;
  int nThreads = 256;
  int nBlocks;

  cudaEvent_t start, end;
  float eventEtime;

  int chunk_size_max = CHUNKSIZEMAX;
  int num_streams = NUMSTREAMS;
  int num_chunk;
  int i;
  int h_offset, d_offset;
  int chunk_size, chunk_stream;

//-- insert CUDA code ----------------
  // streams declarations
  cudaStream_t streams[NGPU][NUMSTREAMS];
//------------------------------------

  int gpuID, gpu_number;


  // allocation and initialization of host buffers
  cudaMallocHost((void**)&h_a, nsize * sizeof(float));
  cudaMallocHost((void**)&h_b, nsize * sizeof(float));
  cudaMallocHost((void**)&h_c, nsize * sizeof(float));

  initArrayData(h_a, 1.0f, nsize);
  initArrayData(h_b, 10.0f, nsize);


  cudaGetDeviceCount( &gpu_number );
  // chunk number calculation
  num_chunk = (nsize/gpu_number-1) / chunk_size_max + 1;

  printf("Number of gpu:      %d\n", gpu_number);
  printf("Number of elements: %d\n", nsize);
  printf("Number of streams:  %d\n", num_streams);
  printf("Number of chunks per GPU:   %d\n", num_chunk);

  for (gpuID=0; gpuID<gpu_number; gpuID++) {

//-- insert CUDA code ----------------
    // GPU selection
    cudaSetDevice(gpuID);
    // allocation of device buffers
    cudaMalloc((void**)&d_a[gpuID], num_streams * chunk_size_max * sizeof(float));
    cudaMalloc((void**)&d_b[gpuID], num_streams * chunk_size_max * sizeof(float));
    cudaMalloc((void**)&d_c[gpuID], num_streams * chunk_size_max * sizeof(float));

    // streams creation
    for (i = 0; i< num_streams; i++)
      cudaStreamCreate(&streams[gpuID][i]);
//------------------------------------

  }

  // creation of cuda events: start, end
  cudaSetDevice(0);
  cudaEventCreate(&start);  
  cudaEventCreate(&end);

  printf ("\nGPU computation ... ");

  cudaEventRecord(start,0);  

  for (gpuID = 0; gpuID < gpu_number; gpuID++) {

    cudaSetDevice(gpuID);

    for (i = 0; i < num_chunk; i++) {

      // please see getChunkInfo function description
      getChunkInfo(gpuID, i, &d_offset, &chunk_size, &h_offset, &chunk_stream, nsize, chunk_size_max, num_chunk, num_streams, gpu_number);

//-- insert CUDA code ----------------
      // host to device buffer copies
      cudaMemcpyAsync(d_a[gpuID]+d_offset, h_a+h_offset, chunk_size*sizeof(float), cudaMemcpyHostToDevice, streams[gpuID][chunk_stream]);
      cudaMemcpyAsync(d_b[gpuID]+d_offset, h_b+h_offset, chunk_size*sizeof(float), cudaMemcpyHostToDevice, streams[gpuID][chunk_stream]);
//------------------------------------

      // block number calculation
      nBlocks = (chunk_size-1) / nThreads + 1;

//-- insert CUDA code ----------------
      // arrayFunc kernel launch
      arrayFunc<<<nBlocks,nThreads, 0, streams[gpuID][chunk_stream]>>>(d_a[gpuID]+d_offset, d_b[gpuID]+d_offset, d_c[gpuID]+d_offset, chunk_size);
//------------------------------------

//-- insert CUDA code ----------------
      // copy back of results from device
      cudaMemcpyAsync(h_c+h_offset, d_c[gpuID]+d_offset, chunk_size*sizeof(float), cudaMemcpyDeviceToHost, streams[gpuID][chunk_stream]);
//------------------------------------

    }
  }
  
  for (gpuID = 0; gpuID < gpu_number; gpuID++) {
    cudaSetDevice(gpuID);
    cudaDeviceSynchronize();
  }

  cudaSetDevice(0);
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
  for (gpuID = 0; gpuID < gpu_number; gpuID++) {
    cudaSetDevice(gpuID);
    for (i = 0; i< num_streams; i++)
      cudaStreamDestroy(streams[gpuID][i]);
  }
  cudaSetDevice(0);
  cudaEventDestroy(start);
  cudaEventDestroy(end);

  // free resources on host
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);

  return 0;

}


void 
initArrayData(float * array, float alpha, int size)
{ 
  int i;
  for (i=0; i< size; i++) 
    array[i] = alpha * (float) rand() / (float) RAND_MAX;
}

// getChunkInfo is used to compute some useful information starting
//   from the i-th chunk, the total number of used chunks, 
//   the maximum chunk size and the array size to process
// getChunkInfo returns:
// * chunk_size: the number of elements to use in current chunk
// * chunk_stream: the stream to use to process i-th chunk
// * the X_offsets to use for accessing the correct elements of host 
//   and device arrays in data movements and kernel launch
//
void getChunkInfo(int gpuID, int i, int *d_offset, int *chunk_size, int *h_offset, int *chunk_stream, int nSize, int chunk_size_max, int num_chunk, int num_streams, int gpu_number){

  int Reminder = nSize%chunk_size_max;

  *h_offset = i*chunk_size_max + gpuID * (nSize/gpu_number);
  *chunk_stream = i%num_streams;
  *chunk_size   = chunk_size_max;
  *d_offset = *chunk_stream * chunk_size_max;

  if (Reminder && (i == num_chunk-1)) *chunk_size = Reminder;

}

void arrayFuncCPU(const float* h_idata, const float* h_jdata, float* h_odata, int size)
{
  int i, j; 
  for (i=0; i<size; i++)
     for(j=0; j<REPEAT; j++)
        h_odata[i] = h_idata[i] * expf(h_jdata[i]);
}
