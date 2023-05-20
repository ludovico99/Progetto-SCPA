#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include <cuda_runtime.h> // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h> // For CUDA SDK timers
#include "../header.h"

//#define get_pitch(BaseAddress, Row, Column, pitch, T)  *((T *)((char *)BaseAddress + Row * pitch) + Column)


double **parallel_product_CSR(int M, int N, int K, int nz, double *as_A, int *ja_A, int *irp_A, double **X,double * time, int nthread)
{

    double **y = NULL;
    int chunk_size;

    struct timespec start, stop;

    AUDIT printf("Computing parallel product ...\n");
    y = (double **)malloc(M * sizeof(double *));
    if (y == NULL)
    {
        printf("Errore malloc per y\n");
        exit(1);
    }

    for (int i = 0; i < M; i++)
    {
        y[i] = (double *)malloc(K * sizeof(double));
        if (y[i] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
        for (int z = 0; z < K; z++)
        {
            y[i][z] = 0.0;
        }
    }
    AUDIT printf("y correctly allocated ... \n");

    // calcola il prodotto matrice - multi-vettore
    if (clock_gettime(CLOCK_MONOTONIC, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    if (M % nthread == 0)
    {
        chunk_size = M / nthread;
    }
    else
        chunk_size = M / nthread + 1;


#pragma omp parallel for collapse(2) schedule(static, chunk_size) num_threads(nthread) shared(y, as_A, X, ja_A, irp_A, M, K, nz, nthread, chunk_size) default(none)
    for (int i = 0; i < M; i++)
    {
        // #pragma omp parallel for schedule(static, K/8) num_threads(nthread) shared(y, as_A, X, ja_A, irp_A, M, K, nz, i) default(none)
        for (int z = 0; z < K; z++)
        {   
            if (i == 0 && irp_A[i] == -1){
                AUDIT printf("Row 0 is the vector zero\n");
                y[i][z] = 0.0;
            }
            if (i > 0 && irp_A[i] == irp_A[i - 1])
            {
                AUDIT printf("Row %d is the vector zero\n", i);
                y[i][z] = 0.0;
            }
            else
            {
                //printf("Computing y[%d][%d]\n", i, z);

                // if (i < (M - 1))
                //     AUDIT printf("Riga %d, id della colonna del primo nz della riga %d e id della colonna del primo nz zero della riga successiva %d\n", i, ja_A[irp_A[i]], ja_A[irp_A[i + 1]]);
                // else
                //     AUDIT printf("Riga %d, id della colonna del primo nz della riga %d\n", i, ja_A[irp_A[i]]);

                for (int j = irp_A[i]; (i < (M - 1) && j < irp_A[i + 1]) || (i >= M - 1 && j < nz); j++)
                {
                    if (as_A != NULL)
                        y[i][z] += as_A[j] * X[ja_A[j]][z];
                    else
                        y[i][z] += 1.0 * X[ja_A[j]][z];
                }
            }
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }
    AUDIT printf("Completed parallel product ...\n");
    double accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;
    if (time != NULL) *time = accum;

    printf("ELAPSED TIME FOR PARALLEL PRODUCT OPENMP:  %lf\n", accum);

    // for (int i = 0; i < M; i++)
    // {
    //     AUDIT printf("\n");
    //     for (int z = 0; z < K; z++)
    //     {
    //         AUDIT printf("y[%d][%d] = %.66lf ", i, z, y[i][z]);
    //     }
    // }

    // AUDIT printf("\n");

    return y;
}

__global__ void CSR_kernel(const int M, const int N, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int i = tid / K;
    int z = tid % K;
    double partial_sum = 0;
    if (tid < numElements)
    {
        if (i == 0 && d_irp[i] == -1)
        {
            d_y[i*K + z]= 0.0;
        }
        if (i > 0 && d_irp[i] == d_irp[i - 1])
        {
             d_y[i*K + z] = 0.0;
        }
        else
        {   
            for (int j = d_irp[i]; (i < (M - 1) && j < d_irp[i + 1]) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL) 
                    partial_sum += d_as[j] * d_X[d_ja[j]* K + z];
                else 
                    partial_sum += 1.0 *d_X [d_ja[j]*K + z];
            }
            d_y[i*K + z] = partial_sum;
        }
       
    }
}

double *convert_2D_to_1D (int M, int K, double **X){


    double * ret = (double*)malloc(M * K * sizeof(double));

    printf("Starting 2D conversion in 1D\n");
    for (int i = 0; i < M; i ++) {
        for (int j = 0; j < K; j++){
            ret[i*K + j] = X[i][j];
        }
        free(X[i]);
    }
    if (X != NULL) free(X);
    return ret;

}

double * CSR_GPU(int M, int N, int K, int nz, double *h_as, int *h_ja, int *h_irp, double **X)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    double *h_y = NULL;
    double * h_X = NULL;
    double *d_y = NULL;

    double *d_X = NULL;

    double *d_as = NULL;
    int *d_ja = NULL;
    int *d_irp = NULL;

    float expireTimeMsec = 0.0;

    h_X = convert_2D_to_1D (M, K, X);

    h_y = (double *)malloc(M * K * sizeof(double));
    if (h_y == NULL)
    {
        printf("Errore malloc per y\n");
        exit(1);
    }

    // for (int i = 0; i < M; i++)
    // {
    //     h_y[i] = (double *)malloc(K * sizeof(double));
    //     if (h_y[i] == NULL)
    //     {
    //         printf("Errore malloc\n");
    //         exit(1);
    //     }
    // }
    printf("Allocating device variables for CPU CSR product ...\n");

    err = cudaMalloc((void **)&d_y, M * K *sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device y (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_X, N * K *sizeof(double));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device irp (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_as, nz * sizeof(double));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device as (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_ja, nz * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device ja (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_irp, M * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device irp (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // err = cudaMemset(d_as, 0.0, sizeof(double) * nz);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr,
    //             "Failed to memset as (error code %s)!\n",
    //             cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMemset(d_irp, -1, sizeof(int) * M);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr,
    //             "Failed to memset irp (error code %s)!\n",
    //             cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMemset(d_ja, 0, sizeof(int) * nz);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr,
    //             "Failed to memset ja (error code %s)!\n",
    //             cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // err = cudaMemset2D(d_X, X_pitch, 0, X_width, X_height);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr,
    //             "Failed to memset ja (error code %s)!\n",
    //             cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }
    // err = cudaMemset2D(d_y, y_pitch, 0, y_width, y_height);
    // if (err != cudaSuccess)
    // {
    //     fprintf(stderr,
    //             "Failed to memset ja (error code %s)!\n",
    //             cudaGetErrorString(err));
    //     exit(EXIT_FAILURE);
    // }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");

    err = cudaMemcpy(d_as, h_as, nz * sizeof(double), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy as from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_ja, h_ja, nz * sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy ja from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_irp, h_irp, M * sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy irp from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_X, h_X, N * K * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy matrix X from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Launch the Vector Add CUDA Kernel
    int numElements = M * K;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // START TIMER
    checkCudaErrors(cudaEventRecord(start, stream));

    CSR_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CSR kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // STOP TIMER
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));
    printf("ELAPSED TIME FOR PARALLEL PRODUCT GPU: %lf\n", expireTimeMsec/1000);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_y, d_y,M * K * sizeof(double), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_as);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device as(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_ja);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device ja(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_irp);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device irp(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_X);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix X (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_y);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix y (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    // free(h_A);
    // free(h_B);
    // free(h_C);

    printf("Completed parallel product ...\n");

    // for (int i = 0; i < M; i++)
    // {
    //     printf("\n");
    //     for (int z = 0; z < K; z++)
    //     {
    //         printf("y[%d][%d] = %.70lf\t", i, z, h_y[i*K + z]);
    //     }
    //     printf("\n");
    // }

    return h_y;
}