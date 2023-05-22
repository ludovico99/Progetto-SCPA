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

__global__ void CSR_kernel(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int i = tid / K;
    int z = tid % K;
    double partial_sum = 0;
    if (tid < numElements)
    {
        if (i == 0 && d_irp[i] == -1)
        {
            d_y[i * K + z] = 0.0;
        }
        if (i > 0 && d_irp[i] == d_irp[i - 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {
            for (int j = d_irp[i]; (i < (M - 1) && j < d_irp[i + 1]) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL)
                    partial_sum += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    partial_sum += 1.0 * d_X[d_ja[j] * K + z];
            }
            d_y[i * K + z] = partial_sum;
        }
    }
}

double *CSR_GPU(int M, int N, int K, int nz, double *h_as, int *h_ja, int *h_irp, double **X, double *time)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    double *h_y = NULL;
    double *h_X = NULL;
    double *d_y = NULL;

    double *d_X = NULL;

    double *d_as = NULL;
    int *d_ja = NULL;
    int *d_irp = NULL;

    float expireTimeMsec = 0.0;

    h_X = convert_2D_to_1D(M, K, X);

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

    err = cudaMalloc((void **)&d_y, M * K * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device y (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_X, N * K * sizeof(double));

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

    CSR_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);
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

    printf("ELAPSED TIME FOR PARALLEL PRODUCT GPU: %lf ns = %lf ms = %lf seconds\n", expireTimeMsec * 1e6, expireTimeMsec, expireTimeMsec * 1e-3);

    if (time != NULL)
        *time = expireTimeMsec * 1e6;
    printf("GFLOPS FOR PARALLEL PRODUCT GPU: %lf\n", compute_GFLOPS(K, nz, expireTimeMsec * 1e6));

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_y, d_y, M * K * sizeof(double), cudaMemcpyDeviceToHost);

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
    printf("Freeing host memory ...\n");
    free(h_X);

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

__global__ void ELLPACK_kernel(const int M, const int K, int *nz_per_row, int * sum_nz, double *d_values, int *d_col_indices, double *d_X, double *d_y, int numElements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int i = tid / K;
    int z = tid % K;
    double partial_sum = 0.0;
    int offset = sum_nz[i];
    if (tid < numElements)
    {
        if (nz_per_row[i] == 0)
            d_y[i * K + z] = 0.0;
        else
        {
            for (int j = 0; j < nz_per_row[i]; j++)
            {
                if (d_values != NULL)
                    partial_sum += d_values[i * offset + j] * d_X[d_col_indices[i * offset + j] * K + z];
                else
                    partial_sum += 1.0 * d_X[d_col_indices[i * offset + j] * K + z];
            }
            d_y[i * K + z] = partial_sum;
        }
    }
}

double *ELLPACK_GPU(int M, int N, int K, int nz, int *nz_per_row, double **values, int **col_indices, double **X, double *time)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    double *h_y = NULL;
    double *d_y = NULL;

    double *h_X = NULL;
    double *d_X = NULL;

    double *h_values = NULL;
    int *h_col_indices = NULL;

    double *d_values = NULL;
    int *d_col_indices = NULL;
    int *d_nz_per_row = NULL;

    int * h_sum_nz = NULL;
    int * d_sum_nz = NULL;

    float expireTimeMsec = 0.0;

    h_X = convert_2D_to_1D(M, K, X);

    h_y = (double *)malloc(M * K * sizeof(double));
    if (h_y == NULL)
    {
        printf("Errore malloc per y\n");
        exit(1);
    }

    h_values = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, values);
    h_col_indices = convert_2D_to_1D_per_ragged_matrix(M, nz, nz_per_row, col_indices);

    printf("Allocating device variables for CPU ELLPACK product ...\n");

    err = cudaMalloc((void **)&d_y, M * K * sizeof(double));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device y (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_X, N * K * sizeof(double));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device X (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_values, M * nz * sizeof(double));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device values (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_col_indices, M * nz * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device col_indices (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_nz_per_row, M * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device nz_per_row (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_sum_nz, M * sizeof(int));

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device sum_mz (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in device memory
    printf("Copy input data from the host memory to the CUDA device\n");

    err = cudaMemcpy(d_values, h_values, M * nz * sizeof(double), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy values from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_col_indices, h_col_indices, M * nz * sizeof(int), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy ja from host to device (error code %s)!\n",
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

    err = cudaMemcpy(d_nz_per_row, nz_per_row, M * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy nz_per_row from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    h_sum_nz = compute_sum_nz(M, nz_per_row);
    err = cudaMemcpy(d_sum_nz, h_sum_nz, M * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy device sum_nz from host to device (error code %s)!\n",
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

    ELLPACK_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, K, d_nz_per_row, d_sum_nz, d_values, d_col_indices, d_X, d_y, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch ELLPACK kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // STOP TIMER
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));

    printf("ELAPSED TIME FOR PARALLEL PRODUCT GPU: %lf ns = %lf ms = %lf seconds\n", expireTimeMsec * 1e6, expireTimeMsec, expireTimeMsec * 1e-3);

    if (time != NULL)
        *time = expireTimeMsec * 1e6;
    printf("GFLOPS FOR PARALLEL PRODUCT GPU: %lf\n", compute_GFLOPS(K, nz, expireTimeMsec * 1e6));

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_y, d_y, M * K * sizeof(double), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    err = cudaFree(d_values);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device values(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_col_indices);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device col_indices(error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_nz_per_row);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device nz_per_row(error code %s)!\n",
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

    err = cudaFree(d_sum_nz);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device sum_nz (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    printf("Freeing host memory ...\n");
    if (h_X != NULL) free(h_X);
    if (h_values != NULL) free(h_values);
    if (h_col_indices != NULL) free(h_col_indices);
    if (h_sum_nz != NULL) free(h_sum_nz);

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