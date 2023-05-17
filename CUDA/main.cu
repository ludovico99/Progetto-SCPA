#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h> // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h> // For CUDA SDK timers

#include "mmio.h"
#include "header.h"

#define SAMPLING_SIZE 30

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

static void create_dense_matrix(int N, int K, double ***x)
{

    AUDIT printf("Creating dense matrix ...\n");
    *x = (double **)malloc(N * sizeof(double *));
    if (*x == NULL)
    {
        printf("Errore malloc per x\n");
        exit(1);
    }

    for (int j = 0; j < N; j++)
    {
        (*x)[j] = (double *)malloc(K * sizeof(double));
        if ((*x)[j] == NULL)
        {
            printf("Errore malloc per x[j]\n");
            exit(1);
        }
        for (int z = 0; z < K; z++)
        {
            (*x)[j][z] = 1.0;
        }
    }

    AUDIT printf("Completed dense matrix creation...\n");
}

static FILE *init_stream(const char *filename)
{
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    if ((f = fopen(filename, "r")) == NULL)
        exit(1);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
        exit(1);

    return f;
}

static double calculate_mean(double x, double mean, int n)
{
    mean += (x - mean) / n;
    return mean;
}

static double calculate_variance(double x, double mean, double variance, int n)
{
    if (n == 1)
    {
        return 0.0;
    }
    double delta = x - mean;
    mean = calculate_mean(x, mean, n);
    variance += delta * (x - mean);
    return variance / (n - 1);
}

int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int *I, *J;
    double *val;
    double **y_serial;
    double **y_parallel;
    int nthread = omp_get_num_procs();

    int chunk_size = 0;

#ifdef ELLPACK
    double **values = NULL;
    int **col_indices = NULL;
#else
    double *as_A = NULL;
    int *ja_A = NULL;
    int *irp_A = NULL;
#endif

    double **X = NULL;

    int K[] = {3, 4, 8, 12, 16, 32, 64};

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }
    else
    {
        if ((f = fopen(argv[1], "r")) == NULL)
            exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (!mm_is_matrix(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }
    else if (!mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if (mm_is_symmetric(matcode))
    {
        if (mm_is_pattern(matcode))
        {
            if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
                exit(1);
            printf("PATTERN-SYMMETRIC MATRIX\n");
            printf("Initial NZ for a symmetric matrix: %d\n", nz);
            int computed_nz = nz * 2;
            int row;    // Raw index for the current nz
            int column; // Column index for the current nz
            if (nz % nthread == 0)
            {
                chunk_size = nz / nthread;
            }
            else
                chunk_size = nz / nthread + 1;

#pragma omp parallel for schedule(static, chunk_size) shared(nz, f, M, computed_nz, chunk_size) private(row, column) num_threads(nthread) default(none)
            for (int i = 0; i < nz; i++)
            {
                fscanf(f, "%d %d\n", &row, &column);
                if (row == column)
                {
#pragma omp atomic
                    computed_nz--;
                }
            }
            f = init_stream(argv[1]);

            I = (int *)malloc(computed_nz * sizeof(int));
            J = (int *)malloc(computed_nz * sizeof(int));
            val = NULL;

            int counter = 0;
#pragma omp parallel for schedule(static, chunk_size) shared(nz, I, J, f, chunk_size, computed_nz, counter) num_threads(nthread) default(none)
            for (int i = 0; i < nz; i++)
            {
                fscanf(f, "%d %d\n", &I[i], &J[i]);
                I[i]--; /* adjust from 1-based to 0-based */
                J[i]--;

                if (I[i] != J[i])
                {

                    I[nz + counter] = J[i];
                    J[nz + counter] = I[i];

#pragma omp atomic
                    counter++;
                }
            }
            if (counter != computed_nz - nz)
            {
                printf("Number of elements that are not on the diagonal is wrong\n");
                exit(1);
            }
            nz = computed_nz;
            printf("TOTAL NZ for a symmetric matrix: %d\n", nz);
        }
        else if (mm_is_real(matcode))
        {
            /* find out size of sparse matrix .... */
            if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
                exit(1);
            printf("REAL-SYMMETRIC MATRIX\n");
            printf("Initial NZ for a symmetric matrix: %d\n", nz);
            int computed_nz = nz * 2;
            int row;    // Raw index for the current nz
            int column; // Column index for the current nz
            double value;

            if (nz % nthread == 0)
            {
                chunk_size = nz / nthread;
            }
            else
                chunk_size = nz / nthread + 1;

#pragma omp parallel for schedule(static, chunk_size) shared(nz, f, computed_nz, chunk_size) private(row, column, value) num_threads(nthread) default(none)
            for (int i = 0; i < nz; i++)
            {
                fscanf(f, "%d %d %lg\n", &row, &column, &value);
                if (row == column)
                {
#pragma omp atomic
                    computed_nz--;
                }
            }

            f = init_stream(argv[1]);
            I = (int *)malloc(computed_nz * sizeof(int));
            J = (int *)malloc(computed_nz * sizeof(int));
            val = (double *)malloc(computed_nz * sizeof(double));

            int counter = 0;
#pragma omp parallel for schedule(static, chunk_size) shared(nz, I, J, val, f, chunk_size, computed_nz, counter) num_threads(nthread) default(none)
            for (int e = 0; e < nz; e++)
            {
                fscanf(f, "%d %d %lg\n", &I[e], &J[e], &val[e]);
                I[e]--; /* adjust from 1-based to 0-based */
                J[e]--;

                if (I[e] != J[e])
                {

                    I[nz + counter] = J[e];
                    J[nz + counter] = I[e];
                    val[nz + counter] = val[e];

#pragma omp atomic
                    counter++;
                }
            }

            if (counter != computed_nz - nz)
            {
                printf("Number of elements that are not on the diagonal is wrong\n");
                exit(1);
            }

            nz = computed_nz;
            printf("TOTAL NZ for a symmetric matrix: %d\n", nz);
        }
        else
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }
    }
    else if (mm_is_general(matcode))
    {
        if (mm_is_pattern(matcode))
        {
            if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
                exit(1);
            printf("PATTERN-GENERAL MATRIX\n");
            printf("total not zero: %d\n", nz);
            I = (int *)malloc(nz * sizeof(int));
            J = (int *)malloc(nz * sizeof(int));
            val = NULL;
            if (nz % nthread == 0)
            {
                chunk_size = nz / nthread;
            }
            else
                chunk_size = nz / nthread + 1;

#pragma omp parallel for schedule(static, chunk_size) shared(nz, I, J, f, chunk_size) num_threads(nthread) default(none)
            for (int i = 0; i < nz; i++)
            {
                fscanf(f, "%d %d\n", &I[i], &J[i]);
                I[i]--; /* adjust from 1-based to 0-based */
                J[i]--;
            }
        }

        else if (mm_is_real(matcode))
        {
            /* find out size of sparse matrix .... */
            if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
                exit(1);
            printf("REAL-GENERAL MATRIX\n");
            printf("total not zero: %d\n", nz);
            I = (int *)malloc(nz * sizeof(int));
            J = (int *)malloc(nz * sizeof(int));
            val = (double *)malloc(nz * sizeof(double));

            if (nz % nthread == 0)
            {
                chunk_size = nz / nthread;
            }
            else
                chunk_size = nz / nthread + 1;
#pragma omp parallel for schedule(static, chunk_size) shared(nz, I, J, val, f, chunk_size) num_threads(nthread) default(none)
            for (int i = 0; i < nz; i++)
            {
                fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
                I[i]--; /* adjust from 1-based to 0-based */
                J[i]--;
            }
        }
        else
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }
    }

    else
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

#ifdef ELLPACK
#ifdef CHECK_CONVERSION
    double **values_B = NULL;
    int **col_indices_B = NULL;

    coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values_B, &col_indices_B);
    int *nz_per_row = coo_to_ellpack_no_zero_padding_parallel_optimization(M, N, nz, I, J, val, &values, &col_indices);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < nz_per_row[i]; j++)
        {
            int found = 0;
            for (int k = 0; k < nz_per_row[i]; k++)
            {
                if (col_indices[i][j] == col_indices_B[i][k])
                {
                    found++;
                    if (found > 1)
                    {
                        printf("The two conversions are different\n");
                        exit(1);
                    }
                    if (values[i][j] != values_B[i][k])
                    {
                        printf("The two conversions are different\n");
                        exit(1);
                    }
                }
            }
            if (found == 0)
            {
                printf("The two conversions are different\n");
                exit(1);
            }
        }
    }

    printf("Same ELLPACK conversions\n");
    if (f != stdin)
        fclose(f);
    return 0;
#else
    // int max_nz_per_row  = coo_to_ellpack_serial(M, N, nz, I, J, val, &values, &col_indices);
    // int max_nz_per_row = coo_to_ellpack_parallel(M, N, nz, I, J, val, &values, &col_indices);
    int *nz_per_row = coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values, &col_indices);
    // int *nz_per_row = coo_to_ellpack_no_zero_padding_parallel_optimization(M, N, nz, I, J, val, &values, &col_indices);
#endif

#else

#ifdef CHECK_CONVERSION
    double *as_B = NULL;
    int *ja_B = NULL;
    int *irp_B = NULL;

    coo_to_CSR_parallel(M, N, nz, I, J, val, &as_B, &ja_B, &irp_B);
    coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);

    for (int i = 0; i < M; i++)
    {
        for (int j = irp_A[i]; (i < (M - 1) && j < irp_A[i + 1]) || (i >= M - 1 && j < nz); j++)
        {
            int found = 0;
            for (int k = irp_B[i]; (i < (M - 1) && k < irp_B[i + 1]) || (i >= M - 1 && k < nz); k++)
            {
                if (ja_A[j] == ja_B[k])
                {
                    found++;
                    if (found > 1)
                    {
                        printf("The two conversions are different\n");
                        exit(1);
                    }
                    if (as_A[j] != as_B[k])
                    {
                        printf("The two conversions are different\n");
                        exit(1);
                    }
                }
            }
            if (found == 0)
            {
                printf("The two conversions are different\n");
                exit(1);
            }
        }
    }

    printf("Same CSR conversions\n");
    if (f != stdin)
        fclose(f);
    return 0;
#else
    // coo_to_CSR_parallel(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);
    // coo_to_CSR_serial(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);
    coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);
#endif

#endif

#ifdef CORRECTNESS
    int k = 16;
    create_dense_matrix(N, k, &X);
#ifdef ELLPACK

    // y_serial = serial_product_ellpack_no_zero_padding(M, N, k, nz_per_row, values, col_indices, X, NULL);

    // y_parallel = parallel_product_ellpack_no_zero_padding(M, N, k, nz_per_row, values, col_indices, X, NULL, nthread);

#else

    // y_serial = serial_product_CSR(M, N, k, nz, as_A, ja_A, irp_A, X, NULL);

    // y_parallel = parallel_product_CSR(M, N, k, nz, as_A, ja_A, irp_A, X, NULL, nthread);

#endif
#endif

    if (f != stdin)
        fclose(f);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = 1.0;
        h_B[i] = 1.0;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector B from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector C from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
