#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#include "lib/mmio.h"
#include "include/header.h"

char *filename = NULL;
FILE *f = NULL;

/**
 * create_matrix_coo - It reads from the file and allocates the matrix in the COO format
 * @param mode: It's a code that represents a matrix type in our application
 * @param M: Number of rows
 * @param N: Number of columns
 * @param nz: Number of nz
 * @param I: Array of integers that contains the row indexes for each number not zero
 * @param J: Array of integers that contains the column indexes for each number not zero
 * @param val: Array of double containing the values for each number not zero
 */

static void create_matrix_coo(int mode, int *M, int *N, int *nz, int **I, int **J, double **val)
{
    switch (mode)
    {
    case SYMM_PATTERN:
        coo_symm(mode, M, N, nz, I, J, val);
        break;

    case SYMM_REAL:
        coo_symm(mode, M, N, nz, I, J, val);
        break;

    case GENERAL_PATTERN:
        coo_general(mode, M, N, nz, I, J, val);
        break;

    case GENERAL_REAL:
        coo_general(mode, M, N, nz, I, J, val);
        break;
    }
}

/**
 * check_matcode_error - Auxiliary function that checks if the matrix in input is supported by this application
 *
 * @param matcode: Array that represents the type of the matrix
 *
 * @returns 1 if the matrix is not supported, otherwise returns 0
 */

static int check_matcode_error(MM_typecode matcode)
{
    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (!mm_is_matrix(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return 1;
    }

    if (!mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return 1;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int nthread;
    int cmp_conversation;
    int M;
    int N;
    int nz;

    // Declaring variables for COO matrix representation
    int *I;
    int *J;
    double *val;

    double **y_serial;

    int *nz_per_row = NULL;

#ifdef ELLPACK
    // Declaring variables for ELLPACK matrix representation
    double **values = NULL;
    int **col_indices = NULL;

#elif CSR

    // Declaring variables for CSR matrix representation
    double *as = NULL;
    int *ja = NULL;
    int *irp = NULL;
#endif

#ifdef CUDA
    double *y_parallel_cuda;
#elif OPENMP
    double **y_parallel_omp;
#endif

    /*
     * The dense matrix is a two-dimensional array represented as an array of pointers to rows,
     * where each pointer points to an array of doubles representing the columns of that specific row.
     */
    double **X;
    MM_typecode matcode;


    // nthread is the number of processors available to the device.
    nthread = omp_get_num_procs();

    if (argc < 2) // The code needs at least 2 parameters: the executable and the name of the matrix
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }

    filename = argv[1];

    if ((f = fopen(filename, "r")) == NULL) // Opening the matrix as described in the .mtx file format
    {
        printf("Error opening the file\n");
        exit(1);
    }

    /*
     * Determine the type of matrix represented in the Matrix Marker file.
     * File Descriptor f is assumed to have been opened for read access.
     */

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (check_matcode_error(matcode))
        exit(1);

    /*
     * Verify that the matrix is symmetric.
     */
    if (mm_is_symmetric(matcode))
    {
        /*
         * Verify that the symmetric matrix is pattern.
         */
        if (mm_is_pattern(matcode))
        {
            create_matrix_coo(SYMM_PATTERN, &M, &N, &nz, &I, &J, &val);
        }
        /*
         * Verify that the symmetric matrix is real.
         */
        else if (mm_is_real(matcode))
        {
            create_matrix_coo(SYMM_REAL, &M, &N, &nz, &I, &J, &val);
        }
        /*
         * Otherwise the symmetric matrix is not supported by the application
         */
        else
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }
    }
    /*
     * Verify that the matrix is general.
     */
    else if (mm_is_general(matcode))
    { /*
       * Verify that the general matrix is pattern.
       */
        if (mm_is_pattern(matcode))
        {
            create_matrix_coo(GENERAL_PATTERN, &M, &N, &nz, &I, &J, &val);
        }
        /*
         * Verify that the general matrix is real.
         */
        else if (mm_is_real(matcode))
        {
            create_matrix_coo(GENERAL_REAL, &M, &N, &nz, &I, &J, &val);
        }
        /*
         * Otherwise the symmetric matrix is not supported by the application
         */
        else
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }
    }
    /*
     * Otherwise the matrix is not supported by the application
     */
    else
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

#ifdef ELLPACK

#ifdef CHECK_CONVERSION
    /**
     * Let's compare the optimized conversion for ELLPACK with the normal one. The optimized one is much faster respect to the normal one.
     */
    cmp_conversation = compare_conversion_algorithms_ellpack(M, N, nz, I, J, val, nthread);

    if (f != stdin)
        fclose(f);

    if (cmp_conversation)
        return 1;

    return 0;

#else // NOT CHECK_CONVERSION

    /* This conversion version of Ellpack stores 0x00 bytes of padding */
    // int max_nz_per_row = coo_to_ellpack_parallel(M, N, nz, I, J, val, &values, &col_indices, nthread);

    /* This conversion version of Ellpack does not store 0x00 bytes of padding */
    // nz_per_row = coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values, &col_indices, nthread);

    /* This conversion version of Ellpack optimizes the version that doesn't use padding*/
    nz_per_row = coo_to_ellpack_no_zero_padding_parallel_optimization(M, N, nz, I, J, val, &values, &col_indices, nthread);

#endif // CHECK_CONVERSION

#elif CSR

#ifdef CHECK_CONVERSION
    /**
     * Let's compare the optimized conversion for CSR with the normal one. The optimized one is much faster respect to the normal one.
     */
    cmp_conversation = compare_conversion_algorithms_csr(M, N, nz, I, J, val, nthread);

    if (f != stdin)
        fclose(f);

    if (cmp_conversation)
        return 1;

    return 0;

#else // NOT CHECK_CONVERSION

    /* Standard conversion from COO to CSR format*/
    // coo_to_CSR_parallel(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A, nthread);

    /* This conversion version of CSR optimizes the previous version*/
    nz_per_row = coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as, &ja, &irp, nthread);

#endif // CHECK_CONVERSION
#endif

#ifdef SAMPLINGS
    /**
     * Let's compute for a fixed number of samplings, the mean and the variance of the times as the number of threads and K vary.
     *  The computed stats are written in a .csv in the plots directory
     */
#ifdef OPENMP
#ifdef CSR
    computing_samplings_openMP(M, N, nz, as, ja, irp, nthread);
#elif ELLPACK
    computing_samplings_openMP(M, N, nz, nz_per_row, values, col_indices, nthread);
#endif
#endif

#ifdef CUDA
#ifdef CSR

    /* The parallel product is executed on the GPU. It first allocates memory on the GPU and then starts the CSR kernel */
    //samplings_GPU_CSR(M, N, nz, as, ja, irp, nz_per_row);
    samplings_GPU_CSR_flush_cache(M, N, nz, as, ja, irp, nz_per_row);

#elif ELLPACK

    /* The parallel product is executed on the GPU. It first allocates memory on the GPU and then starts the ELLPACK kernel */
    //samplings_GPU_ELLPACK(M, N, nz, nz_per_row, values, col_indices);
    samplings_GPU_ELLPACK_flush_cache(M, N, nz, nz_per_row, values, col_indices);

#endif
    free(y_parallel_cuda);

#endif // CUDA

    if (f != stdin)
        fclose(f);
    return 0;

#endif // SAMPLINGS

#ifdef CORRECTNESS
    /**
     * With CORRECTNESS defined, we wants to verify that the serial product is equal to the parallel one within a tolerance range.
     */
    
    int K = atoi(argv[2]);

    create_dense_matrix(N, K, &X);

#ifdef ELLPACK
    /* This version for the serial product uses ELLPACK conversion with 0x00 bytes of padding */
    // y_serial = serial_product_ellpack(M, N, K, nz, max_nz_per_row, values, col_indices, X, NULL);

    /* This version for the serial product uses the ELLPACK conversion without the 0x00 bytes of padding*/
    y_serial = serial_product_ellpack_no_zero_padding(M, N, K, nz, nz_per_row, values, col_indices, X, NULL);

#ifdef OPENMP
    /* This parallel product version uses ELLPACK conversion with 0x00 bytes of padding*/
    // y_parallel_omp = parallel_product_ellpack (M, N, K, nz, max_nz_per_row, values, col_indices, X, NULL, nthread);

    /* This version for the parallel product uses the ELLPACK conversion without the 0x00 bytes of padding*/
    y_parallel_omp = parallel_product_ellpack_no_zero_padding(M, N, K, nz, nz_per_row, values, col_indices, X, NULL, nthread);

    free_X(N, X);
    free_ELLPACK_data_structures(M, values, col_indices);

#elif CUDA
    /* The parallel product is executed on the GPU. It first allocates memory on the GPU and then starts the ELLPACK kernel */
    y_parallel_cuda = ELLPACK_GPU(M, N, K, nz, nz_per_row, values, col_indices, X);

    /*The allocated memory is freed in the previous invokation*/
#endif

    /* Freeing the array nz_per_row */
    if (nz_per_row != NULL)
        free(nz_per_row);

#elif CSR

    /* Invoking serial product for CSR format */
    y_serial = serial_product_CSR(M, N, K, nz, as, ja, irp, X, NULL);

#ifdef OPENMP
    /* Invoking parallel product for CSR format */
    y_parallel_omp = parallel_product_CSR(M, N, K, nz, as, ja, irp, X, NULL, nthread);

    free_X(N, X);

#elif CUDA

    /* The parallel product is executed on the GPU. It first allocates memory on the GPU and then starts the CSR kernel */
    y_parallel_cuda = CSR_GPU(M, N, K, nz, as, ja, irp, X, nz_per_row);

#endif // CUDA

    free_CSR_data_structures(as, ja, irp);

    /* Freeing the array nz_per_row */
    if (nz_per_row != NULL)
        free(nz_per_row);

#endif // CSR

        /**
         * Checking the results: Let's see if the computed products are equal or not.
         * For double precision, the unit roundoff is approximately 2.220446049250313e-16.
         */

#ifdef OPENMP
    check_correctness(M, K, y_serial, y_parallel_omp);
    free_y(M, y_parallel_omp);
#elif CUDA
    check_correctness(M, K, y_serial, y_parallel_cuda);
    free(y_parallel_cuda);
#endif

    free_y(M, y_serial);

#endif // CORRECTNESS

    if (f != stdin)
        fclose(f);

    return 0;
}
