#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#include "mmio.h"
#include "header.h"

#define SAMPLING_SIZE 30

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
    double **y_openMP;
    double *y_cuda;
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
    coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);
#endif

#endif

#ifdef CORRECTNESS
    int k = 64;
    create_dense_matrix(N, k, &X);
#ifdef ELLPACK

    // y_serial = serial_product_ellpack_no_zero_padding(M, N, k, nz_per_row, values, col_indices);

    // y_parallel = parallel_product_ellpack_no_zero_padding(M, N, k, nz_per_row, values, col_indices, X);

#else

    // y_serial = serial_product_CSR(M, N, k, nz, as_A, ja_A, irp_A, X, NULL);
    y_openMP = parallel_product_CSR(M, N, k, nz, as_A, ja_A, irp_A, X, NULL, nthread);
    y_cuda = CSR_GPU(M, N, k, nz, as_A, ja_A, irp_A, X);

#endif
#endif

#ifdef CORRECTNESS

    for (int i = 0; i < M; i++)
    {
        for (int z = 0; z < k; z++)
        {
            if (y_openMP[i][z] != y_cuda[i*k + z])
            {
                printf("Serial result is different ...");
                exit(0);
            }
        }
    }
    printf("Same results...\n");
#endif
    if (f != stdin)
        fclose(f);

    return 0;
}
