#include <stdio.h>
#include <stdlib.h>
#include "include/header.h"
#include <math.h>

#define TOLLERANZA 2.22e-16

#ifdef ELLPACK
int compare_conversion_algorithms_ellpack(int M, int N, int nz, int *I, int *J, double *val, int nthread)
{
    double **values_A = NULL;
    double **values_B = NULL;

    int **col_indices_A = NULL;
    int **col_indices_B = NULL;

    int *nz_per_row_A = NULL;
    int *nz_per_row_B = NULL;

    nz_per_row_A = coo_to_ellpack_no_zero_padding_parallel_optimization(M, N, nz, I, J, val, &values_A, &col_indices_A, nthread);
    nz_per_row_B = coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values_B, &col_indices_B, nthread);

    for (int i = 0; i < M; i++)
    {
        if (nz_per_row_A[i] != nz_per_row_B[i])
        {
            printf("Il numero di non zeri per la riga %d è diverso nelle due conversioni\n", i);
            return 1;
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < nz_per_row_A[i]; j++)
        {
            int found = 0;
            for (int k = 0; k < nz_per_row_A[i]; k++)
            {
                if (col_indices_A[i][j] == col_indices_B[i][k])
                {
                    found++;
                    if (found > 1)
                    {
                        printf("The two conversions are different (found > 1)\n");
                        return 1;
                    }
                    if (values_A[i][j] != values_B[i][k])
                    {
                        printf("The two conversions are different\n");
                        return 1;
                    }
                }
            }
            if (found == 0)
            {
                printf("The two conversions are different (found = 0)\n");
                return 1;
            }
        }
    }

    printf("Freeing memory...\n");
    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);

    free_ELLPACK_data_structures(M, values_A, col_indices_A);
    free_ELLPACK_data_structures(M, values_B, col_indices_B);

    if (nz_per_row_A != NULL)
        free(nz_per_row_A);
    if (nz_per_row_B != NULL)
        free(nz_per_row_B);

    printf("Same ELLPACK conversions\n");

    return 0;
}
#endif

#ifdef CSR
int compare_conversion_algorithms_csr(int M, int N, int nz, int *I, int *J, double *val, int nthread)
{
    double *as_A = NULL;
    double *as_B = NULL;

    int *ja_A = NULL;
    int *ja_B = NULL;

    int *irp_A = NULL;
    int *irp_B = NULL;

    int *nz_per_row_A = NULL;
    int *nz_per_row_B = NULL;

    nz_per_row_A = coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A, nthread);
    nz_per_row_B = coo_to_CSR_parallel(M, N, nz, I, J, val, &as_B, &ja_B, &irp_B, nthread);

    for (int i = 0; i < M; i++)
    {
        if (nz_per_row_A[i] != nz_per_row_B[i])
        {
            printf("Errore nella conversione numero di righe differenti\n");
            exit(1);
        }
    }

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
                        printf("The two conversions are different (found > 1)\n");
                        return 1;
                    }
                    if (as_A[j] != as_B[k])
                    {
                        printf("The two conversions are different\n");
                        return 1;
                    }
                }
            }
            if (found == 0)
            {
                printf("The two conversions are different (found = 0)\n");
                return 1;
            }
        }
    }

    printf("Freeing memory...\n");

    if (I != NULL)
        free(I);
    if (J != NULL)
        free(J);
    if (val != NULL)
        free(val);

    free_CSR_data_structures(as_A, ja_A, irp_A);
    free_CSR_data_structures(as_B, ja_B, irp_B);

    if (nz_per_row_A != NULL)
        free(nz_per_row_A);
    if (nz_per_row_B != NULL)
        free(nz_per_row_B);

    printf("Same CSR conversions\n");

    return 0;
}
#endif

#ifdef OPENMP
void check_correctness(int M, int K, double ** y_serial, double ** y_parallel)
#elif CUDA
void check_correctness(int M, int K, double ** y_serial, double * y_parallel)
#endif
{
    int flag = 1;
    double abs_err;
    double rel_err;
    double max_abs;
    for (int i = 0; i < M; i++)
    {
        for (int z = 0; z < K; z++)
        {

#ifdef CUDA
            max_abs = max(fabs(y_serial[i][z]), fabs(y_parallel[i * K + z]));
            abs_err = fabs(y_serial[i][z] - y_parallel[i * K + z]);
            rel_err = abs_err / max_abs;
#elif OPENMP
            max_abs = max(fabs(y_serial[i][z]), fabs(y_parallel[i][z]));
            abs_err = fabs(y_serial[i][z] - y_parallel[i][z]);
            rel_err = abs_err / max_abs;
#endif

            if (abs_err > TOLLERANZA || abs_err > TOLLERANZA)
            {
                flag = 0;
                break;
            }
        }

        if (!flag)
        {
            break;
        }
    }

    if (flag)
    {
        printf("I due prodotti matrice-matrice sono uguali.\n");
    }
    else
    {
        printf("I due prodotti matrice-matrice sono differenti.\n");
    }
}