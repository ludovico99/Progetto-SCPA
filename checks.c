#include <stdio.h>
#include <stdlib.h>
#include "header.h"

#ifdef ELLPACK
int compare_conversion_algorithms_ellpack(int M, int N, int nz, int *I, int *J, double *val)
{
    double **values_A = NULL;
    double **values_B = NULL;

    int **col_indices_A = NULL;
    int **col_indices_B = NULL;

    int *nz_per_row_A = NULL;
    int *nz_per_row_B = NULL;

    nz_per_row_A = coo_to_ellpack_no_zero_padding_parallel_optimization(M, N, nz, I, J, val, &values_A, &col_indices_A);
    nz_per_row_B = coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values_B, &col_indices_B);

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

    printf("Same ELLPACK conversions\n");

    return 0;
}
#endif

#ifdef CSR
int compare_conversion_algorithms_csr(int M, int N, int nz, int *I, int *J, double *val)
{
    double *as_A = NULL;
    double *as_B = NULL;

    int *ja_A = NULL;
    int *ja_B = NULL;

    int *irp_A = NULL;
    int *irp_B = NULL;

    int *nz_per_row_A = NULL;
    int *nz_per_row_B = NULL;

    nz_per_row_A = coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);
    nz_per_row_B = coo_to_CSR_parallel(M, N, nz, I, J, val, &as_B, &ja_B, &irp_B);

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

    printf("Same CSR conversions\n");

    return 0;
}
#endif