#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "header.h"

int coo_to_ellpack_serial(int rows, int columns, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int i, j, k;
    int max_nz_per_row = 0;

    // Calcola il massimo numero di elementi non nulli in una riga
    printf("ELLPACK serial started...\n");
    for (int i = 0; i < rows; i++)
    {
        int nz_in_row = 0;
        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
                nz_in_row++;
        }
        if (nz_in_row > max_nz_per_row)
            max_nz_per_row = nz_in_row;
    }

    printf("MAX_NZ_PER_ROW is %d\n", max_nz_per_row);

    // Alloca memoria per gli array ELLPACK
    if (val != NULL)
    {
        *values = (double **)malloc(rows * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }

        for (int k = 0; k < rows; k++)
        {
            (*values)[k] = (double *)malloc(max_nz_per_row * sizeof(double));
            if ((*values)[k] == NULL)
            {
                printf("Errore malloc\n");
                exit(1);
            }
        }
    }

    (*col_indices) = (int **)malloc(rows * sizeof(int *));
    if ((*col_indices) == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }
    for (k = 0; k < rows; k++)
    {
        (*col_indices)[k] = (int *)malloc(max_nz_per_row * sizeof(int));
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");

    // Riempie gli array ELLPACK con i valori e gli indici di colonna corrispondenti
    for (int i = 0; i < rows; i++)
    {
        int offset = 0;
        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                if (val != NULL) (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++;
            }
        }

        for (k = offset; k < max_nz_per_row; k++)
        {
            if (val != NULL) (*values)[i][k] = 0.0;
            if (offset != 0)
                (*col_indices)[i][k] = (*col_indices)[i][offset - 1];
            else
                (*col_indices)[i][k] = -1;
        }
        printf("row %d completed\n", i);
    }

    //   for (int j = 0; j < rows; j++)
    // {
    //     for(int k = 0; k < max_nz_per_row; k++)
    //     {
    //         AUDIT printf("ELLPACK VALUE: %.66lf - COL: %d\n", (*values)[j][k], (*col_indices)[j][k]);
    //     }
    // }

    return max_nz_per_row;
}

void coo_to_CSR_serial(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp)
{
    int * nz_per_row = NULL;
    int chunk_size = 0;

    printf("Starting serial CSR conversion ...\n");
    nz_per_row = (int *)calloc(M, sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc per nz_per_row\n");
        exit(1);
    }

    printf("Counting number of non-zero entries in each row...\n");

    for (int i = 0; i < nz; i++)
    {   
        nz_per_row[I[i]]++;
    }

    printf("Allocating memory ...\n");
    // Alloca memoria per gli array CSR
    if (val != NULL)
    {
        *as = (double *)malloc(nz * sizeof(double));
        if (*as == NULL)
        {
            printf("Errore malloc per as\n");
            exit(1);
        }
    }

    *ja = (int *)malloc(nz * sizeof(int));
    if (*ja == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    *irp = (int *)malloc(M * sizeof(int));
    if (*irp == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    memset(*irp, -1, sizeof(int) * M);

    printf("Filling CSR data structure ... \n");
    // Riempie gli array CSR

    if (nz_per_row[0] == 0) {
        (*irp)[0] = -1;
        (*irp)[1] = 0;
    }
    else  {
        (*irp)[0] = 0;
        (*irp)[1] = (*irp)[0] + nz_per_row[0]; 
    }
    for (int i = 1; i < M - 1; i++)
    {    
        (*irp)[i + 1] = (*irp)[i] + nz_per_row[i];
    }

    int offset = 0;
    int row;
    int idx;

    for (int i = 0; i < nz; i++)
    {   
        row = I[i];
        idx = (*irp)[row];
        
        (*ja)[idx] = J[i];
        if (val != NULL)(*as)[idx] = val[i];
        
        (*irp)[row]++;
    }
    

     // Reset row pointers
    for (int i = M - 1; i > 0; i--) {
        (*irp)[i] = (*irp)[i-1];
    }
    
    if (nz_per_row[0] == 0) {
        (*irp)[0] = -1;
        (*irp)[1] = 0;
    }
    else 
        (*irp)[0] = 0;
    
    free(nz_per_row);

    printf("Completed serial CSR conversion ...\n");
}
