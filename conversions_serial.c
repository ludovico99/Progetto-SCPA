#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "include/header.h"

int coo_to_ellpack_serial(int M, int N, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int i, j, k;
    int max_nz_per_row = 0;

    // Calcola il massimo numero di elementi non nulli in una riga
    printf("ELLPACK serial started...\n");
    for (int i = 0; i < M; i++)
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
        memory_allocation(double * , M , *values);
     
        for (int k = 0; k < M; k++)
        {   
            memory_allocation(double , max_nz_per_row , (*values)[k]);
        }
    }

    memory_allocation(int * , M , *col_indices);
  
    for (int k = 0; k < M; k++)
    {
        memory_allocation(int , max_nz_per_row ,(*col_indices)[k]);
    }

    printf("Malloc for ELLPACK data structures completed\n");

    // Riempie gli array ELLPACK con i valori e gli indici di colonna corrispondenti
    for (int i = 0; i < M; i++)
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

    printf("Freeing COO data structures...\n");
    if (I != NULL) free(I);
    if (J != NULL) free(J);
    if (val != NULL) free(val);

    return max_nz_per_row;
}

void coo_to_CSR_serial(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp)
{
    int * nz_per_row = NULL;
    int chunk_size = 0;

    printf("Starting serial CSR conversion ...\n");

    all_zeroes_memory_allocation(int, M, nz_per_row);

    printf("Counting number of non-zero entries in each row...\n");

    for (int i = 0; i < nz; i++)
    {   
        nz_per_row[I[i]]++;
    }

    printf("Allocating memory ...\n");
    // Alloca memoria per gli array CSR
    if (val != NULL)
    {   
        memory_allocation(double, nz , *as );
    }

    memory_allocation(int, nz , *ja );

    memory_allocation(int, M , *irp );
    memset(*irp, -1, sizeof(int) * M);

    printf("Filling CSR data structure ... \n");
    // Riempie gli array CSR

  
    (*irp)[0] = 0;
    
    for (int i = 0; i < M - 1; i++)
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

    (*irp)[0] = 0;
    
    if (nz_per_row != NULL) free(nz_per_row);

    printf("Freeing COO data structures...\n");
    if (I != NULL) free(I);
    if (J != NULL) free(J);
    if (val != NULL) free(val);

    printf("Completed serial CSR conversion ...\n");
}
