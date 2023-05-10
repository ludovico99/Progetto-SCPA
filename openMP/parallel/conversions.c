#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "../header.h"

int coo_to_ellpack_parallel(int rows, int columns, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int max_nz_per_row = 0;
    int max_so_far = 0;
    int nthreads;
    int nz_in_row;
    int offset;
    int chunk_size;

    nthreads = omp_get_num_procs();

    printf("ELLPACK parallel started...\n");

    if (rows % nthreads == 0)
    {
        chunk_size = rows / nthreads;
    }
    else
    {
        chunk_size = (rows / nthreads) + 1;
    }

#pragma omp parallel shared(I, J, val, max_so_far, rows, nz, chunk_size) firstprivate(max_nz_per_row) private(nz_in_row) num_threads(nthreads) default(none)
    {
// Calcola il massimo numero di elementi non nulli in una riga
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < rows; i++)
        {
            nz_in_row = 0;
            for (int j = 0; j < nz; j++)
            {
                if (I[j] == i)
                    nz_in_row++;
            }
            if (nz_in_row > max_nz_per_row)
                max_nz_per_row = nz_in_row;
        }

#pragma omp critical
        if (max_nz_per_row > max_so_far)
            max_so_far = max_nz_per_row;
    }
    printf("MAX_NZ_PER_ROW is %d\n", max_so_far);
    // Alloca memoria per gli array ELLPACK
    if (val != NULL)
    {
        (*values) = (double **)malloc(rows * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }

        for (int k = 0; k < rows; k++)
        {
            (*values)[k] = (double *)malloc(max_so_far * sizeof(double));
            if ((*values)[k] == NULL)
            {
                printf("Errore malloc\n");
                exit(1);
            }
        }
    }

    *col_indices = (int **)malloc(rows * sizeof(int *));
    if (*col_indices == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }
    for (int k = 0; k < rows; k++)
    {

        (*col_indices)[k] = (int *)malloc(max_so_far * sizeof(int));
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");
    fflush(stdout);
    // Riempie gli array ELLPACK con i valori e gli indici di colonna corrispondenti
    int counter = 0;
#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, rows, values, col_indices, I, val, J, nz, max_so_far, counter) num_threads(nthreads) private(offset) default(none)
    for (int i = 0; i < rows; i++)
    {

        offset = 0;
        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                if (val != NULL)
                    (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++;
            }

            if (offset > max_so_far)
            {
                printf("offset maggiore di max_so_far");
                exit(1);
            }
        }

        for (int k = offset; k < max_so_far; k++)
        {
            if (val != NULL)
                (*values)[i][k] = 0.0;
            if (offset != 0)
                (*col_indices)[i][k] = (*col_indices)[i][offset - 1];
            else
                (*col_indices)[i][k] = -1;
        }
#pragma omp atomic
        counter++;
        printf("%d rows completed\n", counter);
    }

    printf("ELLPACK parallel completed...\n");

    // for (int j = 0; j < rows; j++)
    // {
    //     for(int k = 0; k < max_so_far; k++)
    //     {
    //        AUDIT printf("ELLPACK VALUE: %.66lf - COL: %d\n", (*values)[j][k], (*col_indices)[j][k]);
    //     }
    // }

    return max_so_far;
}

void coo_to_CSR_parallel(int M, int N, int nz, int *I, int *J, double *val, double **as, int **ja, int **irp)
{
    int * nz_per_row = NULL;
    int nthread = omp_get_num_procs();
    int chunk_size = 0;

    printf("Starting parallel CSR conversion ...\n");
    nz_per_row = (int *)calloc(M, sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc per nz_per_row\n");
        exit(1);
    }

    printf("Counting number of non-zero entries in each row...\n");

    if (nz % nthread == 0)
    {
        chunk_size = nz / nthread;
    }
    else
        chunk_size = nz / nthread + 1;

#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, I, nz, nz_per_row,stdout) default(none)

    for (int i = 0; i < nz; i++)
    {   
#pragma omp atomic 
        nz_per_row[I[i]] += 1;
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

    (*irp)[0] = 0;
    for (int i = 0; i < M - 1; i++)
    {    
        (*irp)[i + 1] = (*irp)[i] + nz_per_row[i];
    }

    free(nz_per_row);

    int offset = 0;
    int row;
    int idx;
#pragma omp parallel for schedule(static, chunk_size) num_threads(nthread) shared(chunk_size, nz, irp, ja, as, val, offset, I, J) private(row, idx) default(none)

    for (int i = 0; i < nz; i++)
    {   
        row = I[i];
        idx = (*irp)[row];
        
        (*ja)[idx] = J[i];
        if (val != NULL)(*as)[idx] = val[i];
        
        #pragma omp atomic
            (*irp)[row]++;
    }
    

     // Reset row pointers
    for (int i = M - 1; i > 0; i--) {
        (*irp)[i] = (*irp)[i-1];
    }
    (*irp)[0] = 0;

    printf("Completed parallel CSR conversion ...\n");
}

int *coo_to_ellpack_no_zero_padding_parallel(int rows, int columns, int nz, int *I, int *J, double *val, double ***values, int ***col_indices)
{
    int nthreads;
    int nz_in_row;
    int offset;
    int chunk_size;
    int *nz_per_row = NULL;
    nthreads = omp_get_num_procs();

    printf("ELLPACK parallel started...\n");

    if (rows % nthreads == 0)
    {
        chunk_size = rows / nthreads;
    }
    else
    {
        chunk_size = (rows / nthreads) + 1;
    }

    nz_per_row = (int *)malloc(rows * sizeof(int));

    if (nz_per_row == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }

#pragma omp parallel shared(I, J, val, rows, nz, chunk_size, nz_per_row) private(nz_in_row) num_threads(nthreads) default(none)
    {
// Calcola il massimo numero di elementi non nulli in una riga
#pragma omp for schedule(static, chunk_size)
        for (int i = 0; i < rows; i++)
        {
            nz_in_row = 0;
            for (int j = 0; j < nz; j++)
            {
                if (I[j] == i)
                    nz_in_row++;
            }
            nz_per_row[i] = nz_in_row;
        }
    }
    printf("Number of zeroes in raws computed\n");
    // Alloca memoria per gli array ELLPACK
    if (val != NULL)
    {
        (*values) = (double **)malloc(rows * sizeof(double *));
        if (*values == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }

        for (int k = 0; k < rows; k++)
        {

            if (nz_per_row[k] != 0)
                (*values)[k] = (double *)malloc(nz_per_row[k] * sizeof(double));
            else
                (*values)[k] = NULL;
            if ((*values)[k] == NULL)
            {
                printf("Errore malloc\n");
                exit(1);
            }
        }
    }

    *col_indices = (int **)malloc(rows * sizeof(int *));
    if (*col_indices == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }
    for (int k = 0; k < rows; k++)
    {

        if (nz_per_row[k] != 0)
            (*col_indices)[k] = (int *)malloc(nz_per_row[k] * sizeof(int));
        else
            (*col_indices)[k] = NULL;
        if ((*col_indices)[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    printf("Malloc for ELLPACK data structures completed\n");
    fflush(stdout);

#pragma omp parallel for schedule(static, chunk_size) shared(chunk_size, rows, values, col_indices, I, val, J, nz, nz_per_row) num_threads(nthreads) private(offset) default(none)
    for (int i = 0; i < rows; i++)
    {

        offset = 0;
        if (nz_per_row[i] == 0)
            continue;

        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                if (val != NULL)
                    (*values)[i][offset] = val[j];
                (*col_indices)[i][offset] = J[j];
                offset++;
            }
        }
        if (offset != nz_per_row[i])
        {
            printf("offset maggiore di nz_per_row");
            exit(1);
        }
    }

    printf("ELLPACK parallel completed...\n");

    // for (int j = 0; j < rows; j++)
    // {
    //     for(int k = 0; k < max_so_far; k++)
    //     {
    //        AUDIT printf("ELLPACK VALUE: %.66lf - COL: %d\n", (*values)[j][k], (*col_indices)[j][k]);
    //     }
    // }

    return nz_per_row;
}