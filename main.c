#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "mmio.c"



void coo_to_ellpack(int rows, int columns, int nz,int *I, int *J, double* val, double **values, int **col_indices) {
    int i, j, k;
    int max_nz_per_row = 0;

    // Calcola il massimo numero di elementi non nulli in una riga
#pragma omp parallel for schedule (dynamic) shared (I, J, val) num_threads (10) default(none)
    for (i = 0; i < rows; i++) {
        int nz_in_row = 0;
        for (j = 0; j < nz; j++) {
            if (I[j] == i) {
                nz_in_row++;
            }
        }
        if (nz_in_row > max_nz_per_row) {
            max_nz_per_row = nz_in_row;
        }
    }

    // Alloca memoria per gli array ELLPACK
    *values = (double*)malloc(columns* max_nz_per_row * sizeof(double));
    *col_indices = (int*)malloc(columns * max_nz_per_row * sizeof(int));

    // Riempie gli array ELLPACK con i valori e gli indici di colonna corrispondenti
    for (i = 0; i < n; i++) {
        int offset = 0;
        for (j = 0; j < nz; j++) {
            if (entries[j].row == i) {
                (*values)[i * max_nz_per_row + offset] = entries[j].val;
                (*col_indices)[i * max_nz_per_row + offset] = entries[j].col;
                offset++;
            }
        }
        for (k = offset; k < max_nz_per_row; k++) {
            (*values)[i * max_nz_per_row + k] = 0.0;
            (*col_indices)[i * max_nz_per_row + k] = 0;
        }
    }
}

int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int nthreads;
    int M, N, nz;
    int *I, *J;
    double *val;

    double * values = NULL;
    int * col_indices = NULL;

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
    if (mm_is_sparse(matcode))
    {
        /* find out size of sparse matrix .... */

        if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
            exit(1);

        I = (int *)malloc(nz * sizeof(int));
        J = (int *)malloc(nz * sizeof(int));
        val = (double *)malloc(nz * sizeof(double));
    nthreads = nz / 16;
    #pragma omp parallel for schedule (static, 16) num_threads(nthreads) default(none)
        for (int i = 0; i < nz; i++)
        {
            fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
            I[i]--; /* adjust from 1-based to 0-based */
            J[i]--;
            printf("%d %d %lg\n", I[i], J[i], val[i]);
        }

    coo_to_ellpack(M,N,nz,I,J,val, &values, &col_indices);


    }
    else
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(1);
    }

    if (f != stdin)
        fclose(f);

   

    // /************************/
    // /* now write out matrix */
    // /************************/

    // mm_write_banner(stdout, matcode);
    // mm_write_mtx_crd_size(stdout, M, N, nz);
    // for (i = 0; i < nz; i++)
    //     fprintf(stdout, "%d %d %20.19g\n", I[i] + 1, J[i] + 1, val[i]);

    return 0;
}
