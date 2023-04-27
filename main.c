#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "mmio.c"

void coo_to_ellpack(int rows, int columns, int nz, int *I, int *J, double *val, double **values, int **col_indices)
{
    int i, j, k;
    int max_nz_per_row = 0;
    int max_so_far = 0;

// Calcola il massimo numero di elementi non nulli in una riga
#pragma omp parallel shared(I, J, val, max_so_far, rows, nz) firstprivate(max_nz_per_row) num_threads(10) default(none)
    {
#pragma omp for schedule(dynamic)
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
#pragma omp critical
        if (max_nz_per_row > max_so_far)
            max_so_far = max_nz_per_row;
    }
    // printf("MAX_NZ_PER_ROW is %d\n", max_so_far);

    // Alloca memoria per gli array ELLPACK
    values = (double **)malloc(rows * sizeof(double *));
    if (values == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }

#pragma omp parallel for schedule(dynamic) shared(rows, values, max_so_far) num_threads(10) default(none)
    for (int k = 0; k < rows; k++)
    {
        values[k] = (double *)malloc(max_so_far * sizeof(double));
        if (values[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    col_indices = (int **)malloc(rows * sizeof(int *));
    if (col_indices == NULL)
    {
        printf("Errore malloc\n");
        exit(1);
    }
#pragma omp parallel for schedule(dynamic) shared(rows, col_indices, max_so_far) num_threads(10) default(none)
    for (k = 0; k < rows; k++)
    {
        col_indices[k] = (int *)malloc(max_so_far * sizeof(int));
        if (col_indices[k] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
    }

    // Riempie gli array ELLPACK con i valori e gli indici di colonna corrispondenti
    // #pragma omp parallel for schedule(dynamic) shared(rows, values,col_indices max_so_far) num_threads(10) default(none)
    for (int i = 0; i < rows; i++)
    {
        int offset = 0;
        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                values[i][offset] = val[j];
                col_indices[i][offset] = J[j];
                offset++;
            }
        }
        for (k = offset; k < max_so_far; k++)
        {
            values[i][k] = 0.0;
            col_indices[i][k] = 0;
        }
    }

    // for (j = 0; j < max_so_far; j++)
    // {
    //     printf("ELLPACK VALUE: %.66lf - COL: %d\n", values[1812][j], col_indices[1812][j]);
    // }
}

void coo_to_CSR(int rows, int columns, int nz, int *I, int *J, double *val, double *as, int *ja, int **irp)
{
    int i, j, k;
    int max_nz_per_row = 0;
    int max_so_far = 0;
    int *end = NULL;
    int *curr = NULL;
    int offset = 0;

    printf("Starting CSR conversion ...\n");
    // Alloca memoria per gli array CSR
    as = (double *)malloc(nz * sizeof(double));
    if (as == NULL)
    {
        printf("Errore malloc per as\n");
        exit(1);
    }

    ja = (int *)malloc(nz * sizeof(int));
    if (ja == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    memset(ja, 0, sizeof(int)*nz);

    irp = (int **)malloc(rows * sizeof(int *));
    if (irp == NULL)
    {
        printf("Errore malloc per ja\n");
        exit(1);
    }

    // Riempie gli array CSR
    offset = 0;
    for (int i = 0; i < rows; i++)
    {
        irp[i] = &ja[offset];
        for (int j = 0; j < nz; j++)
        {
            if (I[j] == i)
            {
                as[offset] = val[j];
                ja[offset] = J[j];
                offset++;
            }
        }
    }

    printf("Completed CSR conversion ...\n");

    for (int i = 0; i <rows; i++){
        printf ("IRP VALUES: %d\n", *irp[i]);
    }

    for (int i = 0; i <nz; i++){
        printf ("JA VALUES: %d - VALUE: %lf\n", ja[i], as[i]);

    }

    // end = irp[1812];
    // curr = ja;
    // offset = 0;
    // while (curr != end)
    // {
    //     offset++;
    //     curr++;
    // }

    // while (curr != (int*)(&irp[1812] + 1))
    // {
    //     printf("VALUE: %.66lf - COL: %d\n", as[offset], *curr);
    //     curr++;
    //     offset ++;
    // }
}

double **serial_product_CSR(int M, int N, int K, int nz, double *as_A, int *ja_A, int **irp_A, double **X)
{

    double **y = NULL;
    int *curr;
    int offset = 0, row = 0;
    int *start_row = NULL;
    int *end_row = NULL;

    printf("Computing serial product ...\n");
    y = (double **)malloc(M * sizeof(double *));
    if (y == NULL)
    {
        printf("Errore malloc per y\n");
        exit(1);
    }

    for (int i = 0; i < M; i++)
    {   
        y[i] = (double *)malloc(K * sizeof(double));
        if (y[i] == NULL)
        {
            printf("Errore malloc\n");
            exit(1);
        }
        for (int z = 0; z < K; z++)
        {
            y[i][z] = 0.0;
        }
    }
    printf("y correctly allocated ... \n");
    // calcola il prodotto matrice - multi-vettore
    for (int i = 0; i < M; i++)
    {
        printf("iteration %d\n", i);
        start_row = irp_A[i];
        printf("sono qui!!");
        if (i < (M - 1))
        {
            end_row = irp_A[i + 1];

            for (int z = 0; z < K; z++)
            {
                while (start_row != end_row)
                {   
                    printf("Computing y[%d][%d]\n", i, z);
                    y[i][z] += as_A[offset] * X[*start_row][z];
                    start_row++;
                    offset++;
                }
            }
        }
        else
        {
            for (int z = 0; z < K; z++)
            {
                while (offset < nz)
                {
                    printf("Computing y[%d][%d]\n", i, z);
                    y[i][z] += as_A[offset] * X[ja_A[offset]][z];
                    start_row++;
                    offset++;
                }
            }
        }
    }
    printf("Completed serial product ...\n");
}

    double **serial_product(int M, int N, int K, double **A, double **X)
    {
        double **y = NULL;

        y = (double **)malloc(M * sizeof(double *));
        if (y == NULL)
        {
            printf("Errore malloc per y\n");
            exit(1);
        }

        for (int i = 0; i < M; i++)
        {
            y[i] = (double *)malloc(K * sizeof(double));
            if (y[i] == NULL)
            {
                printf("Errore malloc\n");
                exit(1);
            }
            for (int z = 0; z < K; z++)
            {
                y[i][z] = 0.0;
            }
        }

        // calcola il prodotto matrice - multi-vettore
        for (int i = 0; i < M; i++)
        {
            for (int z = 0; z < K; z++)
            {
                for (int j = 0; j < N; j++)
                {
                    y[i][z] += A[i][j] * X[j][z];
                }
            }
        }
    }


    void create_dense_matrix(int N, int K, double **x)
    {
            
        printf("Creating dense matrix ...\n");
        x = (double **)malloc(N * sizeof(double *));
        if (x == NULL)
        {
            printf("Errore malloc per x\n");
            exit(1);
        }

        for (int j = 0; j < N; j++)
        {
            x[j] = (double *)malloc(K * sizeof(double));
            if (x[j] == NULL)
            {
                printf("Errore malloc per x[i]\n");
                exit(1);
            }
             for (int z = 0; z < K; z++)
            {
                x[j][z] = 1.0;
            }
        }

        printf("Completed dense matrix creation...\n");


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
        double **y;

        double **values = NULL;
        int **col_indices = NULL;

        double *as_A = NULL;
        int *ja_A = NULL;
        int **irp_A = NULL;

        double **x = NULL;

        int K = 2; // It could be dynamic...

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
            printf("TOTAL_NOT_ZERO: %d\n", nz);
            I = (int *)malloc(nz * sizeof(int));
            J = (int *)malloc(nz * sizeof(int));
            val = (double *)malloc(nz * sizeof(double));
            nthreads = nz / 16;
#pragma omp parallel for schedule(static, 16) shared(nz, I, J, val, f) num_threads(nthreads) default(none)
            for (int i = 0; i < nz; i++)
            {
                fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
                I[i]--; /* adjust from 1-based to 0-based */
                J[i]--;
                // printf("%d %d %lg\n", I[i], J[i], val[i]);
            }

            // coo_to_ellpack(M, N, nz, I, J, val, values, col_indices);
            coo_to_CSR(M, N, nz, I, J, val, as_A, ja_A, irp_A);
            // Creating a dense matrix ...
            create_dense_matrix(N, K, x);

            y = serial_product_CSR(M, N, K, nz, as_A, ja_A, irp_A, x);
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
