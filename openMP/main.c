#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>

#include "mmio.h"
#include "header.h"


#define SAMPLING_SIZE 10

void create_dense_matrix(int N, int K, double ***x)
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

    AUDIT  printf("Completed dense matrix creation...\n");
}

FILE *init_stream(const char *filename)
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

    int K[] = {3, 4, 8, 12, 16, 32, 64}; // It could be dynamic...

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
        // int max_nz_per_row  = coo_to_ellpack_serial(M, N, nz, I, J, val, &values, &col_indices);
        // int max_nz_per_row = coo_to_ellpack_parallel(M, N, nz, I, J, val, &values, &col_indices);
        int *nz_per_row = coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values, &col_indices);
        
    #else
        //coo_to_CSR_serial(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);
        coo_to_CSR_parallel(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A);
    #endif
    

    

    #ifdef BOTH
        double mean;
        double time;
        const char *filename; 

        FILE *f_samplings_serial;
        FILE *f_samplings_parallel;
        #ifdef ELLPACK
            filename = "samplings_parallel_ELLPACK.csv";
            f_samplings_parallel = fopen(filename, "w+");
            fprintf(f_samplings_parallel, "K,num_threads,mean\n");

            filename = "samplings_serial_ELLPACK.csv";
            f_samplings_serial = fopen(filename, "w+");
            fprintf(f_samplings_serial, "K,mean\n");
        #else
            filename = "samplings_parallel_CSR.csv";
            f_samplings_parallel = fopen(filename, "w+");
            fprintf(f_samplings_parallel, "K,num_threads,mean\n");

            filename = "samplings_serial_CSR.csv";
            f_samplings_serial = fopen(filename, "w+");
            fprintf(f_samplings_serial, "K,mean\n");
        #endif

    #else

        double mean;
        double time;
        const char *filename; 
        FILE *f_samplings;
        #ifdef PARALLEL_SAMPLING
            #ifdef ELLPACK
                filename = "samplings_parallel_ELLPACK.csv";
                f_samplings = fopen(filename, "w+");
                fprintf(f_samplings, "K,num_threads,mean\n");
            #else
                filename = "samplings_parallel_CSR.csv";
                f_samplings = fopen(filename, "w+");
                fprintf(f_samplings, "K,num_threads,mean\n");
            #endif
        #else
            #ifdef ELLPACK
                filename = "samplings_serial_ELLPACK.csv";
                f_samplings = fopen(filename, "w+");
                fprintf(f_samplings, "K,mean\n");
            #else
                filename = "samplings_serial_CSR.csv";
                f_samplings = fopen(filename, "w+");
                fprintf(f_samplings, "K,mean\n");
            #endif
        #endif
    #endif
    
    #ifndef BOTH 
        for (int k = 0; k < 7; k++)
        {
            create_dense_matrix(N, K[k], &X);
            #ifdef PARALLEL_SAMPLING
            for (int num_thread = 1; num_thread <= nthread; num_thread++){
            #endif
                mean = 0;

                for (int curr_samp = 0; curr_samp < SAMPLING_SIZE; curr_samp++)
                {   
                    #ifdef ELLPACK
                        #ifdef PARALLEL_SAMPLING
                            y_parallel = parallel_product_ellpack_no_zero_padding(M, N, K[k], nz_per_row, values, col_indices, X, &time, num_thread);
                            // y_parallel = parallel_product_ellpack(M, N, K, max_nz_per_row, values, col_indices, X, &time, num_thread)
                        #else
                            y_serial = serial_product_ellpack_no_zero_padding(M, N, K[k], nz_per_row, values, col_indices, X,  &time);
                            // y_serial = serial_product_ellpack(M, N, K, max_nz_per_row, values, col_indices, X, &time);
                        #endif
                        
                    #else
                        #ifdef PARALLEL_SAMPLING
                            y_parallel = parallel_product_CSR(M, N, K[k], nz, as_A, ja_A, irp_A, X, &time, num_thread); 
                        #else
                            y_serial = serial_product_CSR(M, N, K[k], nz, as_A, ja_A, irp_A, X, &time);
                        #endif
                    #endif

                    mean += time;
                }

                mean = mean / SAMPLING_SIZE;
                
            #ifdef PARALLEL_SAMPLING
                printf("MEAN for K %d, num_thread %d is %lf\n", K[k], num_thread, mean);
                fprintf(f_samplings, "%d,%d,%lf\n", K[k], num_thread, mean);
                fflush(f_samplings);
                }
            #else 
                printf("MEAN for K %d is %lf\n", K[k], mean);
                fprintf(f_samplings, "%d,%lf\n", K[k], mean);
                fflush(f_samplings);
            #endif

            for (int i = 0; i < N; i++)
            {
                free(X[i]);
            }
            if (X != NULL)
                free(X);
        }
    #else 
    
        for (int k = 0; k < 7; k++)
        {
            create_dense_matrix(N, K[k], &X);
            for (int num_thread = 1; num_thread <= nthread; num_thread++){
                mean = 0;

                for (int curr_samp = 0; curr_samp < SAMPLING_SIZE; curr_samp++)
                {   
                    #ifdef ELLPACK
                        y_parallel = parallel_product_ellpack_no_zero_padding(M, N, K[k], nz_per_row, values, col_indices, X, &time, num_thread);
                        // y_parallel = parallel_product_ellpack(M, N, K[k], max_nz_per_row, values, col_indices, X, &time, num_thread)
                    #else
                        y_parallel = parallel_product_CSR(M, N, K[k], nz, as_A, ja_A, irp_A, X, &time, num_thread); 
                    #endif

                    mean += time;
                }

                mean = mean / SAMPLING_SIZE;
                printf("MEAN for K %d, num_thread %d is %lf\n", K[k], num_thread, mean);
                fprintf(f_samplings_parallel, "%d,%d,%lf\n", K[k], num_thread, mean);
                fflush(f_samplings_parallel);
            }

            for (int i = 0; i < N; i++)
            {
                free(X[i]);
            }
            if (X != NULL)
                free(X);
        }

        for (int k = 0; k < 7; k++)
        {
            create_dense_matrix(N, K[k], &X);
            
            mean = 0;

            for (int curr_samp = 0; curr_samp < SAMPLING_SIZE; curr_samp++)
            {   
                    #ifdef ELLPACK
                        y_serial = serial_product_ellpack_no_zero_padding(M, N, K[k], nz_per_row, values, col_indices, X,  &time);
                        // y_serial = serial_product_ellpack(M, N, K[k], max_nz_per_row, values, col_indices, X, &time);
                    #else
                        y_serial = serial_product_CSR(M, N, K[k], nz, as_A, ja_A, irp_A, X, &time);
                    #endif

                    mean += time;
                }

                mean = mean / SAMPLING_SIZE;
                printf("MEAN for K %d is %lf\n", K[k], mean);
                fprintf(f_samplings_serial, "%d,%lf\n", K[k], mean);
                fflush(f_samplings_serial);
            

            for (int i = 0; i < N; i++)
            {
                free(X[i]);
            }
            if (X != NULL)
                free(X);
        }
    #endif

    #ifndef BOTH
        fclose(f_samplings);
    #else 
        fclose(f_samplings_serial);
        fclose(f_samplings_parallel);
    #endif

    /*
        for (int i = 0; i < M; i++)
        {
            for (int z = 0; z < K; z++)
            {
                if (y_serial[i][z] != y_parallel[i][z])
                {
                    printf("Serial result is different ...");
                    exit(0);
                }
            }
        }
        printf("Same results...\n");*/

    if (f != stdin)
        fclose(f);
    return 0;
}
