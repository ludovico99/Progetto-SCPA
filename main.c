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

char *filename = NULL;
FILE *f = NULL;

/*
 * Il parametro mode identifica la tipologia di matrice
 * che si vuole utilizzare.
 */
void create_matrix_coo(int mode, int *M, int *N, int *nz, int **I, int **J, double **val, int nthread)
{
    switch (mode)
    {
    case SYMM_PATTERN:
        coo_symm(mode, M, N, nz, I, J, val, nthread);
        break;

    case SYMM_REAL:
        coo_symm(mode, M, N, nz, I, J, val, nthread);
        break;

    case GENERAL_PATTERN:
        coo_general(mode, M, N, nz, I, J, val, nthread);
        break;

    case GENERAL_REAL:
        coo_general(mode, M, N, nz, I, J, val, nthread);
        break;
    }
}

int check_matcode_error(MM_typecode matcode)
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

    AUDIT printf("Completed dense matrix creation...\n");
}

int main(int argc, char *argv[])
{
    int nthread;
    int cmp_conversation;
    int M;
    int N;
    int nz;

    // COO MATRIX
    int *I;
    int *J;
    double *val;
    double **y_serial;

#ifdef ELLPACK
    double **values;
    int **col_indices;
#endif

#ifdef CSR
    double *as;
    int *ja;
    int *irp;
#endif

#ifdef CUDA
    double *y_parallel_cuda;
#endif

#ifdef OPENMP
    double **y_parallel_omp;
#endif

    double **X;
    MM_typecode matcode;

#ifdef SAMPLINGS
    int K[] = {3, 4, 8, 12, 16, 32, 64};
#endif

    nthread = omp_get_num_procs();

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }

    filename = argv[1];

    if ((f = fopen(filename, "r")) == NULL)
    {
        printf("Error opening the file\n");
        exit(1);
    }

    /*
     * Determinare il tipo di matrice rappresentata nel file Matrix Marker.
     * Il File Descriptor f si assume essere stato aperto per l'accesso in
     * lettura.
     */
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (check_matcode_error(matcode))
        exit(1);

    /*
     * Verifico se la matrice è simmetrica.
     */
    if (mm_is_symmetric(matcode))
    {
        /*
         * Verifico se la matrice simmetrica è pattern.
         */
        if (mm_is_pattern(matcode))
        {
            create_matrix_coo(SYMM_PATTERN, &M, &N, &nz, &I, &J, &val, nthread);
        }
        else if (mm_is_real(matcode))
        {
            create_matrix_coo(SYMM_REAL, &M, &N, &nz, &I, &J, &val, nthread);
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
            create_matrix_coo(GENERAL_PATTERN, &M, &N, &nz, &I, &J, &val, nthread);
        }

        else if (mm_is_real(matcode))
        {
            create_matrix_coo(GENERAL_REAL, &M, &N, &nz, &I, &J, &val, nthread);
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

    cmp_conversation = compare_conversion_algorithms_ellpack(M, N, nz, I, J, val, nthread);

    if (f != stdin)
        fclose(f);

    if (cmp_conversation)
        return 1;

    return 0;

#else // NOT CHECK_CONVERSION

    /* Questa versione per la conversione di Ellpack memorizza i byte 0x00 di padding */
    // int max_nz_per_row = coo_to_ellpack_parallel(M, N, nz, I, J, val, &values, &col_indices, nthread);

    /* Questa versione per la conversione di Ellpack non memorizza i byte 0x00 di padding */
    // int *nz_per_row = coo_to_ellpack_no_zero_padding_parallel(M, N, nz, I, J, val, &values, &col_indices, nthread);

    /* Questa versione per la conversione di Ellpack ottimizza la versione che non usa il padding */
    int *nz_per_row = coo_to_ellpack_no_zero_padding_parallel_optimization(M, N, nz, I, J, val, &values, &col_indices, nthread);

#endif // CHECK_CONVERSION

#endif // ELLPACK

#ifdef CSR

#ifdef CHECK_CONVERSION

    cmp_conversation = compare_conversion_algorithms_csr(M, N, nz, I, J, val, nthread);

    if (f != stdin)
        fclose(f);

    if (cmp_conversation)
        return 1;

    return 0;

#else // NOT CHECK_CONVERSION

    // coo_to_CSR_parallel(M, N, nz, I, J, val, &as_A, &ja_A, &irp_A, nthread);
    coo_to_CSR_parallel_optimization(M, N, nz, I, J, val, &as, &ja, &irp, nthread);

#endif // CHECK_CONVERSION

#endif // CSR

#ifdef SAMPLINGS

#ifdef OPENMP
    #ifdef CSR
        computing_samplings_openMP(M, N, K, nz, as, ja, irp, nthread);
    #elif ELLPACK
        computing_samplings_openMP(M, N, K, nz, nz_per_row, values, col_indices, nthread);
    #endif
#endif

#ifdef CUDA
    // TODO
#endif // CUDA

    if (f != stdin)
    fclose(f);
    return 0;
    
#endif // SAMPLINGS


#ifdef CORRECTNESS

    int k = 64;
    create_dense_matrix(N, k, &X);

#ifdef ELLPACK

    y_serial = serial_product_ellpack_no_zero_padding(M, N, k, nz_per_row, values, col_indices, X, NULL);

#ifdef OPENMP

    y_parallel_omp = parallel_product_ellpack_no_zero_padding(M, N, k, nz_per_row, values, col_indices, X, NULL, nthread);

    free_X(N, X);

#endif

#ifdef CUDA
#endif

    free_ELLPACK_data_structures(M, values, col_indices);

#endif // ELLPACK

#ifdef CSR

    y_serial = serial_product_CSR(M, N, k, nz, as, ja, irp, X, NULL);

#ifdef OPENMP
    y_parallel_omp = parallel_product_CSR(M, N, k, nz, as, ja, irp, X, NULL, nthread);

    free_X(N, X);

#endif // OPENMP

#ifdef CUDA

    y_parallel_cuda = CSR_GPU(M, N, k, nz, as, ja, irp, X, NULL);

#endif // CUDA

    free_CSR_data_structures(as, ja, irp);

#endif // CSR

    for (int i = 0; i < M; i++)
    {
        for (int z = 0; z < k; z++)
        {

#ifdef CUDA
            if (y_serial[i][z] != y_parallel_cuda[i * k + z])
#endif

#ifdef OPENMP
                if (y_serial[i][z] != y_parallel_omp[i][z])
#endif
                {
                    printf("Serial result is different ...");
                    exit(0);
                }
        }
    }
    printf("Same results...\n");

    printf("Freeing matrix y...\n");
    free_y(M, y_serial);
#ifdef OPENMP
    free_y(M, y_parallel_omp);
#endif
#ifdef CUDA
    free(y_parallel_cuda);
#endif

#endif // CORRECTNESS

    if (f != stdin)
        fclose(f);

    return 0;
}
