#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>

#include "lib/mmio.h"
#include "include/header.h"

static FILE *init_stream(void)
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

void coo_general(int mode, int *M, int *N, int *nz, int **I, int **J, double **val, int nthread)
{
    int ret_code;
    int chunk_size;
    double value;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
    {
        printf("Error reading size file...\n");
        exit(1);
    }

    if (mode == GENERAL_PATTERN)
        printf("GENERAL-PATTERN MATRIX\n");
    else
        printf("GENERAL-REAL MATRIX\n");

    printf("total not zero: %d\n", *nz);

    all_zeroes_memory_allocation(int, *nz, *I);
    all_zeroes_memory_allocation(int, *nz, *J);


    if (mode == GENERAL_PATTERN)
        *val = NULL;
    else
    {   
        memory_allocation(double, *nz, *val);
    }

    chunk_size = compute_chunk_size(*nz, nthread);

    printf("Chunk size computed: %d\n", chunk_size);

    if (mode == GENERAL_PATTERN)
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d\n", &((*I)[i]), &((*J)[i]));
            if (ret_code != 2)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;
        }
    }
    else
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d %lg\n", &((*I)[i]), &((*J)[i]), &((*val)[i]));
            if (ret_code != 3)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;
        }
    }

    printf("COO Matrix Conversion completed\n");
}

/* La variabile type permette di discriminare real e pattern */
void coo_symm(int mode, int *M, int *N, int *nz, int **I, int **J, double **val, int nthread)
{
    int ret_code;
    int computed_nz;
    int row;    // Row index for the current nz
    int column; // Column index for the current nz
    int chunk_size;
    double value;

    /*
     * Leggo le informazioni sulle dimensioni di una matrice sparsa
     *(formato coordinate) in un file Matrix Market.
     */
    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
    {
        printf("Reading Matrix Dimensions error\n");
        exit(1);
    }

    if (mode == SYMM_PATTERN)
        printf("SYMMETRIC-PATTERN MATRIX\n");

    else
        printf("SYMMETRIC-REAL MATRIX\n");

    printf("Initial nz for a symmetric matrix: %d\n", *nz);

    computed_nz = *nz * 2;

    chunk_size = compute_chunk_size(*nz, nthread);

    /*
     * Sottraggo al doppio del numero di non zeri (diagonale
     * triangolo inferiore) che sono stati letti dal file il
     * numero di non zeri che sono presenti sulla diagonale.
     */

    if (mode == SYMM_PATTERN)
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d\n", &row, &column);
            if (ret_code != 2)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            if (row == column)
            {
                computed_nz--;
            }
        }
    }
    else
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d %lg\n", &row, &column, &value);
            if (ret_code != 3)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            if (row == column)
            {
                computed_nz--;
            }
        }
    }

    if (f != NULL)
        fclose(f);

    f = init_stream();

    all_zeroes_memory_allocation(int, *nz, *I);
    all_zeroes_memory_allocation(int, *nz, *J);

    if (mode == SYMM_PATTERN)
        *val = NULL;
    else
    {
        memory_allocation(double, *nz, *val);
    }

    int counter = 0;

    if (mode == SYMM_PATTERN)
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d\n", &((*I)[i]), &((*J)[i]));
            if (ret_code != 2)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;

            if ((*I)[i] != (*J)[i])
            {

                (*I)[*nz + counter] = (*J)[i];
                (*J)[*nz + counter] = (*I)[i];
                counter++;
            }
        }
    }
    else
    {
        for (int i = 0; i < *nz; i++)
        {
            ret_code = fscanf(f, "%d %d %lg\n", &((*I)[i]), &((*J)[i]), &((*val)[i]));
            if (ret_code != 3)
            {
                printf("An error has occured reading the file ...");
                exit(1);
            }
            (*I)[i]--; /* adjust from 1-based to 0-based */
            (*J)[i]--;

            if ((*I)[i] != (*J)[i])
            {

                (*I)[*nz + counter] = (*J)[i];
                (*J)[*nz + counter] = (*I)[i];
                (*val)[*nz + counter] = (*val)[i];
                counter++;
            }
        }
    }

    if (counter != computed_nz - *nz)
    {
        printf("Number of elements that are not on the diagonal is wrong\n");
        exit(1);
    }

    *nz = computed_nz;
    printf("TOTAL NZ for a symmetric matrix: %d\n", *nz);
}