#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#include <cuda_runtime.h> // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h> // For CUDA SDK timers
#include "../include/header.h"

/*
 * CSR_kernel_v1 - Implementazione del prodotto tra matrice sparsa A e matrice densa X
 *
 *@M: Numero di righe
 *@K: Numero di colonne
 *@nz: Numero di elementi non zero della matrice sparsa
 *@d_as: Vettore contenente gli elementi non zero della matrice sparsa
 *@d_ja: Vettore contenente gli indici di colonna degli elementi non zero della matrice sparsa
 *@d_irp: Vettore contenente l'indice di colonna del primo non zero delle righe
 *@d_X: Matrice densa
 *@d_y: Matrice prodotto
 *@numElements: Numero di elementi della matrice prodotto Y
 *
 * Ogni thread ha il compito di computare un singolo elemento della matrice finale Y.
 * La riga dell'elemento viene computata tramite 'thread_id / K' mentre la colonna
 * tramite 'thread_id % K'. In questa versione del prodotto la somma parziale non è
 * ottimizzata poiché abbiamo un accesso alla memoria globale ogni volta che modifichiamo
 * il risultato intermedio.
 */
__global__ void CSR_kernel_v1(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    /* Identificativo del thread */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Riga dell'elemento che il thread deve computare */
    int i = tid / K;

    /* Colonna dell'elemento che il thread deve computare */
    int z = tid % K;

    if (tid < numElements)
    {
        if (i == 0 && d_irp[i] == -1)
        {
            d_y[i * K + z] = 0.0;
        }
        if (i > 0 && d_irp[i] == d_irp[i - 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {
            for (int j = d_irp[i]; (i < (M - 1) && j < d_irp[i + 1]) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL)
                    d_y[i * K + z] += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    d_y[i * K + z] += 1.0 * d_X[d_ja[j] * K + z];
            }
        }
    }
}

/*
 * CSR_kernel_v2 - Implementazione del prodotto tra matrice sparsa A e matrice densa X
 *
 *@M: Numero di righe
 *@K: Numero di colonne
 *@nz: Numero di elementi non zero della matrice sparsa
 *@d_as: Vettore contenente gli elementi non zero della matrice sparsa
 *@d_ja: Vettore contenente gli indici di colonna degli elementi non zero della matrice sparsa
 *@d_irp: Vettore contenente l'indice di colonna del primo non zero delle righe
 *@d_X: Matrice densa
 *@d_y: Matrice prodotto
 *@numElements: Numero di elementi della matrice prodotto Y
 *
 * Ogni thread ha il compito di computare un singolo elemento della matrice finale Y.
 * La riga dell'elemento viene computata tramite 'thread_id / K' mentre la colonna
 * tramite 'thread_id % K'. In questa versione del prodotto la somma parziale è
 * ottimizzata poiché evitiamo di accedere continuamente alla memoria globale durante
 * il calcolo del valore dell'elemento che il thread deve computare.
 */
__global__ void CSR_kernel_v2(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    /* Identificativo del thread */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Riga dell'elemento che il thread deve computare */
    int i = tid / K;

    /* Colonna dell'elemento che il thread deve computare */
    int z = tid % K;

    /* Risultato parziale dell'elemento della matrice Y */
    double partial_sum = 0;

    if (tid < numElements)
    {

        if (i == 0 && d_irp[i] == -1)
        {
            d_y[i * K + z] = 0.0;
        }
        else if (i > 0 && d_irp[i] == d_irp[i - 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {

            for (int j = d_irp[i]; (i < (M - 1) && j < d_irp[i + 1]) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL)
                    partial_sum += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    partial_sum += 1.0 * d_X[d_ja[j] * K + z];
            }
            d_y[i * K + z] = partial_sum;
        }
    }
}

/*
 * CSR_kernel_v3 - Implementazione del prodotto tra matrice sparsa A e matrice densa X
 *
 *@M: Numero di righe
 *@K: Numero di colonne
 *@nz: Numero di elementi non zero della matrice sparsa
 *@d_as: Vettore contenente gli elementi non zero della matrice sparsa
 *@d_ja: Vettore contenente gli indici di colonna degli elementi non zero della matrice sparsa
 *@d_irp: Vettore contenente l'indice di colonna del primo non zero delle righe
 *@d_X: Matrice densa
 *@d_y: Matrice prodotto
 *@numElements: Numero di elementi della matrice prodotto Y
 *
 * Ogni thread ha il compito di computare un singolo elemento della matrice finale Y.
 * La riga dell'elemento viene computata tramite 'thread_id / K' mentre la colonna
 * tramite 'thread_id % K'. In questa versione del prodotto si vuole ottimizzare il numero di accessi a d_irp
 * andando a memorizzarne il valore in una variabile automatica.
 */
__global__ void CSR_kernel_v3(const int M, const int K, const int nz, double *d_as, int *d_ja, int *d_irp, double *d_X, double *d_y, int numElements)
{
    /* Identificativo del thread */
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /* Riga dell'elemento che il thread deve computare */
    int i = tid / K;

    /* Colonna dell'elemento che il thread deve computare */
    int z = tid % K;

    /* Risultato parziale dell'elemento della matrice Y */
    double partial_sum = 0;

    if (tid < numElements)
    {
        int start = d_irp[i];
        int end = d_irp[i + 1];

        if (i == 0 && start == -1)
        {
            d_y[i * K + z] = 0.0;
        }
        else if (i > 0 && start == d_irp[i - 1])
        {
            d_y[i * K + z] = 0.0;
        }
        else
        {

            for (int j = start; (i < (M - 1) && j < end) || (i >= M - 1 && j < nz); j++)
            {
                if (d_as != NULL)
                    partial_sum += d_as[j] * d_X[d_ja[j] * K + z];
                else
                    partial_sum += 1.0 * d_X[d_ja[j] * K + z];
            }
            d_y[i * K + z] = partial_sum;
        }
    }
}

/*
 * Questa funzione esegue dei setup per poter lanciare
 * il kernel:
 *
 *   1. La matrice densa X viene convertita da 2D
 *      a 1D.
 *
 *   2. Viene eseguita l'allocazione di memoria per la
 *      la matrice Y.
 *
 *   3. Viene allocata la memoria su Device.
 *
 *   4. Il contenuto delle strutture dati viene copiato
 *      dall'Host verso il Device.
 *
 *   5. Viene lanciato il kernel.
 *
 */
double *CSR_GPU(int M, int N, int K, int nz, double *h_as, int *h_ja, int *h_irp, double **X, double *time)
{   
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;

    // HOST
    double *h_y = NULL;
    double *h_X = NULL;

    // DEVICE
    double *d_y = NULL;
    double *d_X = NULL;
    double *d_as = NULL;
    int *d_ja = NULL;
    int *d_irp = NULL;

    float expireTimeMsec = 0.0;

    /* Conversione matrice densa X da 2D a 1D */
    h_X = convert_2D_to_1D(M, K, X);

    /* Allocazione memoria host della matrice Y */
    memory_allocation(double, M *K, h_y);

    printf("Allocating device variables for CPU CSR product ...\n");

    /* Allocazione su Device per la matrice Y */
    memory_allocation_Cuda(double, M *K, d_y);
    /* Allocazione su Device per la matrice densa X */
    memory_allocation_Cuda(double, N *K, d_X);
    /* Allocazione su Device per il vettore as contenente gli elementi non zero */
    memory_allocation_Cuda(double, nz, d_as);
    /* Allocazione su Device per il vettore ja contenente gli indici di colonna */
    memory_allocation_Cuda(int, nz, d_ja);
    /* Allocazione su Device per il vettore irp contenente il puntatore alla entry del vettore ja */
    memory_allocation_Cuda(int, M, d_irp);

    printf("Copy input data from the host memory to the CUDA device\n");

    /* Copio il contenuto del vettore as dall'Host verso il Device */
    memcpy_to_dev(h_as, d_as, double, nz);
    /* Copio il contenuto del vettore ja dall'Host verso il Device */
    memcpy_to_dev(h_ja, d_ja, int, nz);
    /* Copio il contenuto del vettore irp dall'Host verso il Device */
    memcpy_to_dev(h_irp, d_irp, int, M);
    /* Copio la matrice densa X dall'Host verso il Device */
    memcpy_to_dev(h_X, d_X, double, N *K);

    /* Numero di elementi della matrice prodotto Y */
    int numElements = M * K;

    /* Numero di thread per blocco */
    int threadsPerBlock = 1024;

    /* Numero di blocchi per griglia */
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // START TIMER
    checkCudaErrors(cudaEventRecord(start, stream));

    /* Versione accesso alla memoria globale non ottimizzato */
    // CSR_kernel_v1<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

    /* Versione accesso alla memoria globale ottimizzato */
    CSR_kernel_v3<<<blocksPerGrid, threadsPerBlock>>>(M, K, nz, d_as, d_ja, d_irp, d_X, d_y, numElements);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CSR kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // STOP TIMER
    checkCudaErrors(cudaEventRecord(stop, stream));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&expireTimeMsec, start, stop));

    printf("ELAPSED TIME FOR PARALLEL PRODUCT GPU: %lf ns = %lf ms = %lf seconds\n", expireTimeMsec * 1e6, expireTimeMsec, expireTimeMsec * 1e-3);

    if (time != NULL)
        *time = expireTimeMsec * 1e6;
    printf("GFLOPS FOR PARALLEL PRODUCT GPU: %lf\n", compute_GFLOPS(K, nz, expireTimeMsec * 1e6));

    printf("Copy output data from the CUDA device to the host memory\n");

    /* Copio la matrice prodotto Y dal Device all'Host */
    memcpy_to_host(d_y, h_y, double, M *K);

    /* Inizio il processo di pulizia della memoria su Device */
    printf("Freeing Device memory ...\n");

    free_memory_Cuda(d_as);
    free_memory_Cuda(d_ja);
    free_memory_Cuda(d_irp);
    free_memory_Cuda(d_X);
    free_memory_Cuda(d_y);

    /* Inizio il processo di pulizia della memoria su Host */

    printf("Freeing host memory ...\n");

    free(h_X);

    printf("Completed parallel product CSR without streams...\n");

    return h_y;
}
