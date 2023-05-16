#include <stdio.h>

#define N 4
#define M 4

__global__ void matrix_vector_product(int *A, int *x, int *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = 0;
        for (int j = 0; j < M; j++) {
            y[i] += A[i * M + j] * x[j];
        }
    }
}

int main() {
    int A[N][M], x[M], y[N];
    int *d_A, *d_x, *d_y;

    // Inizializza la matrice A e il vettore x
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i][j] = i + j;
        }
    }
    for (int i = 0; i < M; i++) {
        x[i] = i;
    }

    // Alloca memoria sulla GPU
    cudaMalloc((void **)&d_A, N * M * sizeof(int));
    cudaMalloc((void **)&d_x, M * sizeof(int));
    cudaMalloc((void **)&d_y, N * sizeof(int));

    // Copia i dati dalla CPU alla GPU
    cudaMemcpy(d_A, A, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, M * sizeof(int), cudaMemcpyHostToDevice);

    // Definisci la griglia e il blocco di thread
    dim3 grid(1, 1, 1);
    dim3 block(N, 1, 1);

    // Esegui il kernel CUDA
    matrix_vector_product<<<grid, block>>>(d_A, d_x, d_y);

    // Copia i dati dalla GPU alla CPU
    cudaMemcpy(y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Stampa il risultato
    for (int i = 0; i < N; i++) {
        printf("%d ", y[i]);
    }
    printf("\n");

    // Libera la memoria sulla GPU
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}