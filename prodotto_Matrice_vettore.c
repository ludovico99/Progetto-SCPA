#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4
#define M 4

int main() {
    int A[N][M], x[M], y[N];
    int i, j;

    // Inizializza la matrice A e il vettore x
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            A[i][j] = i + j;
        }
    }
    for (i = 0; i < M; i++) {
        x[i] = i;
    }

    #pragma omp parallel for shared(A, x, y) private(i, j) default(none)
    for (i = 0; i < N; i++) {
        y[i] = 0;
        for (j = 0; j < M; j++) {
            y[i] += A[i][j] * x[j];
        }
    }

    // Stampa il risultato
    for (i = 0; i < N; i++) {
        printf("%d ", y[i]);
    }
    printf("\n");

    return 0;
}