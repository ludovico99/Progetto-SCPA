#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define BILLION 1000000000L

unsigned long *generate_vector(unsigned long n)
{
    unsigned long *vector;
    unsigned long i;

    vector = (unsigned long *)malloc(sizeof(unsigned long) * n);

    if (vector == NULL)
    {
        printf("Errore malloc()\n");
        exit(1);
    }

    for (i = 0; i < n; i++)
    {
        vector[i] = 1;
    }

    printf("Fine vettore\n");
    fflush(stdout);

    return vector;
}

void serial_dot_product(unsigned long *v1, unsigned long *v2, unsigned long n)
{
    unsigned long i;
    unsigned long v3;
    unsigned long y;
    unsigned long j;

    v3 = 0;

    for (i = 0; i < n; i++)
    {
        v3 += v1[i] * v2[i];
        for (j = 0; j < 120; j++)
        {
            y = v1[i] * v2[i];
        }
    }

    printf("Risultato prodotto scalare seriale %ld\n", v3);

    return;
}

/*La sintassi per realizzare un’operazione di riduzione in openMP `e la seguente:
#pragma omp parallel reduction(operazione: nomevariabile)*/
void parallel_dot_product_red(unsigned long *v1, unsigned long *v2, unsigned long n)
{
    unsigned long i;
    unsigned long v3;
    unsigned long y;
    unsigned long j;
    int nthreads;

    v3 = 0;
    y = 0;
    j = 0;

    nthreads = 10;
#pragma omp parallel num_threads(nthreads) shared(n, v1, v2) firstprivate(y, j) reduction(+ \
                                                                                          : v3) default(none)
    {
        printf("Sono il thread %d\n", omp_get_thread_num());
#pragma omp for schedule(dynamic)
        for (i = 0; i < n; i++)
        {
            v3 += v1[i] * v2[i];
            for (j = 0; j < 120; j++)
            {
                y = v1[i] * v2[i];
            }
        }
    }

    printf("Risultato prodotto scalare parallelo con dynamic %ld\n", v3);

    return;
}

/*Ogni thread esegue un chunk di iterazioni e quindi richiede un altro chunk fino a quando non ci sono pi`u chunk disponibili.*/
void parallel_dot_product_dynamic(unsigned long *v1, unsigned long *v2, unsigned long n)
{
    unsigned long i;
    unsigned long v3;
    unsigned long y;
    unsigned long j;
    int nthreads;

    v3 = 0;
    y = 0;
    j = 0;

    unsigned long partial_sum = 0;
    nthreads = 10;
#pragma omp parallel num_threads(nthreads) shared(n, v1, v2, v3) firstprivate(partial_sum, y, j) default(none)
    {
        printf("Sono il thread %d\n", omp_get_thread_num());
#pragma omp for schedule(dynamic)
        for (i = 0; i < n; i++)
        {
            partial_sum += v1[i] * v2[i];
            for (j = 0; j < 120; j++)
            {
                y = v1[i] * v2[i];
            }
        }
#pragma omp critical
        {
            v3 += partial_sum;
        }
    }

    printf("Risultato prodotto scalare parallelo con dynamic %ld\n", v3);

    return;
}

/* La differenza con il tipo di pianificazione dinamica `e nella dimensione dei chunk.
La dimensione di un chunk `e proporzionale al numero di iterazioni non assegnate diviso per il numero dei thread.
Pertanto la dimensione dei chunk diminuisce.*/
void parallel_dot_product_guided(unsigned long *v1, unsigned long *v2, unsigned long n)
{
    unsigned long i;
    unsigned long v3;
    unsigned long y;
    unsigned long j;
    unsigned long nthreads;

    v3 = 0;
    y = 0;
    j = 0;

    unsigned long partial_sum = 0;
    nthreads = 10;
#pragma omp parallel num_threads(nthreads) shared(n, v1, v2, v3) firstprivate(partial_sum, y, j) default(none)
    {
        printf("Sono il thread %d\n", omp_get_thread_num());

#pragma omp for schedule(guided)
        for (i = 0; i < n; i++)
        {
            partial_sum += v1[i] * v2[i];
            for (j = 0; j < 120; j++)
            {
                y = v1[i] * v2[i];
            }
        }

#pragma omp critical
        {
            v3 += partial_sum;
        }
    }

    printf("Risultato prodotto scalare parallelo con guided %ld\n", v3);

    return;
}

int main(int argc, char **argv)
{
    unsigned long n;
    unsigned long *v1;
    unsigned long *v2;
    double accum;

    struct timespec start, stop;

    if (argc != 2)
        exit(1);

    n = atoi(argv[1]);

    v1 = generate_vector(n);

    v2 = generate_vector(n);

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    serial_dot_product(v1, v2, n);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;

    printf("SERIAL: %lf\n", accum);

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    parallel_dot_product_dynamic(v1, v2, n);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;

    printf("DYNAMIC: %lf\n", accum);

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    parallel_dot_product_guided(v1, v2, n);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;

    if (clock_gettime(CLOCK_REALTIME, &start) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    printf("%lf\n", accum);

    parallel_dot_product_red(v1, v2, n);

    if (clock_gettime(CLOCK_REALTIME, &stop) == -1)
    {
        perror("Errore clock()");
        exit(EXIT_FAILURE);
    }

    accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / (double)BILLION;

    printf("REDUCTION: %lf\n", accum);
    free(v1);
    free(v2);

    return 0;
}