#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define BILLION  1000000000L
#define NUM_THREADS 300
#define NUM_ITER_DELAY 120
#define NUM_SAMPLINGS 10

unsigned long DIM_SPACE_ITERATION = 100000000;

unsigned long * generate_vector()
{
    unsigned long * vector;
    unsigned long i;

    vector = (unsigned long *)malloc(sizeof(unsigned long) * DIM_SPACE_ITERATION);

    if(vector == NULL)
    {
        printf("Errore malloc()\n");
        exit(1);
    }

    for(i= 0; i<DIM_SPACE_ITERATION; i++)
    {
        vector[i] = 1;
    }

    printf("Generazione vettore completata con successo\n");

    fflush(stdout);

    return vector;
}

void serial_dot_product(unsigned long *v1, unsigned long *v2)
{
    unsigned long i;
    unsigned long v3;
    unsigned long y;
    unsigned long j;

    v3 = 0;

    for(i=0; i<DIM_SPACE_ITERATION; i++)
    {
        v3 += v1[i] * v2[i];

        for(j=0;j<NUM_ITER_DELAY;j++)
        {
            y = v1[i] * v2[i];
        }
    }

    printf("Risultato prodotto scalare seriale %ld\n", v3);

    return;
}


void parallel_dot_product_1_static(unsigned long *v1, unsigned long *v2)
{
    unsigned long i;
    unsigned long v3;
    unsigned long y;
    unsigned long j;
    int first;

    v3 = 0;
    y = 0;
    j = 0;
    first = 0;


#ifdef FIRST_PRIV_V3
    /*
    * In questo scenario, il risultato finale di v3 risulta essere 0.
    * Questo è dovuto al fatto che v3 è privata per ogni thread. Di
    * conseguenza, ogni thread ha la propria locazione di memoria
    * relativa alla istanza di v3. La printf conclusiva del valore
    * finale prende la locazione di memoria v3 = 0 inizializzata prima
    * della sezione parallela.
    * I due vettori v1 e v2, insieme alla loro dimensione DIM_SPACE_ITERATION, sono
    * condivisi tra tutti i threads. Questo non è un problema poiché ogni thread
    * deve sapere quali sono gli elementi del vettore che dovrà gestire. La
    * dimensione dell'array serve al run time OPENMP per determinare la
    * suddivisione del carico di lavoro tra i vari threads.
    */
    #pragma omp parallel num_threads(NUM_THREADS) default(none) shared(DIM_SPACE_ITERATION, v1, v2) private(j,y) firstprivate(v3)
#else
    /*
    * Supponiamo di avere un numero di iterazioni che è pari a 
    */
    #pragma omp parallel num_threads(NUM_THREADS) default(none) shared(DIM_SPACE_ITERATION, v1, v2, v3) private(j,y)
#endif
    {
        unsigned long partial_sum = 0;

        #pragma omp for schedule(static, 10000000)
        for(i=0; i<DIM_SPACE_ITERATION; i++)
        {
            partial_sum += v1[i] * v2[i];

            for(j=0;j<NUM_ITER_DELAY;j++)
            {
                y = v1[i] * v2[i];
            }
        }

        printf("Valore di partial_sum per il thread (%d): (%ld)\n", omp_get_thread_num(), partial_sum);      

        #pragma omp critical(Aggiornamento)
        v3+=partial_sum;

    }

    printf("Risultato prodotto scalare parallelo %ld\n", v3);

    return;
}


int main(int argc, char **argv)
{
    unsigned long n;
    unsigned long *v1;
    unsigned long *v2;

    struct timespec start, stop;
    double accum;

    FILE *f;

    if(argc > 1)
    {
        printf("Non bisogna passare alcun parametro, eseguire:\n./a.out\n");
        exit(1);
    }
    
    /* Genero il primo vettore */
    v1 = generate_vector();

    /* Genero il secondo vettore */
    v2 = generate_vector();

    f = fopen("results4.txt", "w+");

    if(f == NULL)
    {
        perror("Errore apertura del file: ");
        return 1;
    }

    for(int i = 0; i<NUM_SAMPLINGS; i++)
    {
        if(clock_gettime(CLOCK_REALTIME, &start)==-1)
        {
            perror("Errore clock()");
            exit(EXIT_FAILURE);
        }

        #ifdef SERIAL
            serial_dot_product(v1, v2);
        #else
            parallel_dot_product_1_static(v1, v2);
        #endif

        if(clock_gettime(CLOCK_REALTIME, &stop)==-1)
        {
            perror("Errore clock()");
            exit(EXIT_FAILURE);
        }

        accum = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;

        fprintf(f, "%lf\n", accum);

        fflush(f);

    }

    return 0;

}