#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

unsigned long * generate_vector(unsigned long n)
{
    unsigned long * vector;
    unsigned long i;

    vector = (unsigned long *)malloc(sizeof(unsigned long) * n);

    if(vector == NULL)
    {
        printf("Errore malloc()\n");
        exit(1);
    }

    for(i= 0; i<n; i++)
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

    for(i=0; i<n; i++)
    {
        v3 += v1[i] * v2[i];
        for(j=0;j<120;j++)
        {
            y = v1[i] * v2[i];
        }
    }



    printf("Risultato prodotto scalare seriale %ld\n", v3);

    return;
}


void parallel_dot_product_1(unsigned long *v1, unsigned long *v2, unsigned long n)
{
    unsigned long i;
    unsigned long v3;
    unsigned long y;
    unsigned long j;

    v3 = 0;
    y = 0;
    j = 0;

    unsigned long partial_sum = 0;

    #pragma omp parallel for schedule(dynamic) num_threads(10) shared(n, v1, v2) firstprivate(partial_sum, y, j) default(none)   

        for(i=0; i<n; i++)
        {
            partial_sum += v1[i] * v2[i];
            for(j=0;j<120;j++)
            {
                y = v1[i] * v2[i];
            }
        }
    #pragma omp critical(v3)
    {
        v3 += partial_sum;
    }

    


    printf("Risultato prodotto scalare parallelo %ld\n", v3);

    return;
}


int main(int argc, char **argv)
{
    unsigned long n;
    unsigned long *v1;
    unsigned long *v2;

    if(argc!=2)
        exit(1);

    n = atoi(argv[1]);

    v1 = generate_vector(n);

    v2 = generate_vector(n);

    //serial_dot_product(v1, v2, n);

    parallel_dot_product_1(v1, v2, n);
}