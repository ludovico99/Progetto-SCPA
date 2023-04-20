#include "omp.h"
#include <stdio.h>
#include <unistd.h>

int main()
{
  int id;
#pragma omp parallel private(id) num_threads(2)
  {
    id = omp_get_thread_num();
    #pragma omp critical // prima critical ====================
    {
      printf("A\n");
    }
    #pragma omp critical // seconda critical ==================
    {
      printf("B\n");
    }
    printf("C\n"); // non e’ in nessuna critical -------
  }
  printf("\n");
}
