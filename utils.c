#include "header.h"
#include "mmio.h"


/* Computazione della dimensione del chunk per parallelizzare */
int compute_chunk_size(int value, int nthread)
{
    int chunk_size;

    if (value % nthread == 0)
        chunk_size = value / nthread;
    else
        chunk_size = value / nthread + 1;

    return chunk_size;
}