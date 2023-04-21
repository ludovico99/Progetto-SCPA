#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>

int main(int argc, char **argv)
{
    FILE *f;
    char *filename;
    char *mode;
    double actual_value;
    double value;
    int size;

    if(argc != 3)
    {
        printf("./a.out mode filename\n");
        exit(1);
    }

    mode = argv[1];

    filename = argv[2];

    actual_value = 0;

    size = 0;

    printf("Nome del file da cui leggere i dati: %s\n", filename);
    printf("Tipologia richiesta: %s\n", mode);

    f = fopen(filename, "r");

    if(f==NULL)
    {
        perror("Errore apertura del file: ");
        return 1;
    }

    if(strcmp("serial", mode)==0)
    {
        printf("E' stata richiesta la modalità seriale.\n");
        
        while(fscanf(f, "%lf\n", &value) != EOF)
        {
            actual_value += value;
            size++;
        }

        printf("Sono stati letti %d valori.\n", size);

        printf("Valore medio: %lf\n", actual_value / (double)size);
    }

    return 0;
}