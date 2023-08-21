#include <stdio.h>
#include <stdlib.h>

/**
 * Questo software consente di costruire un file che ha solamente
 * il numero di non zeri eliminando l'informazione relativa alla riga.
 * L'informazione relativa alla riga non serve poiché il numero di non
 * seri viene calcolato per tutte le righe, anche per quelle che non hanno
 * non zeri.
 */

int main(int argc, char **argv)
{
	int row;
	int num;
	int num_row;
	int counter;
	int *pie_chart_data;
	char *filename_sorgente;
	char *filename_destinazione;
	char *filename_pie_chart;
	double mean;
	FILE* source;
	FILE* dest;
	FILE* pie;
	
	if(argc != 5)
	{
		printf("./data-processing\tfilename-sorgente\tnumero-righe\tfilename-destinazione\tfilename-pie-chart\n");
		return 1;
	}
	
	
	filename_sorgente=argv[1];	
	source = fopen(filename_sorgente, "r");	
    	if (source == NULL) {
        	printf("Il nome del file sorgente passato in input non è corretto.\n");
        	return 1;
    	}
    	
    	
    	filename_destinazione=argv[3];    	
    	dest=fopen(filename_destinazione, "w+");
    	if(dest==NULL)
    	{
    		perror("Il nome del file destinazione passato in input non è corretto:");
    		return 1;
    	}
    	
    	
    	num_row=atoi(argv[2]);
    	if(!num_row)
    	{
    		printf("Errore nella conversione del numero di righe\n");
    		return 1;
    	}
    	
    	/* Questa struttura dati verrà utilizzata come una sorta di tabella hash */
    	pie_chart_data=(int *)calloc(num_row+10, sizeof(int));
    	if(pie_chart_data==NULL)
    	{
    		printf("Errore esecuzione della malloc()\n");
    		return 1;
    	}
    	
    
    	mean=0;
    	counter=0;
    	int nz=0; 	
	while (fscanf(source, "%d - %d\n", &row, &num) != EOF)
	{
		pie_chart_data[num]+=1;
		mean+=num;
		nz+=num;
        	printf("riga: %d\telementi: %d\n", row, num);
        	fprintf(dest, "%d\n", num);
        	fflush(dest);
        	counter++;
 	}
 	
 	
 	if(counter!=num_row)
 	{
 		printf("Errore nella corrispondenza per il numero delle righe\n");
 		printf("%d\t%d\n", counter, num_row);
 		return 1;
 	}
 	
 	printf("Numero di non-zeri totale: %f\n", mean);
 	
 	mean=mean/num_row;
 	
 	printf("Valore medio: %f\n", mean);
 	
 	
	filename_pie_chart=argv[4];
	pie = fopen(filename_pie_chart, "w+");	
    	if (pie == NULL) {
        	perror("Il nome del file PIe passato in input non è corretto:");
        	return 1;
    	}
 	
 	
    	/*
    	 * Scrivo il file che conterrà i dati per graficare il PIE CHART.
    	 * Il file sarà strutturato a coppie di righe nel seguente modo:
    	 * 1. Valore
    	 * 2. Numero di righe che hanno tale valore come non-zeri
    	 */
    	
    	int totale=0;
 	counter=0;
 	for(int i=0; i<(num_row+10);i++)
 	{
 		if(pie_chart_data[i] > 0)
		{
			totale+=pie_chart_data[i];
			printf("Valore: %d\tNumero-righe-con-questi-non-zeri: %d\n", i, pie_chart_data[i]);
        		fprintf(pie, "%d\n", i);
        		fprintf(pie, "%d\n", pie_chart_data[i]);
        		fflush(pie);
		}
 	}



 	printf("Numero di righe: %d\n", totale);
 	printf("Numero di non-zeri totale: %d\n", nz);
 	
 	
 	fclose(source);
 	fclose(dest);
 	fclose(pie);
 	free(pie_chart_data);
 		
	return 0;

}
