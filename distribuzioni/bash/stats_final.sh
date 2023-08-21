#!/bin/usr/env bash

echo -n "Inserire il numero di righe della matrice da analizzare: "
read num_rows
echo $num_rows


echo -n "Inserire il percorso del file in formato matrix market: "
read filename
echo $filename

echo -n "Inserire il nome del file per contenere i dati: "
read output
echo $output

# Verifico se il file esiste per crearlo
if [ -e $output ]; then
        echo "Il file $output esiste"
	exit 1
else    
        echo >> $output
fi


declare -a arr=( $(for i in $(eval echo {1..$num_rows}); do echo 0; done) )
diag=0
const=2
totale_nz=0

for n in $(eval echo {1..$num_rows})
do

	if [ $(expr $n % 100) == "0" ]; then
		echo "$n"
	fi

	# Computo la dimensione della stringa che rappresenta la riga
	len_number=$(echo ${#n})

	# Computo la dimensione della sotto-stringa di interesse per capire se sto sulla diagonale	
        total_length=$((len_number * const +1))

	temp=$(cat $filename | grep "^$n " | wc -l)
	
	if [ $n -eq $num_rows ]; then	
		arr[n]=$((temp - 1))
	else
		arr[n]=$temp
	fi

	totale_nz=$((totale_nz + temp))
done

# Non devo considerare la riga in cui mi dice il numero di righe, il numero di colonne e il numero totale degli elementi
totale_nz=$((totale_nz - 1))

echo "Totale di NZ: $totale_nz"

# Scrivo i dati all'interno del file precedentemente creato utilizzato il formato corretto
for i in ${!arr[@]}; do

	# Prendo l'elemento che deve essere copiato
	elem=${arr[$i]}

	# Formatto la stringa da inserire nel file
	printf -v row '%d - %d\n' "$i" "$elem"

	echo $row >> $output
done

