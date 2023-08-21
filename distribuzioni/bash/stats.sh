#!/bin/usr/env bash

echo -n "Inserire il numero di righe della matrice da analizzare: "
read num_rows
echo $num_rows


echo -n "Inserire il percorso del file in formato matrix market: "
read filename
echo $filename


declare -a arr=( $(for i in $(eval echo {1..$num_rows}); do echo 0; done) )
diag=0
const=2
totale_nz=0

for n in $(eval echo {1..$num_rows})
do
	# Computo la dimensione della stringa che rappresenta la riga
	len_number=$(echo ${#n})
	echo "Lunghezza della riga: $len_number"

	# Computo la dimensione della sotto-stringa di interesse per capire se sto sulla diagonale	
        total_length=$((len_number * const +1))
        echo "Dimensione totale da considerare: $total_length"

	# Itero su tutte le righe che hanno valore $n e cerco i non-zeri sulla diagonale
	# devo dimezzare il numero di non zeri sulla diagonali altrimenti li conterei due volte

	str_compl=$(cat $filename | grep "^$n ")
	while IFS= read -r line; do
		sub_str1=${line:0:len_number}
		sub_str2=${line:len_number+1:total_length-len_number-1}
		sub_str3=${line:len_number+1:total_length-len_number}
		pippo=$(echo ${#sub_str2})
		pluto=$(echo ${#sub_str3})
                echo "Dimensione seconda sottostringa $sub_str2: $pippo"
		echo "Dimensione terza sottostringa $sub_str3: $pluto"
		echo "Sottostringa: $sub_str1-$sub_str2"
		if [ "$sub_str1" == "$sub_str2" ] && [ "$sub_str1 " == "$sub_str3" ]; then
			diag=$((diag + 1))
			echo "[MATCHING] Sottostringa: $sub_str1 - $sub_str2"
		fi
	done <<< "$str_compl"

	temp=$(cat $filename | grep "^$n " | wc -l)
	
	arr[n]=$temp
done

echo "Numero sulla diagonale: $diag"

half=$((diag / 2))

echo "Elementi dimezzati sulla diagonale: $half"

for i in ${!arr[@]}; do
	elem=${arr[$i]}
	totale_nz=$((totale_nz + elem))
#	echo "$i: ${arr[$i]}"
done

# Non devo considerare la riga in cui mi dice il numero di righe, il numero di colonne e il numero totale degli elementi
totale_nz=$((totale_nz - 1))

echo "Totale di NZ: $totale_nz"
