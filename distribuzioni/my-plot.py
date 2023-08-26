from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import sys

num_params=len(sys.argv)

list_colors = ['black', 'red', 'green', 'yellow', 'blue', 'aqua', 'plum','orange','peru', 'tomato', 'lime', 'royalblue', 'grey', 'navy', 'teal', 'gold', 'darkviolet', 'firebrick', 'salmon', 'lawngreen', 'olive', 'purple', 'sienna', 'deepskyblue', 'yellowgreen', 'darksalmon', 'cadetblue', 'magenta', 'coral', 'darkgreen', 'mediumblue']


if(num_params!=4):
	print("python3 my-plot filename-plot titolo-grafico filename-pie")
	exit(1)

# Recupero il nome del file nel nuovo formato. Questo file contiene il numero degli elementi riga per riga
filename_plot = sys.argv[1]

# Recupero il titolo del grafico
titolo=sys.argv[2]

# Recupero il nome del file nel nuovo formato. Questo file contiene coppie di righe (Valore, Numero di righe che hanno 'Valore' come non zeri).
filename_pie=sys.argv[3]


# ----------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(16,9))

plt.subplot(2, 1, 1)


# Leggo le linee del file
data_file = open(filename_plot, 'r')
lines = data_file.readlines()
 
# Costruisco l'array di dati per l'istogramma
y = []
x = []
counter=0
for line in lines:
    y.append(int(line))
    counter=counter+1
    x.append(counter)

mean = np.mean(y)
var = np.var(y)


plt.title("Distribuzione dei non-zeri nelle righe della matrice " + titolo + "\nMEAN = " + str(mean) + " , VAR = " + str(var))
plt.xlabel("Riga")
plt.ylabel("Numero di non-zeri")



#plt.plot(x, y, marker='o', linestyle = 'dotted', ms=2)
plt.scatter(x, y)
plt.tick_params(axis='x', which='major', labelsize=10)


# ----------------------------------------------------------------------------------------------------------------

plt.subplot(2, 1, 2)

# Leggo le linee del file
data_file = open(filename_pie, 'r')
lines = data_file.readlines()
 
# Costruisco l'array di dati per l'istogramma
valori = []
num_righe_con_tale_valore = []
parity=0
for line in lines:
	if(parity==0):
		# Sono nella prima riga con il valore
		valori.append(line)
	else:
		# Sono nella seconda riga con il numero di righe che hanno il valore come numero di non zeri
		num_righe_con_tale_valore.append(int(line))
	parity=(parity+1)%2
	


my_explode=[]
for i in valori:
	my_explode.append(0.2)


plt.pie(num_righe_con_tale_valore, labels=valori, autopct='%1.1f%%', colors=list_colors[:len(valori)], explode=my_explode)


    
# ----------------------------------------------------------------------------------------------------------------


plt.savefig("images/" + titolo + ".png")
# plt.show()

