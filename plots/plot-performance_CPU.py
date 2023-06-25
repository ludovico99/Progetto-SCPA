import pandas
import matplotlib.pyplot as plt
import numpy as np
	
import scipy.stats as stats
import sys

samplings_3_parallel = [[],[],[]]
samplings_4_parallel  = [[],[],[]]
samplings_8_parallel  = [[],[],[]]
samplings_12_parallel  = [[],[],[]]
samplings_16_parallel  = [[],[],[]]
samplings_32_parallel  = [[],[],[]]
samplings_64_parallel  = [[],[],[]]

samplings_parallel = [samplings_3_parallel, samplings_4_parallel, samplings_8_parallel, 
samplings_12_parallel, samplings_16_parallel, samplings_32_parallel , samplings_64_parallel]

samplings_3_serial = []
samplings_4_serial  = []
samplings_8_serial  = []
samplings_12_serial  = []
samplings_16_serial  = []
samplings_32_serial  = []
samplings_64_serial  = []

samplings_serial = [samplings_3_serial, samplings_4_serial, samplings_8_serial, samplings_12_serial,samplings_16_serial, samplings_32_serial, samplings_64_serial]

color_names = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan']

n = 30  # number of samplings
K = [3, 4, 8, 12, 16, 32, 64]


def print_all_results():

	plt.style.use('seaborn-v0_8')
	plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='x')
	plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='y')
	
	fig, ax = plt.subplots()
	fig.set_size_inches(16, 9)

	for i in range (0, len(K)):

		x = samplings_parallel[i][0]
		mean = samplings_parallel[i][1]
		variance = samplings_parallel[i][2]
		std = np.sqrt(variance)

		serial_mean = samplings_serial[i][0]
		serial_variance = samplings_serial[i][1]
		std_serial = np.sqrt(serial_variance)

		ax.scatter(1, serial_mean, label="Mean value for serial product with K = {}".format(K[i]), color=color_names[i])

		ci = stats.t.interval(0.995, n-1, loc = serial_mean, scale=std_serial/np.sqrt(n))
		ax.fill_between([1], ci[0], ci[1],
						color=color_names[i], alpha=0.1)

		ax.plot(x, mean,marker='o', markersize=2, label="Mean value with K = {} and variable num_thread". format(K[i]), linewidth=0.5, color=color_names[i])

		ci = stats.t.interval(0.995, n-1, loc = mean, scale=std/np.sqrt(n))

		ax.fill_between(x, ci[0], ci[1],
						color=color_names[i], alpha=0.1)



	ax.legend(loc='upper right', shadow=True, fontsize=10)

	plt.title("Matrice {}: Plot della media dei tempi in secondi al variare del numero di threads e K".format(sys.argv[1]), 
	   fontsize=20, fontname='DejaVu Sans', weight='bold', style='italic')

	plt.xlabel("Numero Thread")
	
	plt.xticks(samplings_parallel[0][0])

	plt.ylabel("Mean time in seconds")

	#plt.savefig("Immagini/CSR_CPU_{}.png".format(sys.argv[1]))
	plt.savefig("Immagini/ELLPACK_CPU_{}.png".format(sys.argv[1]))

	
	
	


# Leggo le misure delle prestazioni per il parallelo

if len(sys.argv) >= 2:
	df_parallel = pandas.read_csv("samplings_ELLPACK_CPU_parallel_{}.csv".format(sys.argv[1]))
	#df_parallel = pandas.read_csv("samplings_CSR_CPU_parallel_{}.csv".format(sys.argv[1]))
else:
	print("usage: prog matrix\n")
	exit(1)

for row in df_parallel.itertuples(index= False):

	try:	
		if(row[0] == 3):
			samplings_3_parallel[0].append(row[1])
			samplings_3_parallel[1].append(row[2])
			samplings_3_parallel[2].append(row[3])
		elif(row[0] == 4):
			samplings_4_parallel[0].append(row[1])
			samplings_4_parallel[1].append(row[2])
			samplings_4_parallel[2].append(row[3])
		elif(row[0] == 8):
			samplings_8_parallel[0].append(row[1])
			samplings_8_parallel[1].append(row[2])
			samplings_8_parallel[2].append(row[3])
		elif(row[0] == 12):
			samplings_12_parallel[0].append(row[1])
			samplings_12_parallel[1].append(row[2])
			samplings_12_parallel[2].append(row[3])
		elif(row[0] == 16):
			samplings_16_parallel[0].append(row[1])
			samplings_16_parallel[1].append(row[2])
			samplings_16_parallel[2].append(row[3])
		elif(row[0] == 32):
			samplings_32_parallel[0].append(row[1])
			samplings_32_parallel[1].append(row[2])
			samplings_32_parallel[2].append(row[3])
		elif(row[0] == 64):
			samplings_64_parallel[0].append(row[1])
			samplings_64_parallel[1].append(row[2])
			samplings_64_parallel[2].append(row[3])
		else:
			print("Valore di K non ammissibile.")
	except:
		print("Errore nella lettura della riga ")
		exit(1)

# Leggo le misure delle prestazioni per il seriale
if len(sys.argv) >= 2:
	df_serial = pandas.read_csv("samplings_ELLPACK_CPU_serial_{}.csv".format(sys.argv[1]))
	#df_serial = pandas.read_csv("samplings_CSR_CPU_serial_{}.csv".format(sys.argv[1]))
else:
    print("usage: prog matrix\n")



for row in df_serial.itertuples(index= False):
	try:	
	
		if(row[0] == 3):
			samplings_3_serial.append(row[1])
			samplings_3_serial.append(row[2])
		elif(row[0] == 4):
			samplings_4_serial.append(row[1])
			samplings_4_serial.append(row[2])
		elif(row[0] == 8):
			samplings_8_serial.append(row[1])
			samplings_8_serial.append(row[2])
		elif(row[0] == 12):
			samplings_12_serial.append(row[1])
			samplings_12_serial.append(row[2])
		elif(row[0] == 16):
			samplings_16_serial.append(row[1])
			samplings_16_serial.append(row[2])
		elif(row[0] == 32):
			samplings_32_serial.append(row[1])
			samplings_32_serial.append(row[2])
		elif(row[0] == 64):
			samplings_64_serial.append(row[1])
			samplings_64_serial.append(row[2])
		else:
			print("Valore di K non ammissibile.")
	except:
		print("Errore nella lettura della riga ")
		exit(1)
		

print_all_results()


print("I dati sono stati letti con successo.")

