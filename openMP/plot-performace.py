import pandas
import matplotlib.pyplot as plt
import numpy as np
	
def print_all_results(samplings_parallel,sampling_serial, k):

	plt.style.use('seaborn')
	plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0, axis='x')
	plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='y')
	
	fig, ax = plt.subplots()
	fig.set_size_inches(16, 9)

	ax.plot(samplings_parallel[0], samplings_parallel[1], label="Plot di mean al variare del numero di thread e K fissato", linewidth=2.0, color='indigo')
	ax.scatter(1, sampling_serial, label="Mean value for serial product with K fixed", color='red')
	ax.legend(loc='upper right', shadow=True, fontsize=10)

	plt.title("Plot di (num_threads, media dei tempi) fissato K ={} ".format(k), fontsize=20, fontname='DejaVu Sans', weight='bold', style='italic')

	plt.xlabel("Numero Thread")
	
	plt.ylabel("Mean")
	
	plt.show()
	
	
	

samplings_3_parallel = [[],[]]
samplings_4_parallel  = [[],[]]
samplings_8_parallel  = [[],[]]
samplings_12_parallel  = [[],[]]
samplings_16_parallel  = [[],[]]
samplings_32_parallel  = [[],[]]
samplings_64_parallel  = [[],[]]


# Leggo le misure delle prestazioni per il parallelo
df_parallel = pandas.read_csv("samplings_parallel_CSR.csv", dtype={'K': 'int64','num_thread': 'int64','mean': 'float64'})

print("Stampa campionamenti esecuzione parallela...")

for row in df_parallel.itertuples(index= False):
	try:	
		if(row[0] == 3):
			samplings_3_parallel[0].append(row[1])
			samplings_3_parallel[1].append(row[2])
		elif(row[0] == 4):
			samplings_4_parallel[0].append(row[1])
			samplings_4_parallel[1].append(row[2])
		elif(row[0] == 8):
			samplings_8_parallel[0].append(row[1])
			samplings_8_parallel[1].append(row[2])
		elif(row[0] == 12):
			samplings_12_parallel[0].append(row[1])
			samplings_12_parallel[1].append(row[2])
		elif(row[0] == 16):
			samplings_16_parallel[0].append(row[1])
			samplings_16_parallel[1].append(row[2])
		elif(row[0] == 32):
			samplings_32_parallel[0].append(row[1])
			samplings_32_parallel[1].append(row[2])
		elif(row[0] == 64):
			samplings_64_parallel[0].append(row[1])
			samplings_64_parallel[1].append(row[2])
		else:
			print("Valore di K non ammissibile.")
	except:
		print("Errore nella lettura della riga ")
		exit(1)

# Leggo le misure delle prestazioni per il seriale
df_serial = pandas.read_csv("samplings_serial_CSR.csv", dtype={'K': 'int64','mean': 'float64'})

print("Stampa campionamenti esecuzione parallela...")


for row in df_serial.itertuples(index= False):
	try:	
		if(row[0] == 3):
			samplings_3_serial = row[1]
		elif(row[0] == 4):
			samplings_4_serial = row[1]
		elif(row[0] == 8):
			samplings_8_serial= row[1]
		elif(row[0] == 12):
			samplings_12_serial= row[1]
		elif(row[0] == 16):
			samplings_16_serial= row[1]
		elif(row[0] == 32):
			samplings_32_serial= row[1]
		elif(row[0] == 64):
			samplings_64_serial= row[1]
		else:
			print("Valore di K non ammissibile.")
	except:
		print("Errore nella lettura della riga ")
		exit(1)
		

print_all_results(samplings_3_parallel,samplings_3_serial, '3')
print_all_results(samplings_4_parallel, samplings_4_serial, '4')
print_all_results(samplings_8_parallel, samplings_8_serial, '8')
print_all_results(samplings_12_parallel, samplings_12_serial, '12')
print_all_results(samplings_16_parallel, samplings_16_serial, '16')
print_all_results(samplings_32_parallel, samplings_32_serial, '32')
print_all_results(samplings_64_parallel, samplings_64_serial, '64')

print("I dati sono stati letti con successo.")

