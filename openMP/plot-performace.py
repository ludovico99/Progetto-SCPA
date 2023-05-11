import pandas
import matplotlib.pyplot as plt
import numpy as np
	
import scipy.stats as stats

def print_all_results(samplings_parallel,sampling_serial, k):

	n = 30
	x = samplings_parallel[0]
	mean = samplings_parallel[1]
	variance = samplings_parallel[2]
	std = np.sqrt(variance)

	serial_mean = sampling_serial[0]
	serial_variance = sampling_serial[1]
	std_serial = np.sqrt(serial_variance)

	plt.style.use('seaborn-v0_8')
	plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='x')
	plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='y')
	
	fig, ax = plt.subplots()
	fig.set_size_inches(16, 9)

	ax.plot(x, mean,marker='o', markersize=2, label="Mean value with K fixed and variable num_thread", linewidth=0.5, color='indigo')

	ci = stats.t.interval(0.995, n-1, loc = mean, scale=std/np.sqrt(n))

	ax.fill_between(x, ci[0], ci[1],
                    color='darkviolet', alpha=0.1,
                    label="Confidence band of 95")


	
	ax.scatter(1, serial_mean, label="Mean value for serial product with K fixed", color='red')

	ci = stats.t.interval(0.995, n-1, loc = serial_mean, scale=std_serial/np.sqrt(n))
	ax.fill_between([1], ci[0], ci[1],
                    color='red', alpha=0.1,
                    label="Confidence band of 95")

	ax.legend(loc='upper right', shadow=True, fontsize=10)

	plt.title("Plot di (num_threads, media dei tempi) fissato K ={} ".format(k), fontsize=20, fontname='DejaVu Sans', weight='bold', style='italic')

	plt.xlabel("Numero Thread")
	
	plt.ylabel("Mean")
	
	plt.show()
	
	
	

samplings_3_parallel = [[],[],[]]
samplings_4_parallel  = [[],[],[]]
samplings_8_parallel  = [[],[],[]]
samplings_12_parallel  = [[],[],[]]
samplings_16_parallel  = [[],[],[]]
samplings_32_parallel  = [[],[],[]]
samplings_64_parallel  = [[],[],[]]


# Leggo le misure delle prestazioni per il parallelo
#df_parallel = pandas.read_csv("samplings_parallel_ELLPACK.csv", dtype={'K': 'int64','num_thread': 'int64','mean': 'float64'})
df_parallel = pandas.read_csv("samplings_parallel_CSR.csv", dtype={'K': 'int64','num_thread': 'int64','mean': 'float64'})

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
#df_serial = pandas.read_csv("samplings_serial_ELLPACK.csv", dtype={'K': 'int64','mean': 'float64', 'variance': 'float64'})
df_serial = pandas.read_csv("samplings_serial_CSR.csv", dtype={'K': 'int64','mean': 'float64'})

samplings_3_serial = []
samplings_4_serial  = []
samplings_8_serial  = []
samplings_12_serial  = []
samplings_16_serial  = []
samplings_32_serial  = []
samplings_64_serial  = []

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
		

print_all_results(samplings_3_parallel,samplings_3_serial, '3')
print_all_results(samplings_4_parallel, samplings_4_serial, '4')
print_all_results(samplings_8_parallel, samplings_8_serial, '8')
print_all_results(samplings_12_parallel, samplings_12_serial, '12')
print_all_results(samplings_16_parallel, samplings_16_serial, '16')
print_all_results(samplings_32_parallel, samplings_32_serial, '32')
print_all_results(samplings_64_parallel, samplings_64_serial, '64')

print("I dati sono stati letti con successo.")

