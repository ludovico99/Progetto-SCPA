import pandas
import matplotlib.pyplot as plt
		
def point:
	def __init__(self,num_threads,mean):
		self.num_threads = num_threads
		self.mean = mean
		

def print_all_results(samplings_x,k):

	plt.plot(samplings_x[0], samplings_x[1])
	
	plt.title("K = ",k)
	
	plt.xlabel("Numero Thread")
	
	plt.ylabel("Mean")
	
	plt.show()
	
	
	

samplings_3 = [[],[]]
samplings_4 = [[],[]]
samplings_8 = [[],[]]
samplings_12 = [[],[]]
samplings_16 = [[],[]]
samplings_32 = [[],[]]
samplings_64 = [[],[]]


# Leggo le misure delle prestazioni per il parallelo
df_parallel = pandas.read_csv("samplings_parallel.csv")

df_parallel = df.reset_index()

print("Stampa campionamenti esecuzione parallela...")

for index, row in df_serial.iterrrows():
	try:	
		samplings.append(sampling(row['K'], row['num_threads'], row['mean']))
		
		if(row['K'] == '3'):
			samplings_3[0].append(point(row['num_threads']))
			samplings_3[1].append(point(row['mean']))
		elif(row['K'] == 4):
			samplings_4[0].append(point(row['num_threads']))
			samplings_4[1].append(point(row['mean']))
		elif(row['K'] == 8):
			samplings_8[0].append(point(row['num_threads']))
			samplings_8[1].append(point(row['mean']))
		elif(row['K'] == 12):
			samplings_12[0].append(point(row['num_threads'])
			samplings_12[1].append(point(row['mean'])
		elif(row['K'] == 16):
			samplings_16[0].append(point(row['num_threads']))
			samplings_16[1].append(point(row['mean']))
		elif(row['K'] == 32):
			samplings_32[0].append(point(row['num_threads']))
			samplings_32[1].append(point(row['mean']))
		elif(row['K'] == 64):
			samplings_64[0].append(point(row['num_threads']))
			samplings_64[1].append(point(row['mean']))
		else:
			print("Valore di K non ammissibile.")
	except:
		print("Errore nella lettura della riga ", index)
		exit(1)
		
print("Numero di campioni per k = 3: ", len(samplings_3[0]))
print("Numero di campioni per k = 4: ", len(samplings_4[0]))
print("Numero di campioni per k = 8: ", len(samplings_8[0]))
print("Numero di campioni per k = 12: ", len(samplings_12[0]))
print("Numero di campioni per k = 16: ", len(samplings_16[0]))
print("Numero di campioni per k = 32: ", len(samplings_32[0]))
print("Numero di campioni per k = 64: ", len(samplings_64[0]))

print_all_results(samplings_3, '3')
print_all_results(samplings_4, '4')
print_all_results(samplings_8, '8')
print_all_results(samplings_12, '12')
print_all_results(samplings_16, '16')
print_all_results(samplings_32, '32')
print_all_results(samplings_64, '64')

print("I dati sono stati letti con successo.")

print(df.to_string())
