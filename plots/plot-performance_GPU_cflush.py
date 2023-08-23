import pandas
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.stats as stats

n = 10  # number of samplings
K = [1, 3, 4, 8, 12, 16, 32, 64]

samplings_csr_scalar = []
samplings_csr_vector = []
samplings_csr_vector_by_row = []
samplings_csr_vector_sw = []
samplings_csr_adaptive_p = []

stats_csr_scalar = [[], []]
stats_csr_vector = [[], []]
stats_csr_vector_by_row = [[], []]
stats_csr_vector_sw = [[], []]
stats_csr_adaptive_p = [[], []]

samplings_ellpack = [[], []]
samplings_ellpack_sub_warp = [[], []]

 
def compute_mean_var():

    for i in range(0, len(samplings_csr_scalar), n):

        gruppo = samplings_csr_scalar[i:i+n]
        media_gruppo = sum(gruppo) / len(gruppo)
        varianza_gruppo = sum((x - media_gruppo) ** 2 for x in gruppo) / len(gruppo)
        stats_csr_scalar[0].append(media_gruppo)
        stats_csr_scalar[1].append(varianza_gruppo)

        gruppo = samplings_csr_vector[i:i+n]
        media_gruppo = sum(gruppo) / len(gruppo)
        varianza_gruppo = sum((x - media_gruppo) ** 2 for x in gruppo) / len(gruppo)
        stats_csr_vector[0].append(media_gruppo)
        stats_csr_vector[1].append(varianza_gruppo)

        gruppo = samplings_csr_vector_by_row[i:i+n]
        media_gruppo = sum(gruppo) / len(gruppo)
        varianza_gruppo = sum((x - media_gruppo) ** 2 for x in gruppo) / len(gruppo)
        stats_csr_vector_by_row[0].append(media_gruppo)
        stats_csr_vector_by_row[1].append(varianza_gruppo)

        gruppo = samplings_csr_vector_sw[i:i+n]
        media_gruppo = sum(gruppo) / len(gruppo)
        varianza_gruppo = sum((x - media_gruppo) ** 2 for x in gruppo) / len(gruppo)
        stats_csr_vector_sw[0].append(media_gruppo)
        stats_csr_vector_sw[1].append(varianza_gruppo)

        gruppo = samplings_csr_adaptive_p[i:i+n]

        media_gruppo = sum(gruppo) / len(gruppo)
        varianza_gruppo = sum((x - media_gruppo) ** 2 for x in gruppo) / len(gruppo)
        stats_csr_adaptive_p[0].append(media_gruppo)
        stats_csr_adaptive_p[1].append(varianza_gruppo)

 


def print_all_results_CSR():

    plt.style.use('seaborn-v0_8')
    plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='x')
    plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='y')

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    mean_scalar = stats_csr_scalar[0]
    variance_scalar = stats_csr_scalar[1]
    std_scalar = np.sqrt(variance_scalar)

    mean_vector = stats_csr_vector[0]
    variance_vector = stats_csr_vector[1]
    std_vector = np.sqrt(variance_vector)

    mean_vector_by_row = stats_csr_vector_by_row[0]
    variance_vector_by_row = stats_csr_vector_by_row[1]
    std_vector_by_row = np.sqrt(variance_vector_by_row)

    mean_vector_sw = stats_csr_vector_sw[0]
    variance_vector_sw = stats_csr_vector_sw[1]
    std_vector_sw = np.sqrt(variance_vector_sw)

    mean_adaptive_p = stats_csr_adaptive_p[0]
    variance_adaptive_p = stats_csr_adaptive_p[1]
    std_adaptive_p = np.sqrt(variance_adaptive_p)

    ax.plot(K, mean_scalar, marker='o', markersize=2,
            label="GLOPS for CSR SCALAR", linewidth=0.5, color='indigo')
    ci = stats.t.interval(0.995, n-1, loc=mean_scalar,
                          scale=std_scalar/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='indigo', alpha=0.1)

    ax.plot(K, mean_vector, marker='o', markersize=2,
            label="GLOPS for CSR VECTOR", linewidth=0.5, color='red')
    ci = stats.t.interval(0.995, n-1, loc=mean_vector,
                          scale=std_vector/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='red', alpha=0.1)

    ax.plot(K, mean_vector_by_row, marker='o', markersize=2,
            label="GLOPS for CSR VECTOR BY ROW", linewidth=0.5, color='orange')
    ci = stats.t.interval(0.995, n-1, loc=mean_vector_by_row,
                          scale=std_vector_by_row/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='orange', alpha=0.1)

    ax.plot(K, mean_vector_sw, marker='o', markersize=2,
            label="GLOPS for CSR VECTOR SUB-WARP", linewidth=0.5, color='blue')
    ci = stats.t.interval(0.995, n-1, loc=mean_vector_sw,
                          scale=std_vector_sw/np.sqrt(n))

    ax.fill_between(K, ci[0], ci[1],
                    color='blue', alpha=0.1)

    ax.plot(K, mean_adaptive_p, marker='o', markersize=2,
            label="GLOPS for CSR ADAPTIVE PERSONALIZZATO", linewidth=0.5, color='cyan')
    ci = stats.t.interval(0.995, n-1, loc=mean_adaptive_p,
                          scale=std_adaptive_p/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='cyan', alpha=0.1)

    ax.legend(loc='upper left', shadow=True, fontsize=10)

    plt.title("Matrice {}: Plot dei GFLOPS al variare di K e dell'algoritmo utilizzato per il formato CSR".format(sys.argv[1]),
              fontsize=18, fontname='DejaVu Sans', weight='bold', style='italic')

    plt.xticks(K)

    plt.xlabel("Number of columns (K)")

    plt.ylabel("GLOPS")

    # Salva il grafico come immagine
    plt.savefig("Immagini/CSR_GPU_{}.png".format(sys.argv[1]))
   


# Leggo le misure delle prestazioni per il parallelo
if len(sys.argv) >= 2:
	#df_parallel = pandas.read_csv("samplings_cflush_ELLPACK_GPU_{}.csv".format(sys.argv[1]))
	df_parallel = pandas.read_csv("samplings_cflush_CSR_GPU_{}.csv".format(sys.argv[1]))
else:
	print("usage: prog matrix\n")
	exit(1)


for row in df_parallel.itertuples(index=False):

    try:
        
        if (row[0] == "csr_scalar"):

            samplings_csr_scalar.append(row[2])
        elif (row[0] == "csr_vector"):
            samplings_csr_vector.append(row[2])
        elif (row[0] == "csr_vector_by_row"):
            samplings_csr_vector_by_row.append(row[2])
        elif (row[0] == "csr_adaptive_personalizzato"):
            samplings_csr_adaptive_p.append(row[2])
        elif (row[0] == "csr_vector_sub_warp"):
            samplings_csr_vector_sw.append(row[2])
        elif (row[0] == "ellpack"):
            samplings_ellpack.append(row[2])
        elif (row[0] == "ellpack_sub_warp"):
            samplings_ellpack_sub_warp.append(row[2])
        else:
            print("Valore di algorithm non ammissibile.")
    except:
        print("Errore nella lettura della riga ")
        exit(1)


print("I dati sono stati letti con successo.")

compute_mean_var()

print_all_results_CSR()
#print_all_results_ELLPACK()
