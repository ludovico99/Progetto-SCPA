import pandas
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats

n = 3  # number of samplings
K = [1, 3, 4, 8, 12, 16, 32, 64]
samplings_csr_scalar = [[], []]
samplings_csr_vector = [[], []]
samplings_csr_vector_sw = [[], []]
samplings_csr_adaptive = [[], []]

samplings_ellpack = [[], []]
samplings_ellpack_sub_warp = [[], []]


def print_all_results_CSR():

    plt.style.use('seaborn-v0_8')
    plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='x')
    plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='y')

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    mean_scalar = samplings_csr_scalar[0]
    variance_scalar = samplings_csr_scalar[1]
    std_scalar = np.sqrt(variance_scalar)

    mean_vector = samplings_csr_vector[0]
    variance_vector = samplings_csr_vector[1]
    std_vector = np.sqrt(variance_vector)

    mean_vector_sw = samplings_csr_vector_sw[0]
    variance_vector_sw = samplings_csr_vector_sw[1]
    std_vector_sw = np.sqrt(variance_vector_sw)

    mean_adaptive = samplings_csr_adaptive[0]
    variance_adaptive = samplings_csr_adaptive[1]
    std_adaptive = np.sqrt(variance_adaptive)

    ax.plot(K, mean_scalar, marker='o', markersize=2,
            label="GLOPS for CSR SCALAR", linewidth=0.5, color='indigo')
    ci = stats.t.interval(0.995, n-1, loc=mean_scalar,
                          scale=std_scalar/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='indigo', alpha=0.1,
                    label="Confidence band of 95 for CSR SCALAR")

    ax.plot(K, mean_vector, marker='o', markersize=2,
            label="GLOPS for CSR VECTOR", linewidth=0.5, color='red')
    ci = stats.t.interval(0.995, n-1, loc=mean_vector,
                          scale=std_vector/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='red', alpha=0.1,
                    label="Confidence band of 95 for CSR VECTOR")

    ax.plot(K, mean_vector_sw, marker='o', markersize=2,
            label="GLOPS for CSR VECTOR SUB-WARP", linewidth=0.5, color='blue')
    ci = stats.t.interval(0.995, n-1, loc=mean_vector_sw,
                          scale=std_vector_sw/np.sqrt(n))

    ax.fill_between(K, ci[0], ci[1],
                    color='blue', alpha=0.1,
                    label="Confidence band of 95 for CSR VECTOR SUB-WARP")

    ax.plot(K, mean_adaptive, marker='o', markersize=2,
            label="GLOPS for CSR ADAPTIVE", linewidth=0.5, color='green')
    ci = stats.t.interval(0.995, n-1, loc=mean_adaptive,
                          scale=std_adaptive/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='green', alpha=0.1,
                    label="Confidence band of 95 for CSR ADAPTIVE")

    ax.legend(loc='upper left', shadow=True, fontsize=10)

    plt.title("Plot dei GFLOPS al variare di K e dell'algoritmo (FORMATO CSR)",
              fontsize=20, fontname='DejaVu Sans', weight='bold', style='italic')

    plt.xticks(K)

    plt.xlabel("Number of columns K")

    plt.ylabel("GLOPS")

    plt.show()


def print_all_results_ELLPACK():

    mean = samplings_ellpack[0]
    variance = samplings_ellpack[1]
    std = np.sqrt(variance)

    mean_sw = samplings_ellpack_sub_warp[0]
    variance_sw = samplings_ellpack_sub_warp[1]
    std_sw = np.sqrt(variance_sw)

    plt.style.use('seaborn-v0_8')
    plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='x')
    plt.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.2, axis='y')

    fig, ax = plt.subplots()
    fig.set_size_inches(16, 9)

    ax.plot(K, mean, marker='o', markersize=2,
            label="GLOPS for ELLPACK", linewidth=0.5, color='indigo')
    ci = stats.t.interval(0.995, n-1, loc=mean, scale=std/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='indigo', alpha=0.1,
                    label="Confidence band of 95 of ELLPACK")

    ax.plot(K, mean_sw, marker='o', markersize=2,
            label="GLOPS for ELLPACK with sub-warps", linewidth=0.5, color='red')
    ci = stats.t.interval(0.995, n-1, loc=mean_sw, scale=std_sw/np.sqrt(n))
    ax.fill_between(K, ci[0], ci[1],
                    color='red', alpha=0.1,
                    label="Confidence band of 95 of ELLPACK with sub-warps")

    ax.legend(loc='upper right', shadow=True, fontsize=10)

    plt.title("Plot dei GFLOPS al variare di K e dell'algoritmo (FORMATO ELLPACK)",
              fontsize=20, fontname='DejaVu Sans', weight='bold', style='italic')

    plt.xlabel("Number of columns K")

    plt.xticks(K)

    plt.ylabel("GLOPS")

    plt.show()


# Leggo le misure delle prestazioni per il parallelo
# df_parallel = pandas.read_csv("samplings_ELLPACK_GPU.csv")
df_parallel = pandas.read_csv("samplings_CSR_GPU.csv")

for row in df_parallel.itertuples(index=False):

    try:
        if (row[0] == "csr_scalar"):
            samplings_csr_scalar[0].append(row[2])
            samplings_csr_scalar[1].append(row[3])
        elif (row[0] == "csr_vector"):
            samplings_csr_vector[0].append(row[2])
            samplings_csr_vector[1].append(row[3])
        elif (row[0] == "csr_adaptive"):
            samplings_csr_adaptive[0].append(row[2])
            samplings_csr_adaptive[1].append(row[3])
        elif (row[0] == "csr_vector_sub_warp"):
            samplings_csr_vector_sw[0].append(row[2])
            samplings_csr_vector_sw[1].append(row[3])
        elif (row[0] == "ellpack"):
            samplings_ellpack[0].append(row[2])
            samplings_ellpack[1].append(row[3])
        elif (row[0] == "ellpack_sub_warp"):
            samplings_ellpack_sub_warp[0].append(row[2])
            samplings_ellpack_sub_warp[1].append(row[3])
        else:
            print("Valore di algorithm non ammissibile.")
    except:
        print("Errore nella lettura della riga ")
        exit(1)


print("I dati sono stati letti con successo.")

print_all_results_CSR()
# print_all_results_ELLPACK()
