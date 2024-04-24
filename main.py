import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hclust
from functii import *

def execute():
    tabel = pd.read_csv('C:\\Users\\MateBook D15\\Downloads\\proiect_dsad_ex1\\date_tari.csv',
                        index_col=0)

    variabile = list(tabel)[1:]
    instante = list(tabel.index)

    n = len(instante)
    m = len(variabile)

    nan_replace(tabel)
    x = tabel[variabile].values

    # construire ierarhie
    h = hclust.linkage(x, method='ward')
    print('h', h)

    p = n - 1

    k_dif_max = np.argmax(h[1:, 2] - h[:(p-1), 2])
    print("Dif max", k_dif_max)

    nr_clusteri = p - k_dif_max
    print("Nr clusteri recomandati", nr_clusteri)

    partitie_opt = partitie(h, nr_clusteri, p, instante)
    print("Partitie optima", partitie_opt)

    partitie_opt_t = pd.DataFrame(data = {"Cluster": partitie_opt},
                                  index=instante)
    partitie_opt_t.to_csv("C:\\Users\\MateBook D15\\Downloads\\proiect_dsad_ex1\\PartitieOptima.csv")

    # cu 3 clusteri
    partitie_3 = partitie(h, 3, p, instante)
    print("Partitie 3", partitie_3)

    partitie_3_t = pd.DataFrame(data={"Cluster": partitie_3},
                                  index=instante)
    partitie_3_t.to_csv("C:\\Users\\MateBook D15\\Downloads\\proiect_dsad_ex1\\Partitie3.csv")

    # cu 4 clusteri
    partitie_4 = partitie(h, 4, p, instante)
    print("Partitie 4", partitie_4)

    partitie_4_t = pd.DataFrame(data={"Cluster": partitie_4},
                                  index=instante)
    partitie_4_t.to_csv("C:\\Users\\MateBook D15\\Downloads\\proiect_dsad_ex1\\Partitie4.csv")

    for i in range(3):
        histograma(x[:, i], variabile[i], partitie_opt)

    show()


if __name__ == '__main__':
    execute()