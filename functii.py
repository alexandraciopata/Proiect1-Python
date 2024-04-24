import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import scipy.cluster.hierarchy as hclust
import matplotlib.pyplot as plt


def nan_replace(tabel):
    nume_variabile = list(tabel.columns)
    for each in nume_variabile:
        if any(tabel[each].isna()):
            if is_numeric_dtype(tabel[each]):
                tabel[each].fillna(tabel[each].mean(),
                                   inplace=True)
            else:
                tabel[each].fillna(tabel[each].mode()[0],
                                   inplace=True)


def partitie(h, nr_clusteri, p, instante):
    k_dif_max = p - nr_clusteri
    prag = (h[k_dif_max, 2] + h[k_dif_max + 1, 2]) / 2

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Partitionare cu " +
                 str(nr_clusteri) + " clusteri - metoda Ward")
    hclust.dendrogram(h, labels=instante, ax=ax,
                      color_threshold=prag)

    n = p + 1

    # incercam sa ajungem in pct in care in
    # c[i] = clusterul din care face parte instanta i
    c = np.arange(n)

    for i in range(n - nr_clusteri):
        k1 = h[i, 0]
        k2 = h[i, 1]

        c[c == k1] = n + i
        c[c == k2] = n + i

    coduri = pd.Categorical(c).codes
    return np.array(["c" + str(cod + 1) for cod in coduri])


def histograma(x, variabila, partitia):
    fig = plt.figure(figsize=(9, 9))
    fig.suptitle(" Histograme pt variabila " + variabila)

    clusteri = list(set(partitia))
    dim = len(clusteri)

    axs = fig.subplots(1, dim, sharey=True)

    for i in range(dim):
        ax = axs[i]
        ax.set_xlabel(clusteri[i])
        ax.hist(x[partitia == clusteri[i]], bins=10,
                rwidth=0.9, range=(min(x), max(x)))


def show():
    plt.show()
