import numpy as np
import matplotlib
import pylab as pl
import pandas
from ae_measure2 import *
from feature_extraction import *
import glob
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import SpectralClustering

if __name__ == "__main__":

    '''
    Set hyperparameters
    '''

    sig_len = 1024
    k = 2 # NOTE: number of clusters
    kmeans = KMeans(n_clusters=k, n_init=20000)
    #kmeans = SpectralClustering(n_clusters=k, n_init=1000, eigen_solver='arpack'
        #                            ,affinity="nearest_neighbors",  n_neighbors=5)
    reference_index = 3



    my_scaler = StandardScaler() # NOTE: normalize to unit variance


    '''
    Read-in and Setup
    '''
    mypath = 'C:/Research/Framework_Benchmarking/Data/PLB_data.json'
    data = load_PLB(mypath)
    waves = data['data']
    targets = data['target']
    angles = data['target_angle']
    energy = data['energy']





    energy_set = energy[np.where(targets==reference_index)]
    wave_set = waves[np.where(targets==reference_index)]





    '''
    Cast experiment as vectors
    '''
    mean = []
    error = []
    for angle in np.unique(targets):
        wave_set = waves[np.where(targets==angle)]

        vect = []
        for wave in wave_set:
            feature_vector = get_peak_freq(waveform=wave)/1000
            vect.append(feature_vector) # set of all waveforms from channel as a vector


        mean.append(np.mean(vect))
        error.append(np.std(vect))

    deltaTheta = np.array([int(angle[0:2]) for angle in angles])-20
    print(deltaTheta)

    '''
    Make plot
    '''

    colors = np.array(list(islice(cycle(['tab:blue','tab:orange', 'tab:gray','tab:brown',
                                         'tab:red', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']), len(deltaTheta)+1)))
    tick_marks = np.array(list(islice(cycle(['o','>', 's', 'd','v']), len(deltaTheta)+1)))
    lines = np.array(list(islice(cycle(['-','--', '-.', ':']), len(deltaTheta)+1)))


    SMALL_SIZE = 14
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 18
    width = 3
    mark_size = 8

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Peak frequency (kHz)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.set_xlim(-2, max(deltaTheta)+2)
    #ax1.set_ylim(-.01, 1)
    ax1.set_xlabel(r'$\Delta\theta$ (Degrees)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)





    pl.errorbar(deltaTheta, mean, yerr=error, ls='',color='black',fmt='o',mfc='white',linewidth=width,zorder=2,label='')



    #ax1.plot(deltaTheta, cut_off, color='black', linewidth=2)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box') # NOTE: makes plot square
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax1.legend(fontsize=SMALL_SIZE)
    pl.show()
