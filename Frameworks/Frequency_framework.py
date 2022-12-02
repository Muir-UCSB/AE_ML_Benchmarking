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
    reference_index = 1
    test_index = 3



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



    reference_energies = energy[np.where(targets==reference_index)]
    test_energies = energy[np.where(targets==test_index)]
    energy_set = np.hstack((reference_energies, test_energies))

    reference_waves = waves[np.where(targets==reference_index)]
    test_waves = waves[np.where(targets==test_index)]
    wave_set = np.vstack((reference_waves, test_waves))

    reference_targets  = targets[np.where(targets==reference_index)]
    test_targets  = targets[np.where(targets==test_index)]
    target_set = np.hstack((reference_targets, test_targets))





    '''
    Cast experiment as vectors
    '''

    vect = []

    for wave in wave_set:
        feature_vector = extract_Sause_vect(waveform=wave)
        vect.append(feature_vector) # set of all waveforms from channel as a vector


    # NOTE: do rescaling
    vect = my_scaler.fit_transform(vect)

    '''
    vect = np.array(vect)

    x = vect.T[0]
    y = vect.T[1]
    pl.scatter(x,y,c=target_set)
    pl.show()
    '''



    '''
    Do k-means clustering and get labels
    '''
    print('Beginning clustering')
    labels = kmeans.fit(vect).labels_
    #print(kmeans.n_iter_)


    print('ARI: ', ari(labels,target_set))



    #df = pd.DataFrame({'Stress': stress, 'Ch_A': A_lads, 'Ch_B': B_lads, 'Ch_C': C_lads, 'Ch_D': D_lads})
    #df.to_csv(r'Frequency_framework_labels.csv')
