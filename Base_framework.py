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
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import adjusted_rand_score as ari


if __name__ == "__main__":

    '''
    Set hyperparameters
    '''

    os.chdir('E:/Research/Framework_Benchmarking')

    sig_len = 1024
    explained_var = 0.95
    k = 3 # NOTE: number of clusters
    kmeans = KMeans(n_clusters=k, n_init=20000)
    max_abs_scaler = MaxAbsScaler() # NOTE: normalize between -1 and 1


    '''
    Read-in and Setup
    '''
    sig_len = 1024
    datapath = 'E:/Research/Framework_Benchmarking/Data/PLB_data.json'

    ref_index = 0 # NOTE: 20 degree has label 0
    exp_index = 3 # NOTE: 26 degree has label 1, subject to change

    data = load_PLB(datapath)

    waves = data['data']
    target = data['target']
    angles = data['target_angle']
    energy = data['energy']

    reference_waves = waves[np.where(target==ref_index)]
    reference_labels = target[np.where(target==ref_index)]
    reference_energy = energy[np.where(target==ref_index)]

    experiment_waves = waves[np.where(target==exp_index)]
    experiment_labels = target[np.where(target==exp_index)]
    experiment_energy = energy[np.where(target==exp_index)]

    wave_set = np.vstack((reference_waves, experiment_waves))
    ground_truth = np.hstack((reference_labels, experiment_labels))
    energy_set = np.hstack((reference_energy,experiment_energy))



    '''
    Cast experiment as vectors
    '''

    vect = []
    for i, wave in enumerate(wave_set):
        feature_vector = extract_Moevus_vect(waveform=wave, energy=energy_set[i])
        vect.append(feature_vector) # set of all waveforms from channel as a vector


    # NOTE: do rescaling
    vect = max_abs_scaler.fit_transform(vect)


    '''
    Do PCA mapping on feature vectors and normalize by channel
    '''
    pca = PCA(explained_var) #Note: determines the number of principal components to explain no less than 0.95 variance


    X = pca.fit_transform(vect)
    eigenvalues = pca.explained_variance_
    vect = Moevus_rescale(X, eigenvalues) # NOTE: do rescaling of feature vectors so distances conform to metric in Moevus2008


    '''
    Do k-means clustering on channels A,B,C, and D
    '''
    print('Beginning clustering')
    labels = kmeans.fit(vect).labels_
    print(kmeans.n_iter_)



    print('ARI: ', ari(labels, ground_truth))
