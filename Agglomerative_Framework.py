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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ari


if __name__ == "__main__":

    '''
    Set hyperparameters
    '''

    os.chdir('E:/Research/Framework_Benchmarking')

    sig_len = 1024
    k = 2 # NOTE: number of clusters
    agglomerative = AgglomerativeClustering(linkage='ward', n_clusters=k)


    '''
    Read-in and Setup
    '''
    sig_len = 1024
    datapath = 'E:/Research/Framework_Benchmarking/Data/PLB_data.json'

    ref_index = 0 # NOTE: 20 degree has label 0
    exp_index = 1 # NOTE: 26 degree has label 1, subject to change

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
    for wave in wave_set:
        feature_vector = extract_agglomerative_vect(waveform=wave)
        vect.append(feature_vector) # set of all waveforms from channel as a vector





    '''
    Do agglomerative clustering
    '''
    print('Beginning clustering')
    labels = agglomerative.fit(vect).labels_



    print('ARI: ', ari(labels, ground_truth))
