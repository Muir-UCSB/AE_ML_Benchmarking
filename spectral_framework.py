'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 210518

This script takes in a file path with one experiment as text file
It clusters with k=2


'''

#Imports
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
import glob
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
from scipy.cluster.vq import whiten
import pandas as pd
import os
from scipy.cluster.vq import whiten
from scipy.integrate import simps
import time as tm



def get_match_rate(label_1, label_2):
    acc = 0
    for i, label in enumerate(label_1):
        if label == label_2[i]:
            acc = acc+1
    acc = acc/len(label_1)
    if acc <.5:
        return 1-acc
    else:
        return acc



if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    sig_len = 1024
    os.chdir('E:/Research/Framework_Benchmarking')
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
    Parameters
    '''
    k = 2
    NN = 5

    dt = 10**-7 #s
    Low = 00*10**3 #Hz
    High = 450*10**3 #Hz

    num_bins = np.arange(2,30)
    print(num_bins)





    FFT_units = 1000 #FFT outputs in Hz, this converts to kHz

    spectral = SpectralClustering(n_clusters=k, n_init=100, eigen_solver='arpack'
                                    ,affinity="nearest_neighbors",  n_neighbors=NN)

    '''
    Cast experiment as vectors
    '''
    for bins in num_bins:
        vect = []
        for waveform in wave_set:
            feature_vector, freq_bounds, spacing = wave2vec(dt, waveform, Low, High, bins, FFT_units)
            vect.append(feature_vector) # set of all waveforms from channel as a vector



        # Cluster waveform
        spectral = spectral.fit(vect)
        labels = spectral.labels_




        print(bins, 'ARI: ', ari(labels, ground_truth))
