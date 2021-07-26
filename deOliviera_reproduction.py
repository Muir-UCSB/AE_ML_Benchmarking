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
from minisom import MiniSom

if __name__ == "__main__":



    '''
    Set hyperparameters
    '''

    os.chdir('E:/Research/Framework_Comparison')

    sig_len = 1024
    explained_var = 0.95
    k = 5 # NOTE: number of clusters
    kmeans = KMeans(n_clusters=k, n_init=200)

    n_neurons = 9
    m_neurons = 9
    max_som_iter = 1000
    sig = 3 # NOTE: neighborhood function
    alpha = .5 # NOTE: learning rate

    fname_raw = '210308-1_waveforms'
    fname_filter = '210308-1_filter'


    '''
    Read-in and Setup
    '''
    raw = glob.glob("./Raw_Data/210308-1/"+fname_raw+".txt")[0]
    filter = glob.glob("./Filtered_Data/210308-1/"+fname_filter+".csv")[0]

    csv = pd.read_csv(filter)
    time = np.array(csv.Time)
    stress = np.array(csv.Adjusted_Stress_MPa)

    en_ch1 = np.array(csv.Energy_ch1)
    en_ch2 = np.array(csv.Energy_ch2)
    en_ch3 = np.array(csv.Energy_ch3)
    en_ch4 = np.array(csv.Energy_ch4)

    energies = [en_ch1, en_ch2, en_ch3, en_ch4] # NOTE: set up energy list to parse by channel


    v0, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len) # S9225
    v1, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len) # S9225
    v2, ev = filter_ae(raw, filter, channel_num=2, sig_length=sig_len) # B1025
    v3, ev = filter_ae(raw, filter, channel_num=3, sig_length=sig_len) # B1025

    channels = [v0, v1, v2, v3] # NOTE: collect waveforms by channel


    '''
    Cast experiment as vectors
    '''

    feat_vect_set = []
    for i, channel in enumerate(channels):
        energy = energies[i]
        channel_vector = []
        for j, wave in enumerate(channel):
            feature_vector = extract_Moevus_vect(waveform=wave, energy=energy[j])
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        feat_vect_set.append(channel_vector) # set of all feature vectors from experiment, index: i,j,k is (channel, feature vector, feature)




    '''
    # TODO: normalize data to 0 mean and unit variance
    '''


    '''
    No PCA
    '''




    dims = np.array(feat_vect_set[0]).shape[1] # TODO: rename to ch0_dims


    # Initialization and training
    # NOTE: dataset has approx 200 data points, roccomendtiaons in 5*sprt(N), 8x8 is ok
    # NOTE: quantization error is average difference of output samples to winning neurons, https://www.intechopen.com/chapters/69305

    '''
    train SOM
    '''

    som = MiniSom(n_neurons, m_neurons, dims, sigma=sig, learning_rate=alpha,
              neighborhood_function='gaussian')
    som.pca_weights_init(feat_vect_set[0])
    som.train(feat_vect_set[0], max_som_iter, verbose=True)

    ch0_weights = som.get_weights()
    ch0_weights = np.reshape(ch0_weights, (n_neurons*m_neurons, dims)) # NOTE: reshpae for kmeans




    '''
    Cluster SOM weights (high-dimensional representation)
    '''
    kmeans_A = kmeans.fit(ch0_weights)
    A_lads = kmeans_A.labels_ # NOTE: grab labels
    A_lads = np.reshape(A_lads, (n_neurons, m_neurons))
    ch0_labels = np.zeros(len(feat_vect_set[0]), dtype=int)

    for i, feature_vect in enumerate(feat_vect_set[0]): # NOTE: assign original feature vectors to the label that the closest weight vector channels
        winner = som.winner(feature_vect) # NOTE: get winning neurons
        label = A_lads[winner[0]][winner[1]]# NOTE: get label of winning neurons
        ch0_labels[i]=label

    plot_cumulative_AE_labeled(ch0_labels, stress)
