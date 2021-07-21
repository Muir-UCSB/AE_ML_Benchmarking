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

if __name__ == "__main__":

    '''
    Set hyperparameters
    '''

    os.chdir('C:/Research/Framework_Comparison')

    sig_len = 1024
    explained_var = 0.95
    max_k = 11

    experiment = '210308-1'
    fname_raw = experiment+'_waveforms'
    fname_filter = experiment+'_filter'


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

    channels = [v0, v1, v2, v3] # NOTE: set up waveform list to parse by channel





    '''
    Cast experiment as vectors
    '''

    vect = []
    for i, channel in enumerate(channels):
        energy = energies[i]
        channel_vector = []
        for j, wave in enumerate(channel):
            feature_vector = extract_Moevus_vect(waveform=wave, energy=energy[j])
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        vect.append(channel_vector) # set of all waveforms from experiment as vector index: i,j,k

    # NOTE: set of feature vectors by channel
    ch0_X = vect[0]
    ch1_X = vect[1]
    ch2_X = vect[2]
    ch3_X = vect[3]

    feat_vect_set = [ch0_X, ch1_X, ch2_X, ch3_X]



    '''
    Do PCA mapping on feature vectors by channel
    '''
    mapped_feature_vectors = []
    pca = PCA(explained_var) #Note: determines the number of principal components to explain no less than 0.95 variance

    for i, channel in enumerate(feat_vect_set):
        X = pca.fit_transform(channel)
        eigenvalues = pca.explained_variance_
        mapped_feature_vectors.append(rescale(X, eigenvalues)) # NOTE: do rescaling of feature vectors so distances conform to metric in Moevus2008

    ch0_X = mapped_feature_vectors[0]
    ch1_X = mapped_feature_vectors[1]
    ch2_X = mapped_feature_vectors[2]
    ch3_X = mapped_feature_vectors[3]


    '''
    Grab DBI scores, re-initializations of kmeans auto sets to 200
    '''
    n_clust, DBI_ch0 = get_DBI(ch0_X, max_k)
    _, DBI_ch1 = get_DBI(ch1_X, max_k)
    _, DBI_ch2 = get_DBI(ch2_X, max_k)
    _, DBI_ch3 = get_DBI(ch3_X, max_k)







    MEDIUM_SIZE = 14
    LARGE_SIZE = 16
    width = 2
    fig, ax1 = pl.subplots()

    ax1.set_ylabel('Davies-Bouldin Index', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(.1))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))


    ax1.set_xlabel('Number of Clusters', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)

    ax1.plot(n_clust, DBI_ch0, linewidth=width, label = 'Channel A')
    ax1.plot(n_clust, DBI_ch1, linewidth=width, label = 'Channel B')
    ax1.plot(n_clust, DBI_ch2, linewidth=width, label = 'Channel C')
    ax1.plot(n_clust, DBI_ch3, linewidth=width, label = 'Channel D')
    pl.legend()
    pl.title('Experiment '+experiment, fontsize=LARGE_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.show()
