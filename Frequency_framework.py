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

if __name__ == "__main__":

    '''
    Set hyperparameters
    '''

    os.chdir('E:/Research/Framework_Comparison')

    sig_len = 1024
    k = 4 # NOTE: number of clusters
    kmeans = KMeans(n_clusters=k, n_init=20000)

    experiment = '210330-1'
    fname_raw = experiment+'_waveforms'
    fname_filter = experiment+'_filter'

    my_scaler = StandardScaler() # NOTE: normalize to unit variance


    '''
    Read-in and Setup
    '''
    raw = glob.glob('./Raw_Data/'+experiment+'/'+fname_raw+'.txt')[0]
    filter = glob.glob('./Filtered_Data/'+experiment+'/'+fname_filter+'.csv')[0]

    csv = pd.read_csv(filter)
    time = np.array(csv.Time)
    stress = np.array(csv.Adjusted_Stress_MPa)

    v0, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len) # S9225
    v1, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len) # S9225
    v2, ev = filter_ae(raw, filter, channel_num=2, sig_length=sig_len) # B1025
    v3, ev = filter_ae(raw, filter, channel_num=3, sig_length=sig_len) # B1025

    channels = [v0, v1, v2, v3] # NOTE: set up waveform list to parse by channel



    '''
    Cast experiment as vectors
    '''

    feat_vect_set = []
    for i, channel in enumerate(channels):
        channel_vector = []
        for j, wave in enumerate(channel):
            feature_vector = extract_Sause_vect(waveform=wave)
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        feat_vect_set.append(channel_vector) # set of all waveforms from experiment as vector index: i,j,k


    # NOTE: do rescaling
    for i, data in enumerate(feat_vect_set):
        feat_vect_set[i] = my_scaler.fit_transform(data)





    '''
    Do k-means clustering on channels A,B,C, and D
    '''

    clustered_channels = []
    for channel_data in feat_vect_set:
        clustered_channels.append(kmeans.fit(channel_data).labels_)
        print(kmeans.n_iter_)



    A_lads = clustered_channels[0] # NOTE: labels
    B_lads = clustered_channels[1]
    C_lads = clustered_channels[2]
    D_lads = clustered_channels[3]

    print('S9225 ARI: ', ari(A_lads,B_lads))
    print('B1025 ARI: ', ari(C_lads, D_lads))
    print('Left ARI: ', ari(A_lads,C_lads))
    print('Right ARI: ', ari(B_lads, D_lads))
    print('S9225-1/B1025-2 ARI: ' , ari(A_lads,D_lads))
    print('S9225-2/B1025-1 ARI: ' , ari(B_lads,C_lads))



    df = pd.DataFrame({'Stress': stress, 'Ch_A': A_lads, 'Ch_B': B_lads, 'Ch_C': C_lads, 'Ch_D': D_lads})
    df.to_csv(r'Frequency_framework_labels.csv')

    '''
    Generate some plots
    '''

    plot_cumulative_AE_labeled(A_lads, stress)
    plot_cumulative_AE_labeled(B_lads, stress)
    plot_cumulative_AE_labeled(C_lads, stress)
    plot_cumulative_AE_labeled(D_lads, stress)
