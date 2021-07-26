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
import skfuzzy as fuzz




if __name__ == "__main__":

    '''
    Set hyperparameters
    '''

    os.chdir('E:/Research/Framework_Comparison')

    sig_len = 1024
    c = 8 # NOTE: number of clusters
    m = 2 # NOTE: reccomended value, used in Chelliah2019
    re_init=500
    explained_var = .95

    experiment = '210330-1'
    fname_raw = experiment+'_waveforms'
    fname_filter = experiment+'_filter'

    max_abs_scaler = MaxAbsScaler() # NOTE: normalize between -1 and 1


    '''
    Read-in and Setup
    '''
    raw = glob.glob('./Raw_Data/'+experiment+'/'+fname_raw+'.txt')[0]
    filter = glob.glob('./Filtered_Data/'+experiment+'/'+fname_filter+'.csv')[0]

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
            feature_vector = extract_Chelliah_vect(waveform=wave, energy=energy[j])
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        vect.append(channel_vector) # set of all waveforms from experiment as vector index: i,j,k

    # NOTE: set of feature vectors by channel
    ch0_X = vect[0]
    ch1_X = vect[1]
    ch2_X = vect[2]
    ch3_X = vect[3]

    feat_vect_set = [ch0_X, ch1_X, ch2_X, ch3_X] # NOTE: needs to be a list for PCA map block

    # NOTE: do rescaling
    for i, data in enumerate(feat_vect_set):
        feat_vect_set[i] = max_abs_scaler.fit_transform(data)


    '''
    Do c-means clustering on channels A,B,C, and D, note this requires transpose of data
    '''

    Au, minjm = my_cmeans(np.array(ch0_X).T, c=c, n_init=re_init, verbose=True)
    A_lads = np.argmax(Au, axis=0) # NOTE: hardens soft labels

    Bu, minjm = my_cmeans(np.array(ch1_X).T, c=c, n_init=re_init, verbose=True)
    B_lads = np.argmax(Bu, axis=0)

    Cu, minjm = my_cmeans(np.array(ch2_X).T, c=c, n_init=re_init, verbose=True)
    C_lads = np.argmax(Cu, axis=0)

    Du, minjm = my_cmeans(np.array(ch3_X).T, c=c, n_init=re_init, verbose=True)
    D_lads = np.argmax(Du, axis=0)


    print('S9225 ARI: ', ari(A_lads,B_lads))
    print('B1025 ARI: ', ari(C_lads, D_lads))
    print('Left ARI: ', ari(A_lads,C_lads))
    print('Right ARI: ', ari(B_lads, D_lads))
    print('S9225-1/B1025-2 ARI: ' , ari(A_lads,D_lads))
    print('S9225-2/B1025-1 ARI: ' , ari(B_lads,C_lads))

    #print(A_lads)


    '''
    Generate some plots
    '''
    #plot_cumulative_AE_labeled(A_lads, stress)
