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

    os.chdir('E:/Research/Framework_Comparison')

    sig_len = 1024
    explained_var = 0.95
    k = 2 # NOTE: number of clusters
    kmeans = KMeans(n_clusters=k, n_init=20000)
    n_drop = 3 #number of features to drop

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
        channel_vector = []
        for j, wave in enumerate(channel):
            feature_vector, leaf_names = get_wpt_energies(waveform=wave)
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        vect.append(channel_vector) # set of all waveforms from experiment as vector index: i,j,k

    # NOTE: set of feature vectors by channel
    ch0_X = vect[0]
    ch1_X = vect[1]
    ch2_X = vect[2]
    ch3_X = vect[3]



    feat_vect_set = [pd.DataFrame(ch0_X, columns = leaf_names),
        pd.DataFrame(ch1_X, columns = leaf_names),
        pd.DataFrame(ch2_X, columns = leaf_names),
        pd.DataFrame(ch3_X, columns = leaf_names)]


    '''
    Drop n_drop most correlated features
    '''
    dropped_labels = []
    for i, channel in enumerate(feat_vect_set):
        for j in range(n_drop):
            corr_matrix = channel.corr().abs()
            #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
            sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                              .stack()
                              .sort_values(ascending=False)) #first element of sol series is the pair with the biggest correlation
            label_to_drop = sol.index[0][1] # NOTE: grabs label of first element
            channel.drop(labels = label_to_drop, axis=1, inplace=True)
            dropped_labels.append(label_to_drop)
        feat_vect_set[i] = channel.to_numpy().tolist()

    dropped_labels = np.reshape(dropped_labels, (len(feat_vect_set), n_drop))




    '''
    Add total energy as calculated by AE aquisition system to features
    '''
    for i, channel in enumerate(feat_vect_set):
        energy = energies[i]
        for j, vector in enumerate(channel):
            vector.append(energy[j])




    '''
    Normalize then do PCA mapping on feature vectors and normalize by channel
    '''
    pca = PCA(explained_var) #Note: determines the number of principal components to explain no less than 0.95 variance
    for i, data in enumerate(feat_vect_set):
        feat_vect_set[i] = pca.fit_transform(max_abs_scaler.fit_transform(data))
        eigenvalues = pca.explained_variance_
        feat_vect_set[i] = Moevus_rescale(feat_vect_set[i], eigenvalues) # NOTE: do rescaling of feature vectors so distances conform to metric in Moevus2008



    '''
    Do k-means clustering on channels A,B,C, and D
    '''
    clustered_channels = []
    for channel_data in feat_vect_set:
        clustered_channels.append(kmeans.fit(channel_data).labels_)
        print(kmeans.inertia_)
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
    df.to_csv(r'WPT_framework_labels.csv')

    '''
    Generate some plots
    '''

    plot_cumulative_AE_labeled(A_lads, stress)
    plot_cumulative_AE_labeled(B_lads, stress)
    plot_cumulative_AE_labeled(C_lads, stress)
    plot_cumulative_AE_labeled(D_lads, stress)
