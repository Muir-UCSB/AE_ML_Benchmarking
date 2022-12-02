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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import MinMaxScaler

exp1 = [[0.9272727272727271, 2.896969696969697, 0.03413458524227864],[0.8878787878787878, 2.0303030303030303, 0.037237390548873865],[0.8878787878787878, 2.187878787878788, 0.05054077161515559],[0.809090909090909, 2.6999999999999997, 0.040747520622268336]]
exp2 = [[0.7696969696969697, 2.4242424242424243, 0.04264894356107533],[0.809090909090909, 2.6606060606060606, 0.045201991022524854],[0.8484848484848484, 2.621212121212121, 0.05278998895580039],[0.8878787878787878, 2.5424242424242425, 0.046646645829104744]]
exp3 = [[0.9272727272727271, 2.8575757575757574, 0.05157526456627252],[0.8484848484848484, 2.306060606060606, 0.05187094257194289],[0.8878787878787878, 1.9909090909090907, 0.048538888719459716],[0.809090909090909, 2.896969696969697, 0.04599348104910329]]

if __name__ == "__main__":



    '''
    Set hyperparameters
    '''

    os.chdir('E:/Research/Framework_Comparison')

    sig_len = 1024
    k = 2 # NOTE: number of clusters
    kmeans = KMeans(n_clusters=k, n_init=20000)
    myScaler = MinMaxScaler()

    n_neurons = 9
    m_neurons = 9
    max_som_iter = 1000

    exp = exp3
    experiment = '210330-1'
    fname_raw = experiment+'_waveforms'
    fname_filter = experiment+'_filter'

    label_set = []


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

    channels = [v0, v1, v2, v3] # NOTE: collect waveforms by channel


    '''
    Cast experiment as vectors
    '''

    feat_vect_set = []
    for i, channel in enumerate(channels):
        energy = energies[i]
        channel_vector = []
        for j, wave in enumerate(channel):
            feature_vector = extract_SOM_vect(waveform=wave, energy=energy[j])
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        feat_vect_set.append(channel_vector) # set of all feature vectors from experiment, index: i,j,k is (channel, feature vector, feature)


    dims = np.array(feat_vect_set[0]).shape[1]

    '''
    Normalize data to [0,1]
    '''
    for i, data in enumerate(feat_vect_set):
        feat_vect_set[i] = myScaler.fit_transform(data)


    # Initialization and training
    # NOTE: dataset has approx 200 data points, roccomendtiaons in 5*sqrt(N), 9x9 is ok
    # NOTE: quantization error is average difference of output samples to winning neurons, https://www.intechopen.com/chapters/69305

    '''
    Do SOM clustering and grab labels
    '''
    for i, data in enumerate(feat_vect_set):
        '''
        train SOM
        '''
        sig = exp[i][0]
        alpha = exp[i][1]
        som = MiniSom(n_neurons, m_neurons, dims, sigma=sig, learning_rate=alpha,
                  neighborhood_function='gaussian')
        som.pca_weights_init(data)
        som.train(data, max_som_iter, verbose=True)
        #print(som.quantization_error(data))
        weights = som.get_weights()
        weights = np.reshape(weights, (n_neurons*m_neurons, dims)) # NOTE: reshpae for kmeans


        '''
        Cluster SOM weights (high-dimensional representation)
        '''
        kmeans = kmeans.fit(weights)
        print(kmeans.inertia_)

        weight_labels = kmeans.labels_ # NOTE: grab labels
        weight_labels = np.reshape(weight_labels, (n_neurons, m_neurons))
        feature_labels = np.zeros(len(data), dtype=int)

        for i, feature_vect in enumerate(data): # NOTE: assign original feature vectors to the label that the closest weight vector channels
            winner = som.winner(feature_vect) # NOTE: get winning neurons
            feature_labels[i] = weight_labels[winner[0]][winner[1]]# NOTE: get label of winning neurons

        label_set.append(feature_labels)




    A_lads = label_set[0]
    B_lads = label_set[1]
    C_lads = label_set[2]
    D_lads = label_set[3]


    print('S9225 ARI: ', ari(A_lads,B_lads))
    print('B1025 ARI: ', ari(C_lads, D_lads))
    print('Left ARI: ', ari(A_lads,C_lads))
    print('Right ARI: ', ari(B_lads, D_lads))
    print('S9225-1/B1025-2 ARI: ' , ari(A_lads,D_lads))
    print('S9225-2/B1025-1 ARI: ' , ari(B_lads,C_lads))

    df = pd.DataFrame({'Stress': stress, 'Ch_A': A_lads, 'Ch_B': B_lads, 'Ch_C': C_lads, 'Ch_D': D_lads})
    df.to_csv(r'SOM_framework_labels.csv')

    '''
    Generate some plots
    '''

    plot_cumulative_AE_labeled(A_lads, stress)
    plot_cumulative_AE_labeled(B_lads, stress)
    plot_cumulative_AE_labeled(C_lads, stress)
    plot_cumulative_AE_labeled(D_lads, stress)
