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
    k = 2 # NOTE: number of clusters
    kmeans = KMeans(n_clusters=k, n_init=20000)
    n_drop = 3 #number of features to drop
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


    for wave in wave_set:
        feature_vector, leaf_names = get_wpt_energies(waveform=wave)
        vect.append(feature_vector) # set of all waveforms from channel as a vector



    vect = pd.DataFrame(vect, columns = leaf_names)


    '''
    Drop n_drop most correlated features
    '''
    dropped_labels = []
    for j in range(n_drop):
        corr_matrix = vect.corr().abs()
        #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
        sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                          .stack()
                          .sort_values(ascending=False)) #first element of sol series is the pair with the biggest correlation
        label_to_drop = sol.index[0][1] # NOTE: grabs label of first element
        vect.drop(labels = label_to_drop, axis=1, inplace=True)
        dropped_labels.append(label_to_drop)
    vect = vect.to_numpy().tolist()

    print(dropped_labels)





    '''
    Add total energy as calculated by AE aquisition system to features
    '''

    for i, energy in enumerate(energy_set):
        vect[i].append(energy)





    '''
    Normalize then do PCA mapping on feature vectors and normalize by channel
    '''
    pca = PCA(explained_var) #Note: determines the number of principal components to explain no less than 0.95 variance

    vect = pca.fit_transform(max_abs_scaler.fit_transform(vect))
    eigenvalues = pca.explained_variance_
    vect = Moevus_rescale(vect, eigenvalues) # NOTE: do rescaling of feature vectors so distances conform to metric in Moevus2008



    '''
    Do k-means clustering on channels A,B,C, and D
    '''
    print('Beginning clustering')

    labels = kmeans.fit(vect).labels_
    #print(kmeans.n_iter_)

    print('ARI: ', ari(labels, ground_truth))
