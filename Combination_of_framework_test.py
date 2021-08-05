import numpy as np
import matplotlib
import pylab as pl
import pandas
from ae_measure2 import *
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from minisom import MiniSom
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import adjusted_rand_score as ari
import time
from pylab import plot,axis,show,pcolor,colorbar,bone
from hyperopt import fmin, tpe, hp
import time
from hyperopt import Trials, STATUS_OK
from librosa import zero_crossings as zc
import matplotlib.ticker as ticker
from Combination_of_framework_functions import *
'''Hyperparameters'''
sig_len = 1024
explained_var = 0.95
k = 4 # NOTE: number of clusters
kmeans = KMeans(n_clusters=k, n_init=2000, max_iter=2000)
myScaler = MaxAbsScaler()

n_neurons = 9
x = n_neurons
m_neurons = 9
y = m_neurons
max_som_iter = 1000
channel_1=0
channel_2=1

'''Read-in'''
raw = '210316-1_waveforms.txt'
filter = '210316-1_filter.csv'

'''Obtaining Feature Vectors'''

csv = pd.read_csv(filter)
time = np.array(csv.Time)
stress = np.array(csv.Adjusted_Stress_MPa)

en_ch1 = np.array(csv.Energy_ch1)
en_ch2 = np.array(csv.Energy_ch2)
en_ch3 = np.array(csv.Energy_ch3)
en_ch4 = np.array(csv.Energy_ch4)

energies = [en_ch1, en_ch2, en_ch3, en_ch4]  # NOTE: set up energy list to parse by channel

v0, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len)  # S9225
v1, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len)  # S9225
v2, ev = filter_ae(raw, filter, channel_num=2, sig_length=sig_len)  # B1025
v3, ev = filter_ae(raw, filter, channel_num=3, sig_length=sig_len)  # B1025

channels = [v0, v1, v2, v3]  # NOTE: collect waveforms by channel

'''
Cast experiment as vectors
'''

feat_vect_set = []
for i, channel in enumerate(channels):
    energy = energies[i]
    channel_vector = []
    for j, wave in enumerate(channel):
        feature_vector = extract_Chelliah_vect(waveform=wave, energy=energy[j])
        channel_vector.append(feature_vector)  # set of all waveforms from channel as a vector
    feat_vect_set.append(
        channel_vector)  # set of all feature vectors from experiment, index: i,j,k is (channel, feature vector, feature)

dims = np.array(feat_vect_set[0]).shape[1]


'''
Normalize data from 0 to 1
'''
for i, data in enumerate(feat_vect_set):
    feat_vect_set[i] = myScaler.fit_transform(data)

print(feat_vect_set)

''' Hyperparameter Optimization'''

space={
        'sig': hp.uniform('sig',0.001,4),
        'learning_rate': hp.uniform('learning_rate',0.001,4)
}
#Note: To optimize 2 variables, needs to be in a dictionary. Here it is called space.
def som_fn(space):
    sig = space['sig']
    learning_rate = space['learning_rate']
    val = MiniSom(x=x,
                  y=x,
                  input_len=dims,
                  sigma = sig,
                  learning_rate=learning_rate,
                  random_seed=1
                  ).quantization_error(feat_vect_set[channel_1])
    print(val)
    return{'loss':val,'status':STATUS_OK}
trials = Trials()
best=fmin(fn=som_fn,
          space=space,
          algo=tpe.suggest,
          max_evals=1000,
          trials=trials,
          rstate=np.random.RandomState(1))
print('best: {}'.format(best))

for i, trial in enumerate(trials.trials[:2]):
    print(i,trial)

sigma = best['sig']
learning_rate=best['learning_rate']
print("n_neurons: {}\nm_neurons: {}\ninput_len: {}\nsigma {}\nlearning_rate: {}".format(n_neurons,m_neurons,dims,sigma,learning_rate))

'''Make SOM'''
som = MiniSom(n_neurons, m_neurons, dims, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function='gaussian', random_seed=1)
som.train(feat_vect_set[channel_1], max_som_iter, verbose=True)

'''Plot SOM'''
plt.figure(figsize = (9,9))
bone()
pcolor(som.distance_map().T)
colorbar()
#plt.title('Test Plot')
axis([0,som._weights.shape[0],0,som._weights.shape[1]])
show()

ch0_weights = som.get_weights()
ch0_weights = np.reshape(ch0_weights, (n_neurons*m_neurons, dims)) # NOTE: reshpae for kmeans

'''
Cluster SOM weights (high-dimensional representation)
'''
kmeans_A = kmeans.fit(ch0_weights)
print(kmeans_A.inertia_)
A_lads = kmeans_A.labels_ # NOTE: grab labels
A_lads = np.reshape(A_lads, (n_neurons, m_neurons))
ch0_labels = np.zeros(len(feat_vect_set[channel_1]), dtype=int)

for i, feature_vect in enumerate(feat_vect_set[channel_1]): # NOTE: assign original feature vectors to the label that the closest weight vector channels
    winner = som.winner(feature_vect) # NOTE: get winning neurons
    label = A_lads[winner[0]][winner[1]]# NOTE: get label of winning neurons
    ch0_labels[i]=label

''' Hyperparameter Optimization'''

space={
        'sig': hp.uniform('sig',0.001,4),
        'learning_rate': hp.uniform('learning_rate',0.001,4)
}
#Note: To optimize 2 variables, needs to be in a dictionary. Here it is called space.
def som_fn2(space):
    sig = space['sig']
    learning_rate = space['learning_rate']
    val = MiniSom(x=x,
                  y=x,
                  input_len=dims,
                  sigma = sig,
                  learning_rate=learning_rate,
                  random_seed=1
                  ).quantization_error(feat_vect_set[channel_2])
    print(val)
    return{'loss':val,'status':STATUS_OK}
trials = Trials()
best2=fmin(fn=som_fn2,
          space=space,
          algo=tpe.suggest,
          max_evals=1000,
          trials=trials,
          rstate=np.random.RandomState(1))
print('best2: {}'.format(best))

for i, trial in enumerate(trials.trials[:2]):
    print(i,trial)

sigma = best2['sig']
learning_rate=best2['learning_rate']
print("n_neurons: {}\nm_neurons: {}\ninput_len: {}\nsigma {}\nlearning_rate: {}".format(n_neurons,m_neurons,dims,sigma,learning_rate))

'''SOM 2 generate'''
som2 = MiniSom(n_neurons, m_neurons, dims, sigma=sigma, learning_rate=learning_rate,
              neighborhood_function='gaussian',random_seed=1)
som2.train(feat_vect_set[channel_2], max_som_iter, verbose=True)

'''Plot SOM'''
plt.figure(figsize = (9,9))
bone()
pcolor(som2.distance_map().T)
colorbar()
#plt.title('Test Plot')
axis([0,som2._weights.shape[0],0,som2._weights.shape[1]])
show()

ch2_weights = som2.get_weights()
ch2_weights = np.reshape(ch2_weights, (n_neurons*m_neurons, dims)) # NOTE: reshpae for kmeans

'''
Cluster SOM weights (high-dimensional representation)
'''
kmeans_B = kmeans.fit(ch2_weights)
B_lads = kmeans_B.labels_ # NOTE: grab labels
B_lads = np.reshape(B_lads, (n_neurons, m_neurons))
ch2_labels = np.zeros(len(feat_vect_set[channel_2]), dtype=int)

for i, feature_vect in enumerate(feat_vect_set[channel_2]): # NOTE: assign original feature vectors to the label that the closest weight vector channels
    winner = som.winner(feature_vect) # NOTE: get winning neurons
    label = B_lads[winner[0]][winner[1]]# NOTE: get label of winning neurons
    ch2_labels[i]=label

'''
Generate some plots
'''
print(np.unique(ch0_labels))
print(np.unique(ch2_labels))
print(np.unique(A_lads))
print(np.unique(B_lads))
plot_cumulative_AE_labeled(ch0_labels, stress)
plot_cumulative_AE_labeled(ch2_labels,stress)
ARI = ari(ch0_labels, ch2_labels)
print(ARI)

