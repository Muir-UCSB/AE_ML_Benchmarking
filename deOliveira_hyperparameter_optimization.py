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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ari
import time
from pylab import plot,axis,show,pcolor,colorbar,bone
from hyperopt import fmin, tpe, hp
import time
from hyperopt import Trials, STATUS_OK
from librosa import zero_crossings as zc

'''Hyperparameters'''
sig_len = 1024
explained_var = 0.95
k = 5 # NOTE: number of clusters
kmeans = KMeans(n_clusters=k, n_init=200)
myScaler = StandardScaler()

n_neurons = 9
x = n_neurons
m_neurons = 9
y = m_neurons
max_som_iter = 1000

'''Read-in'''
raw = '210308-1_waveforms.txt'
filter = '210308-1_filter.csv'

'''Feature Vector extraction Function'''
def get_average_freq2(waveform, dt=10**-7, threshold=0.1):
    '''
    Gets average frequency defined as the number of zero crossings
    divided by the length of the signal according to Moevus2008
    waveform (array-like): voltage time-series of the waveform
    dt (float): sampling rate (s)
    threshold (float): Floating threshold that defines the start and end of signal
    return
    average_frequency (float): Average frequency of signal in Hz
    '''
    imin, imax = get_signal_start_end(waveform, threshold=threshold)
    cut_signal = waveform[imin:imax]
    num_zero_crossings = len(np.nonzero(zc(cut_signal)))
    return num_zero_crossings/(len(cut_signal)*dt)

def get_signal_start_end(waveform, threshold=0.1):
    '''
    Gets indicies of the signal start and end defined by a floating threshold
    waveform (array-like): Voltage time series of the waveform
    threshold (float): floating threshold that defines signal start and end
    return
    start_index, end_index (int): Array index of signal start and signal end respectively
    '''
    if threshold<0 or threshold>1:
        raise ValueError('Threshold must be between 0 and 1')

    max_amp = np.max(waveform)
    start_index, end_index = np.nonzero(waveform > threshold*max_amp)[0][[0, -1]]
    return start_index, end_index

def extract_Moevus_vect(waveform=[], dt=10**-7, energy=None, threshold=.1):
    '''
    waveform (array-like): Voltage time series describing the waveform
    dt(float): sampling rate (s)
    energy (float): energy of the waveform as calculated by the AE software
    low_pass (float): lower bound on bandpass (Hz)
    high_pass (float): upper bound on bandpass (Hz)
    return:
    vect (array-like): feature vector extracted from a waveform according to Moevus2008
    '''
    if waveform == [] or energy == None:
        raise ValueError('An input is missing')

    ln_energy = np.log(energy)
    max_amp = np.max(waveform)
    peak_time = np.argmax(waveform)/10 # NOTE: time of signal peak in microseconds

    imin, imax = get_signal_start_end(waveform)
    start_time = imin/10 #Note: converts index location to a start time (microseconds)
    end_time = imax/10 #Note: converts index location to an end time (microseconds)

    rise_time = peak_time - start_time
    duration = end_time-start_time
    decay_time = end_time-peak_time

    risingpart = waveform[imin:np.argmax(waveform)] #Note: grabs the rising portion of the waveform

    average_freq= get_average_freq2(waveform, dt=dt, threshold=threshold)#Note: is this new???
    rise_freq = get_average_freq2(risingpart, dt, threshold=threshold)

    log_risetime = np.log(rise_time)
    log_rd = np.log(rise_time/duration)
    log_ar = np.log(max_amp/rise_time)
    log_ad = np.log(max_amp/decay_time)
    log_af = np.log(max_amp/average_freq)


    feature_vector = [log_risetime, average_freq, rise_freq, ln_energy, log_rd, log_ar, log_ad, log_af]
    return feature_vector

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
        feature_vector = extract_Moevus_vect(waveform=wave, energy=energy[j])
        channel_vector.append(feature_vector)  # set of all waveforms from channel as a vector
    feat_vect_set.append(
        channel_vector)  # set of all feature vectors from experiment, index: i,j,k is (channel, feature vector, feature)

dims = np.array(feat_vect_set[0]).shape[1]

'''
Normalize data to 0 mean and unit variance
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
                  ).quantization_error(data)
    print(val)
    return{'loss':val,'status':STATUS_OK}
trials = Trials()
best=fmin(fn=som_fn,
          space=space,
          algo=tpe.suggest,
          max_evals=10000,
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
              neighborhood_function='gaussian')
som.pca_weights_init(feat_vect_set[0])
som.train(feat_vect_set[0], max_som_iter, verbose=True)

'''Plot SOM'''
plt.figure(figsize = (9,9))
bone()
pcolor(som.distance_map().T)
colorbar()
plt.title('Test Plot')
axis([0,som._weights.shape[0],0,som._weights.shape[1]])
show()

