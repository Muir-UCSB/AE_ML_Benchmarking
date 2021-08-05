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

def plot_cumulative_AE_labeled(label_set, stress, colormap = 'Set1', show=True, save_as=None):
    '''
    Generates an AE vs. Stress plots and separates the events by cluster.
    # TODO: add functionality for custom colormap input
    labels (array-like): List of labels assigned to each AE event
    stress (array-like): List of stresses that each AE event occurs at
    show (bool): If true then plot will show
    save_as (str): if path is entered as file_name.png then the figure will save
    return:    None
    '''

    labels = np.unique(label_set)
    num_events = len(label_set)
    ymax=0

    '''
    Plotting parameters
    '''
    MEDIUM_SIZE = 14
    cmap = plt.cm.get_cmap(colormap, len(labels)) # NOTE: make colormap with number of elements equal to number of clusters

    fig, ax1 = pl.subplots()

    ax1.set_ylabel('Cumulative Number AE', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(.1))


    ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    for i, label in enumerate(labels):
        labeled_stress = stress[np.where(label_set==label)]
        cumulative_num = (np.arange(len(labeled_stress))+1)/num_events # NOTE: +1 indexes from 1
        ax1.scatter(labeled_stress, cumulative_num, color=str(matplotlib.colors.rgb2hex(cmap(i))),
            label='Cluster '+str(i+1))
        if max(cumulative_num) > ymax:
            ymax = max(cumulative_num)

    pl.ylim([0, ymax+ymax*.05])
    pl.legend()

    if show == True:
        pl.show()

    if type(save_as) is str:
        pl.savefig(save_as)


def SOMDBIplot(feat_vect, n_neurons, m_neurons, sigma = 0.95896, learning_rate = 2.709,max_som_iter=1000):
    '''Takes a feature vecture and makes a DBI plot based on an SOM'''
    dims = np.array(feat_vect).shape[1]
    som = MiniSom(n_neurons, m_neurons, dims, sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian')
    som.pca_weights_init(feat_vect)
    som.train(feat_vect, max_som_iter, verbose=True)
    weights = som.get_weights()
    weights = np.reshape(weights, (n_neurons * m_neurons, dims))  # NOTE: reshape for kmeans
    clusterarray = list(range(2, 11)) #Note: array of clusters [2 3 ... 10]
    dbscores = []

    for i in clusterarray:
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter= 300).fit(weights)
        lads = kmeans.labels_
        lads = np.reshape(lads, (n_neurons, m_neurons))
        ch_labels = np.zeros(len(feat_vect), dtype=int)

        for i, feature_vect in enumerate(feat_vect):  # NOTE: assign original feature vectors to the label that the closest weight vector channels
            winner = som.winner(feature_vect)  # NOTE: get winning neurons
            label = lads[winner[0]][winner[1]]  # NOTE: get label of winning neurons
            ch_labels[i] = label

        davies_bouldin_score1 = davies_bouldin_score(feat_vect, ch_labels)
        dbscores.append(davies_bouldin_score1)
    plt.plot(clusterarray,dbscores)
    plt.show()

def SOMiteration_inertia(feat_vect, n_neurons, m_neurons, sigma = 0.95896, learning_rate = 2.709,max_som_iter=1000):
    '''Takes a feature vecture and makes a DBI plot based on an SOM'''
    dims = np.array(feat_vect).shape[1]
    som = MiniSom(n_neurons, m_neurons, dims, sigma=sigma, learning_rate=learning_rate,
                  neighborhood_function='gaussian')
    som.pca_weights_init(feat_vect)
    som.train(feat_vect, max_som_iter, verbose=True)
    weights = som.get_weights()
    weights = np.reshape(weights, (n_neurons * m_neurons, dims))  # NOTE: reshape for kmeans
    inertia = []

    for i in range(1,1000):
        kmeans = KMeans(n_clusters=10, n_init=1000, max_iter= i).fit(weights)
        inertia.append(kmeans.inertia_)
        lads = kmeans.labels_
        lads = np.reshape(lads, (n_neurons, m_neurons))
        ch_labels = np.zeros(len(feat_vect), dtype=int)

        for i, feature_vect in enumerate(feat_vect):  # NOTE: assign original feature vectors to the label that the closest weight vector channels
            winner = som.winner(feature_vect)  # NOTE: get winning neurons
            label = lads[winner[0]][winner[1]]  # NOTE: get label of winning neurons
            ch_labels[i] = label
    print(inertia)
    plt.plot(inertia)
    plt.show()

def get_peak_freq(waveform, dt=10**-7, low=None, high=None):
    '''
    Gets peak frequency of signal
    waveform (array-like): Voltage time series of the waveform
    dt (float): Time between samples (s) (also inverse of sampling rate)
    return
    peak_freq (float): frequency of maximum FFT power in Hz
    '''
    w, z = fft(dt, waveform, low_pass=low, high_pass=high)
    max_index = np.argmax(z)
    peak_freq = w[max_index]
    return peak_freq

def get_counts(waveform, threshold=0.1):
    '''
    Gets number of counts in AE signal, equiv to number of zero crossings
    waveform (array-like): voltage time-series of the waveform
    threshold (float): Floating threshold that defines the start and end of signal
    return
    counts (int): Average frequency of signal in Hz
    '''
    imin, imax = get_signal_start_end(waveform)
    cut_signal = waveform[imin:imax]
    num_zero_crossings = len(np.nonzero(zc(cut_signal)))
    return num_zero_crossings

def extract_Chelliah_vect(waveform=[], dt=10**-7, energy=None, threshold=.1):
    '''
    Extracts features from a waveform according to Chelliah2019. Note for the peak frequency
    calculation we choose to only consider freqeuncies from 300 kHz to 1000 kHz because that is
    what the s9225 sensors are calibrated for. We can justify this because if a waveform travels the
    same path from its source to sensor the transfer functions for the sensor with the narrowest
    calibration range will be the transfer functions for the other sensor, at that frequency. Therefore,
    the peak frequency of both sensors will match.
    waveform (array-like): Voltage time series describing the waveform
    dt(float): time between samples (s)
    energy (float): energy of the waveform as calculated by the AE software
    low_pass (float): lower bound on bandpass (Hz)
    high_pass (float): upper bound on bandpass (Hz)
    return:
    vect (array-like): feature vector extracted from a waveform according to Chelliah2019
    '''
    if waveform == [] or energy == None:
        raise ValueError('An input is missing')

    low = 300*10**3
    high = 1000*10**3

    max_amp = np.max(waveform)
    peak_time = np.argmax(waveform)/10 # NOTE: time of signal peak in microseconds

    imin, imax = get_signal_start_end(waveform)
    start_time = imin/10 #Note: converts index location to a start time (microseconds)
    end_time = imax/10 #Note: converts index location to an end time (microseconds)

    rise_time = peak_time - start_time
    duration = end_time-start_time
    counts = get_counts(waveform)
    peak_freq = get_peak_freq(waveform, dt=dt, low=low, high=high)



    feature_vector = [max_amp, rise_time, counts, energy, duration, peak_freq]
    return feature_vector
