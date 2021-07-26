'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 210720

'''

import numpy as np
import matplotlib
import pandas
from ae_measure2 import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import pylab as pl
import matplotlib.ticker as ticker
from librosa import zero_crossings as zc
import skfuzzy as fuzz
import pywt

def get_wpt_energies(waveform=[], wavelet='db2', mode='zero', maxlevel=3):
    '''
    Extracts partial energies from each leaf of wavelet packet transform, as put forth by
    Maillet2014

    wave (array-like): single waveform voltage array
    wavelet (str): Mother wavelet, see pywt for other wavelets
    mode (str): signal extension mode, affects artifacts at decomposition edges.
                Choice has not been found to affect results.
    maxlevel (int): Wavelet decomposition maxlevel

    return:
    partial_energies (array-like): normalized energy of leaf nodes, sums to 1.
    '''
    E_tot = np.sum(np.power(waveform,2))

    wp = pywt.WaveletPacket(data=waveform, wavelet = 'db2', mode = 'zero', maxlevel=3)
    leaf_names = [n.path for n in wp.get_leaf_nodes(True)]
    decomposed_signals = [wp[path].data for path in leaf_names]

    partial_energies = [np.sum(np.power(decomp,2))/E_tot for decomp in decomposed_signals]

    return partial_energies, leaf_names



def my_cmeans(data, c=2, m=2, error=.000005, max_iter=1000, n_init=500, verbose=False):
    '''
    c_means with multiple restarts. Inputs inherited from parent function.

    return:
    u (array-like): u-matrix associated with lowest objective function evaluation
    minjm (float): lowest objective function evaluation
    '''
    jm_history =[]
    u_history = []
    for i in range(n_init):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data, c=c, m=2, error=error, maxiter=max_iter) # NOTE: jm is objective function history
        jm_history.append(jm[-1])
        u_history.append(u)
        if verbose==True:
            print('Restart ',i+1,'/',n_init)

    u = u_history[np.argmin(jm_history)]
    return u, min(jm_history)

def get_peak_freq(waveform, dt=10**-7, low=None, high=None):
    '''
    Gets peak frequency of signal

    waveform (array-like): Voltage time series of the waveform
    dt (float): Sampling rate of transducers

    return
    peak_freq (float): frequency of maximum FFT power in Hz
    '''
    w, z = fft(dt, waveform, low_pass=low, high_pass=high)
    max_index = np.argmax(z)
    peak_freq = w[max_index]

    return peak_freq

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

def get_average_freq(waveform, dt=10**-7, threshold=0.1):
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

    average_freq= get_average_freq(waveform, dt=dt, threshold=threshold)
    rise_freq = get_average_freq(risingpart, dt=dt, threshold=threshold)

    log_risetime = np.log(rise_time)
    log_rd = np.log(rise_time/duration)
    log_ar = np.log(max_amp/rise_time)
    log_ad = np.log(max_amp/decay_time)
    log_af = np.log(max_amp/average_freq)


    feature_vector = [log_risetime, average_freq, rise_freq, ln_energy, log_rd, log_ar, log_ad, log_af]
    return feature_vector

def extract_Chelliah_vect(waveform=[], dt=10**-7, energy=None, threshold=.1):
    '''
    Extracts features from a waveform according to Chelliah2019. Note for the peak frequency
    calculation we choose to only consider freqeuncies from 300 kHz to 1000 kHz because that is
    what the s9225 sensors are calibrated for. We can justify this because if a waveform travels the
    same path from its source to sensor the transfer functions for the sensor with the narrowest
    calibration range will be the transfer functions for the other sensor, at that frequency. Therefore,
    the peak frequency of both sensors will match.

    waveform (array-like): Voltage time series describing the waveform
    dt(float): sampling rate (s)
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

def Moevus_rescale(feat_vect, eigenvalues):
    '''
    rescales a set of feature vectors under PCA mapping so distnaces conform to Moevus2008

    feat_vects (array-like): An array of feature vectors under the image of a PCA transform
    eigenvalues (array-like): An array of eigenvalues that corresponds to the principal components

    return:    rescaled_vects (array-like): rescaled feature vectors
    '''
    rescaled_vects = []
    feat_vect = feat_vect.T

    for i, value in enumerate(eigenvalues):
        rescaled_vects.append(feat_vect[i]*np.sqrt(value))

    return np.array(rescaled_vects).T

def plot_cumulative_AE(stress, color='black', show=True, save_as=False):
    '''
    Plots cumulative AE in the stress domain

    stress (array-like): List of stresses that each AE event occurs at

    return:    None
    '''

    num_events = len(stress)
    '''
    Plotting parameters
    '''
    MEDIUM_SIZE = 14
    ymax = 1
    fig, ax1 = pl.subplots()

    ax1.set_ylabel('Cumulative Number AE', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(.1))


    ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    cumulative_num = (np.arange(len(stress))+1)/num_events
    ax1.scatter(stress, cumulative_num, color=color)
    pl.ylim([-.01, ymax+ymax*.05])


    if show == True:
        pl.show()

    if type(save_as) is str:
        pl.savefig(save_as)

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

def get_DBI(feat_vect, max_clust=int):
    '''
    Grabs DBI as clustered via kmeans as a function of number of clusters.
        n_init for kmeans is 200

    vects (array-like): Takes a set of feature vectors
        (i.e. 4-channels [chA, chB,...])
    max_clust (int): Maximum number of clusters to grab DBI from

    return:
    n_clust (array-like): list of
    '''
    dbscores = []
    n_clust = np.arange(max_clust+1)
    n_clust = n_clust[2::]

    for cluster in n_clust:
        kmeans = KMeans(n_clusters=cluster, n_init=200).fit(feat_vect)
        labels = kmeans.labels_
        DBI = davies_bouldin_score(feat_vect, labels)
        dbscores.append(DBI)
    dbscores = np.array(dbscores)

    return n_clust, dbscores

def get_cmeans_DBI(feat_vect, max_clust=int, verbose=True):
        '''
        Grabs DBI as clustered via kmeans as a function of number of clusters.
            n_init for kmeans is 200

        vects (array-like): Takes a set of feature vectors
            (i.e. 4-channels [chA, chB,...])
        max_clust (int): Maximum number of clusters to grab DBI from
        verbose (bool): If true, print percentage done (under construction)

        return:
        n_clust (array-like): list of
        '''
        dbscores = []
        n_clust = np.arange(max_clust+1)
        n_clust = n_clust[2::]

        for cluster in n_clust:
            u, minjm = my_cmeans(np.array(feat_vect).T, c=cluster, verbose=verbose)
            labels = np.argmax(u, axis=0) # NOTE: hardens soft labels
            DBI = davies_bouldin_score(feat_vect, labels)
            dbscores.append(DBI)
        dbscores = np.array(dbscores)

        return n_clust, dbscores
