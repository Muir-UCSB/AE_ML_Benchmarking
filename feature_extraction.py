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
from sklearn.metrics import silhouette_score
import pylab as pl
import matplotlib.ticker as ticker
from librosa import zero_crossings as zc
import skfuzzy as fuzz
import pywt
from minisom import MiniSom
from itertools import cycle, islice

def optimize_SOM_hyperparameters(data, sig_range, alpha_range, resolution, neurons,
    epochs=1000, verbose=True):
    '''
    Estimates optimum SOM hyperparameters for a data set, initializes with PCA

    :param data: (array-like) Feature vectors [n x dims]
    :param sig_range: (array-like) Range of sigma to explore [lower, upper]
    :param alpha_range: (array-like) Range of alpha to explore [lower, upper]
    :param resolution: (int) number of points on single axis to explore
    :param neurons: (array-like) [n_neurons, m_neurons]
    :param epochs: (int) Number of epochs to train SOM over

    :return sigma_opt: (float) Optimal neighborhood spread
    :return alpha_opt: (float) Optimal learning rate
    :return min_quant_error: (float) quantization error corresponding to minimum
    '''
    sig_search = np.linspace(sig_range[0], sig_range[1], resolution)
    alpha_search = np.linspace(alpha_range[0], alpha_range[1], resolution)
    n_neurons = neurons[0]
    m_neurons = neurons[1]
    dims = np.array(data).shape[1]

    error_surface = []

    for i, sig in enumerate(sig_search):
        for rate in alpha_search:
            som = MiniSom(n_neurons, m_neurons, dims, sigma=sig, learning_rate=rate,
                      neighborhood_function='gaussian')
            som.pca_weights_init(data)
            som.train(data, epochs, verbose=False)
            weights = som.get_weights()
            error_surface.append([sig, rate, som.quantization_error(data)])
        if verbose is True:
            percent = np.round((i+1)/len(sig_search)*100, decimals=1)
            print(percent,'% complete')

    error_surface = np.array(error_surface).T
    minimum = error_surface.T[np.argmin(error_surface[2])]

    sigma_opt = minimum[0]
    alpha_opt = minimum[1]
    min_quant_error = minimum[2]
    return sigma_opt, alpha_opt, min_quant_error

def extract_agglomerative_vect(waveform=[], dt=10**-7, threshold=.1):
    '''
    waveform (array-like): Voltage time series describing the waveform
    dt(float): Spacing between time samples (s)

    return:
    vect (array-like): feature vector extracted from a waveform according to Moevus2008
    '''
    if waveform == []:
        raise ValueError('An input is missing')

    imin, imax = get_signal_start_end(waveform)
    peak_freq = get_peak_freq(waveform[imin:imax])
    max_amp = np.max(waveform)

    feature_vector = [peak_freq, max_amp]
    return feature_vector


def get_partial_pow(waveform=[], lower_bound=None, upper_bound=None, dt = 10**-7):
    '''
    Gets partial power of signal from waveform from f_0 to f_1

    :param waveform: (array-like) Voltage time series of the waveform
    :param lower_bound: (float) Lower bound of partial power in Hz
    :param upper_bound: (float) Upper bound of partial power in Hz
    :param dt: (float) Time between samples (s) (also inverse of sampling rate)

    :return pow: (float) Partial power
    '''
    if lower_bound is None or upper_bound is None:
        raise ValueError('Partial power bounds not defined')

    w, z = fft(dt, waveform)
    total_pow = np.sum(z)

    pow = np.sum(z[np.where((w>=lower_bound) & (w<upper_bound))])
    partial_pow = pow/total_pow

    return partial_pow



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
            data, c=c, m=m, error=error, maxiter=max_iter) # NOTE: jm is objective function history
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
    dt (float): Time between samples (s) (also inverse of sampling rate)

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
    dt (float): spacing between time samples (s)

    threshold (float): Floating threshold that defines the start and end of signal

    return
    average_frequency (float): Average frequency of signal in Hz
    '''
    imin, imax = get_signal_start_end(waveform, threshold=threshold)
    cut_signal = waveform[imin:imax]
    num_zero_crossings = len(np.nonzero(zc(cut_signal)))

    return num_zero_crossings/(len(cut_signal)*dt)


def extract_Sause_vect(waveform=[], dt=10**-7, threshold=.1):
    '''
    waveform (array-like): Voltage time series describing the waveform
    dt(float): Spacing between time samples (s)
    energy (float): energy of the waveform as calculated by the AE software
    low_pass (float): lower bound on bandpass (Hz)
    high_pass (float): upper bound on bandpass (Hz)

    return:
    vect (array-like): feature vector extracted from a waveform according to Moevus2008
    '''
    if waveform == []:
        raise ValueError('An input is missing')

    imin, imax = get_signal_start_end(waveform)

    risingpart = waveform[imin:np.argmax(waveform)] #Note: grabs the rising portion of the waveform
    fallingpart = waveform[np.argmax(waveform):imax] #Note: grabs the falling portion of the waveform

    average_freq= get_average_freq(waveform, dt=dt, threshold=threshold)
    rise_freq = get_average_freq(risingpart, dt=dt, threshold=threshold)
    reverb_freq = get_average_freq(fallingpart, dt=dt, threshold=threshold)

    w, z = fft(dt, waveform[imin:imax])
    freq_centroid = get_freq_centroid(w,z)

    peak_freq = get_peak_freq(waveform[imin:imax])
    wpf = np.sqrt(freq_centroid*peak_freq)

    pp1 = get_partial_pow(waveform[imin:imax], lower_bound=0, upper_bound=150*10**3)
    pp2 = get_partial_pow(waveform[imin:imax], lower_bound=150*10**3, upper_bound=300*10**3)
    pp3 = get_partial_pow(waveform[imin:imax], lower_bound=300*10**3, upper_bound=450*10**3)
    pp4 = get_partial_pow(waveform[imin:imax], lower_bound=450*10**3, upper_bound=600*10**3)
    pp5 = get_partial_pow(waveform[imin:imax], lower_bound=600*10**3, upper_bound=900*10**3)
    pp6 = get_partial_pow(waveform[imin:imax], lower_bound=900*10**3, upper_bound=1200*10**3)

    #feature_vector = [pp1, pp2]
    #feature_vector = [average_freq, reverb_freq, freq_centroid, rise_freq, peak_freq, wpf, pp1, pp2, pp3]

    feature_vector = [average_freq, reverb_freq, rise_freq, peak_freq, freq_centroid, wpf, pp1, pp2, pp3, pp4, pp5, pp6]

    return feature_vector

def extract_SOM_vect(waveform=[], dt=10**-7, energy=None, threshold=.1):
    '''
    Extracts features from waveforms according to Gutkin at al. 2011

    waveform (array-like): Voltage time series describing the waveform
    dt(float): Spacing between time samples (s)
    energy (float): energy of the waveform as calculated by the AE software
    low_pass (float): lower bound on bandpass (Hz)
    high_pass (float): upper bound on bandpass (Hz)

    return:
    vect (array-like): feature vector extracted from a waveform according to Moevus2008
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

    peak_freq = get_peak_freq(waveform, dt=dt, low=low, high=high)

    feature_vector = [max_amp, peak_freq, energy, rise_time, duration]
    return feature_vector

def extract_Moevus_vect(waveform=[], dt=10**-7, energy=None, threshold=.1):
    '''
    waveform (array-like): Voltage time series describing the waveform
    dt(float): Spacing between time samples (s)
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
    #feature_vector = [ln_energy]

    return feature_vector

def extract_FCM_vect(waveform=[], dt=10**-7, energy=None, threshold=.1):
    '''
    Extracts features from a waveform according to Shateri2017. Note for the peak frequency
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

    feature_vector = [max_amp, counts, energy, duration]
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

def plot_cumulative_AE_labeled(label_set, stress, show=True, save_as=None):
    '''
    Generates an AE vs. Stress plots and separates the events by cluster.
    # TODO: add functionality for custom colormap input

    labels (array-like): List of labels assigned to each AE event
    stress (array-like): List of stresses that each AE event occurs at
    show (bool): If true then plot will show
    save_as (str): if path is entered as file_name.png then the figure will save

    return:    None
    '''

    colors = np.array(list(islice(cycle(['tab:blue','tab:orange', 'tab:gray','tab:brown',
                                         '#f781bf', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(label_set) + 1))))
    tick_marks = np.array(list(islice(cycle(['o','>', 's', 'd']),
                                  int(max(label_set) + 1))))
    lines = np.array(list(islice(cycle(['-','--', '-.', ':']),
                                  int(max(label_set) + 1))))

    labels = np.unique(label_set)
    num_events = len(label_set)
    ymax=0

    '''
    Plotting parameters
    '''
    MEDIUM_SIZE = 14
    LARGE_SIZE = 18
    #cmap = plt.cm.get_cmap(colormap, len(labels)) # NOTE: make colormap with number of elements equal to number of clusters

    fig, ax1 = pl.subplots()

    ax1.set_ylabel('Normalized Cumulative Number AE', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(.1))


    ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    for i, label in enumerate(labels):
        labeled_stress = stress[np.where(label_set==label)]
        cumulative_num = (np.arange(len(labeled_stress))+1)/num_events # NOTE: +1 indexes from 1
        ax1.plot(labeled_stress, cumulative_num, color=colors[i],
            label='Cluster '+str(i+1), marker = tick_marks[i], linestyle=lines[i], linewidth=2)
        if max(cumulative_num) > ymax:
            ymax = max(cumulative_num)

    pl.ylim([-0.02, ymax+ymax*.05])
    pl.legend(fontsize=MEDIUM_SIZE)
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box') # NOTE: makes plot square

    if show == True:
        pl.show()

    if type(save_as) is str:
        pl.savefig(save_as)

def get_DBI(feat_vect, max_clust=int, init=int, verbose=False):
    '''
    Grabs DBI as clustered via kmeans as a function of number of clusters.
        n_init for kmeans is 200

    vects (array-like): Takes a set of feature vectors
        (i.e. 4-channels [chA, chB,...])
    max_clust (int): Maximum number of clusters to grab DBI from

    return:
    n_clust (array-like): list of of number of clusters
    '''
    dbscores = []
    n_clust = np.arange(max_clust+1)
    n_clust = n_clust[2::]

    for cluster in n_clust:
        kmeans = KMeans(n_clusters=cluster, n_init=init).fit(feat_vect)
        if verbose is True:
            print('Cluster: ', cluster)
        labels = kmeans.labels_
        DBI = davies_bouldin_score(feat_vect, labels)
        dbscores.append(DBI)
    if verbose is True:
        print('Complete')
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

def get_SV(feat_vect, max_clust=int, verbose=False):
    '''
    Grabs silhouette_score as clustered via kmeans as a function of number of clusters.
        n_init for kmeans is 2000

    vects (array-like): Takes a set of feature vectors
        (i.e. 4-channels [chA, chB,...])
    max_clust (int): Maximum number of clusters to grab DBI from

    return:
    n_clust (array-like): list of
    '''
    svscores = []
    n_clust = np.arange(max_clust+1)
    n_clust = n_clust[2::]

    for cluster in n_clust:
        kmeans = KMeans(n_clusters=cluster, n_init=2000).fit(feat_vect)
        if verbose is True:
            print('Cluster: ', cluster)
        labels = kmeans.labels_
        SV = silhouette_score(feat_vect, labels)
        svscores.append(SV)
    if verbose is True:
        print('Complete')
    svscores = np.array(svscores)

    return n_clust, svscores
