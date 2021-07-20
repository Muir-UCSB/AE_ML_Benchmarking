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

def extract_Moevus_vect(waveform=[], dt=10**-7, energy=None, low=None, high=None):
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

    threshold = .1*max_amp  #Note: floating threshold is 10% of the maximum amplitude
    imin, imax = np.nonzero(waveform > threshold)[0][[0, -1]]
    start_time = imin/10 #Note: converts index location to a start time (microseconds)
    end_time = imax/10 #Note: converts index location to an end time (microseconds)

    rise_time = peak_time - start_time
    duration = end_time-start_time
    decay_time = end_time-peak_time

    risingpart = waveform[imin:np.argmax(waveform)] #Note: grabs the rising portion of the waveform
    decaypart = waveform[np.argmax(waveform):imax] #Note: grabs the falling portion of the waveform

    w_rise, z_rise = fft(dt, risingpart, low_pass=low, high_pass=high)
    rise_freq = get_freq_centroid(w_rise, z_rise)

    w, z = fft(dt, waveform, low_pass=low, high_pass=high)
    freq_centroid = get_freq_centroid(w, z)

    log_risetime = np.log(rise_time)
    log_rd = np.log(rise_time/duration)
    log_ar = np.log(max_amp/rise_time)
    log_ad = np.log(max_amp/decay_time)
    log_af = np.log(max_amp/freq_centroid)

    feature_vector = [log_risetime, freq_centroid, rise_freq, ln_energy, log_rd, log_ar, log_ad, log_af]
    return feature_vector








def rescale(feat_vect, eigenvalues):
    '''
    rescale rescales a set of feature vectors under PCA mapping so distnaces conform to Moevus2008

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
