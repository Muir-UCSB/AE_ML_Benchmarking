'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200127

This script takes in a file path with one experiment as text file
It clusters with k=3 and prints each fingerprint to a folder for its respective label.
Labels of Fingerprints correspond to Danny Don't-Vito.png Note the
initialization scheme defaults to k-means++. Random state is not called.

Fingerprints saved to folders in your home directory named */cluster_i_prints

You have to manually delete the contents of the cluster_i_prints folder prior
to running, otherwise you will get prints from the last run mixed in.
'''

#Imports
from SAX import *
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import glob
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
import pandas as pd
import os
from scipy.integrate import simps
from Muir_hht import *





'''
Read-in and Setup
'''

if __name__ == "__main__":


    '''
    Read-in and Setup
    '''

    # new test load in

    sig_len = 1024
    fs = 10
    duration  = sig_len/fs

    fname_raw = 'test_10_waveforms'
    fname_filter = '201006_test_10_filter'
    raw = glob.glob("./Raw_Data/2_sensor/201006/"+fname_raw+".txt")[0]
    filter = glob.glob("./Filtered_Data/2_sensor/201006/"+fname_filter+".csv")[0]

    csv = pd.read_csv(filter)
    time = np.array(csv.Time_aligned)
    stress = np.array(csv.Stress_MPa)



    v0, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len) # S9225
    v1, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len) # S9225
    v2, ev = filter_ae(raw, filter, channel_num=2, sig_length=sig_len) # B1025
    v3, ev = filter_ae(raw, filter, channel_num=3, sig_length=sig_len) # B1025





    i=-2
    print(stress[i])

    y0 = v0[i]
    y1 = v1[i]
    y2 = v2[i]
    y3 = v3[i]

    pi = np.pi

    duration = 102.4
    fs = 10
    samples = int(fs*duration)
    t = np.arange(samples) / fs


    '''
    get noise
    '''

    signal = y1

    #eimf1 = get_Imfs(signal)
    eimf2 = get_eImfs(signal, num_trials=100)
    #plot_hilbert(eimf2[0], duration, fs)

    #spect1 = get_spectrogram(eimf1, duration, fs, t, plot=True).T
    spect2 = get_spectrogram(eimf2, duration, fs, t, plot=True)
'''
    freq, power = get_marginal_spectrum(spect2, smooth = True, dw = .01)
    pl.plot(freq, power)
    pl.show()
'''
