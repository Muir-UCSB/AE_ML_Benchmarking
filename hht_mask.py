'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 200127

This script takes in a file path with one experiment as text file
It clusters with k=3 and prints each fingerprint to a folder for its respective label.
Labels of Fingerprints correspond to Danny Don't-Vito.png


'''

#Imports
from SAX import *
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import glob
import sklearn
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
import pandas as pd
import os
from scipy.integrate import simps
from pyhht import *
from scipy.signal import hilbert
from Muir_hht import *




if __name__ == "__main__":

    '''
    Read-in and Setup
    '''

    # new test load in
    '''
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
    '''


    sig_len = 1024
    fs = 10
    duration  = sig_len/fs

    # old test load in
    filter = glob.glob("./Filtered_Data/2_sensor/Test1/s9225_matched_filter_v2.csv")[0]
    raw = glob.glob("./Raw_Data/2_sensor/Test1/HNSB_2-1_S9225.txt")[0]
    v0, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len) # S9225
    v1, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len) # S9225

    filter_b1025 = glob.glob("./Filtered_Data/2_sensor/Test1/b1025_matched_filter_v2.csv")[0]
    raw_b1025 = glob.glob("./Raw_Data/2_sensor/Test1/HNSB2-1_1025.txt")[0]
    v2, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len) # B1025
    v3, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len) # B1025

    csv = pd.read_csv(filter)
    time = np.array(csv.Time)




    channels = [v0, v1, v2, v3]



    '''
    Parameters
    '''
    k = 2
    fs = 10
    duration  = sig_len/fs
    Low = 0 #Hz
    High = 1.0
    num_bins = 20

    fs = 10
    duration  = sig_len/fs
    samples = sig_len  # also int(fs*duration)
    t = np.arange(samples) / fs

    gam = 1



    '''
    Cast experiment as vectors
    '''
    imfset = []
    vect = []
    for i, channel in enumerate(channels):
        channel_vector = []
        imf_holder = []
        for l, waveform in enumerate(channel):
            imfs = get_eImfs(waveform, num_trials=1000)

            imf_holder.append(imfs)

            spec = get_spectrogram(imfs, duration, fs, t)
            freq, power = get_marginal_spectrum(spec, smooth = True, dw=.01)
            freq, power = band_pass(freq, power, low_pass=Low, high_pass=High)

            dw = freq[1]-freq[0]
            spacing = (High-Low)/(num_bins)
            interval_width = int(spacing/dw) # number of indicies between interval ceilings

            area_tot = simps(power)  #since we don't care about the total area under the
                                #curve, we don't need to specify the x axis as long as
                                #it remains the same.
            feature_vector = []
            print(i, l)
            for j in range(num_bins):
                subinterval = power[j*interval_width: (j+1)*interval_width]
                myint = simps(subinterval)/area_tot
                feature_vector.append(myint) # single waveform as a vector
            channel_vector.append(feature_vector) # set of all waveforms from channel as a vector
        vect.append(channel_vector) # set of all waveforms from experiment as vector index: i,j,k -> channel, waveform #, vector entry
        imfset.append(imf_holder)
    print(spacing)
    imfset = np.array(imfset)

    myimf = pd.DataFrame({'ch0_imf': imfset[0],'ch1_imf': imfset[1], 'ch2_imf': imfset[2], 'ch3_imf':imfset[3]})
    myimf.to_csv(r'imf_hht_eemf1000trial_smooth_01.csv')

    # Cluster waveform3
    ch0_X = vect[0]
    ch1_X = vect[1]
    ch2_X = vect[2]
    ch3_X = vect[3]


    '''
    Clustering
    '''

    spectral_A = SpectralClustering(n_clusters=k, n_init=100, gamma = gam).fit(ch0_X)
    A_lads = spectral_A.labels_


    # Cluster waveform
    spectral_B = SpectralClustering(n_clusters=k, n_init=100, gamma = gam,).fit(ch1_X)
    B_lads = spectral_B.labels_

    spectral_C = SpectralClustering(n_clusters=k, n_init=100, gamma = gam).fit(ch2_X)
    C_lads = spectral_C.labels_


    # Cluster waveform
    spectral_D = SpectralClustering(n_clusters=k, n_init=100, gamma = gam,).fit(ch3_X)
    D_lads = spectral_D.labels_





    '''
    Channel A
    '''

    # Plotting routine for Danny_dont_vito
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.set_xlabel('Time (s)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(time, A_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel A', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('Channel_A_mask_hht.png')

    df = pd.DataFrame({'Event': ev,'Cluster': A_lads, 'Time': time})
    df.to_csv(r'Channel_A_mask_hht.csv')

    pl.clf()



    '''
    # Channel B mask
    '''



    # Plotting routine for Danny_dont_vito
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax1.set_xlabel('Time (s)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(time, B_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel B', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('channel_B_mask_hht.png')



    df2 = pd.DataFrame({'Event': ev,'Cluster': B_lads, 'Time': time})
    df2.to_csv(r'Channel_B_mask_hht.csv')

    pl.clf()



    '''
    Channel C
    '''

    # Plotting routine for Danny_dont_vito
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #ax1.set_xlabel('Stress (MPa)', fontsize=MEDIUM_SIZE)
    ax1.set_xlabel('Time (s)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(time, C_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel C', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('Channel_C_mask_hht.png')

    df3 = pd.DataFrame({'Event': ev,'Cluster': C_lads, 'Time': time})
    df3.to_csv(r'Channel_C_mask_hht.csv')

    pl.clf()



    '''
    # Channel D mask
    '''



    # Plotting routine for Danny_dont_vito
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Cluster number', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax1.set_xlabel('Time (s)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
    ax1.grid()

    plot1 = ax1.scatter(time, D_lads+1 , color=color1, linewidth=width) #plot silh
    pl.title('Channel D', fontsize=BIGGER_SIZE)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    pl.savefig('channel_D_mask_hht.png')



    df4 = pd.DataFrame({'Event': ev,'Cluster': D_lads, 'Time': time})
    df4.to_csv(r'Channel_D_mask_hht.csv')

    pl.clf()
