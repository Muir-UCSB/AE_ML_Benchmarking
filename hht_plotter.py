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
from scipy.cluster.vq import whiten
import pandas as pd
import os
from scipy.cluster.vq import whiten
from scipy.integrate import simps
from Muir_hht import *

if __name__ == "__main__":
    '''
    Read-in and Setup
    '''


    duration = 102.4 # microseconds
    fs = 10 # sampling rate, 10 MHz
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    sig_len = 1024

    fname_raw = 'test_13_waveforms'
    fname_filter = '201015_test_13_filter'
    raw = glob.glob("./Raw_Data/2_sensor/201015/"+fname_raw+".txt")[0]
    filter = glob.glob("./Filtered_Data/2_sensor/201015/"+fname_filter+".csv")[0]

    csv = pd.read_csv(filter)

    time = np.array(csv.Time_aligned)
    stress = np.array(csv.Stress_MPa)


    v0, ev = filter_ae(raw, filter, channel_num=0, sig_length=sig_len) # S9225
    v1, ev = filter_ae(raw, filter, channel_num=1, sig_length=sig_len) # S9225
    v2, ev = filter_ae(raw, filter, channel_num=2, sig_length=sig_len) # B1025
    v3, ev = filter_ae(raw, filter, channel_num=3, sig_length=sig_len) # B1025


    os.chdir("./test_13_HHT_bank_emf_smooth_01/")

    print(os.getcwd())


    for i, signal in enumerate(v2):
        imfs = get_Imfs(signal)
        spectrogram = get_spectrogram(imfs, duration, fs, t)
        freq, power = get_marginal_spectrum(spectrogram, smooth = True, dw = .01)
        freq, power = band_pass(freq, power, low_pass=.1, high_pass = 1.2)



        SMALL_SIZE = 10
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 18
        width = 2.0

        fig, ax1 = pl.subplots()
        color1 = 'black'
        color2 = 'blue'
        color3 = 'red'

        ax1.set_ylabel('HHT Power (arb. units)', fontsize=MEDIUM_SIZE)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
        #ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.set_xlim(.2, 1.2)
        ax1.set_xlabel('Frequency (kHz)', fontsize=MEDIUM_SIZE)
        ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)
        ax1.grid()

        plot1 = ax1.plot(freq, power, color=color1, linewidth=width)
        pl.title('Event '+str(ev[i]))
        #ax1.legend(loc='best', fontsize=MEDIUM_SIZE, framealpha=1, shadow=True)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        pl.savefig('ev_'+str(ev[i])+'.png')
        pl.close()
