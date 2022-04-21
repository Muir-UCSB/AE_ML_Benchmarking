#Imports
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
import os
from itertools import cycle, islice

'''
Example file
'''



if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    mypath = 'E:/Research/Framework_Benchmarking/Data/PLB_data.json'
    data = load_PLB(mypath)
    print(data.keys())




    '''
    Plot example wave
    '''
    index = -15

    waves = data['data']
    targets = data['target']
    angles = data['target_angle']


    FFT = []
    for angle in np.unique(targets):
        FFT.append(waves[np.where(targets==angle)][index])




    dt = 10**-7 #s
    fs = 1/dt
    Low = 200*10**3 #Hz
    High = 800*10**3 #Hz
    offset = 0.0

    for i, signal in enumerate(FFT):
        FFT[i] = butter_bandpass_filter(signal, Low, High, fs)
        w, FFT[i] = fft(dt, FFT[i], low_pass=Low, high_pass=High)
        FFT[i] = FFT[i]/max(FFT[i])



    w=w/1000




    '''
    Make plots
    '''
    colors = np.array(list(islice(cycle(['tab:blue','tab:orange', 'tab:gray','tab:brown',
                                         'black', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(targets) + 1))))


    SMALL_SIZE = 15
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 18
    width = 2.5

    fig, ax1 = pl.subplots()

    ax1.set_ylabel('FFT Power (arb. units)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))
    pl.ylim((0,21))
    ax1.set_xlabel('Frequency (kHz)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)


    for i, spectrum in enumerate(FFT):
        if i == 0 or i == 3 or i==4:
            ax1.plot(w, spectrum , color=colors[i], linewidth=width, label=angles[i][0:2]+' Degrees', linestyle='-')



    fig.tight_layout() # NOTE: prevents clipping of plot
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box') # NOTE: make plot square
    pl.legend(fontsize=SMALL_SIZE)
    pl.show()
