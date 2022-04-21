#Imports
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
import os
from itertools import cycle, islice
from matplotlib.pyplot import figure



'''
Example file
'''



if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    mypath = 'C:/Research/Framework_Benchmarking/Data/PLB_data.json'
    data = load_PLB(mypath)
    print(data.keys())




    '''
    Plot example wave
    '''
    index = -15
    sig_len = 1024

    waves = data['data']
    targets = data['target']
    angles = data['target_angle']
    time = np.arange(0,sig_len)/10 # NOTE: convert to microseconds


    signal = []
    for angle in np.unique(targets):
        signal.append(waves[np.where(targets==angle)][index])





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

    ax1.set_ylabel('Amplitude (V)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(25))
    #pl.ylim((0,21))
    ax1.set_xlabel('Time ($\mu s$)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)

    offset = 0
    for i, wave in enumerate(signal):

        if i == 0 or i == 3 or i==4:
            ax1.plot(time, wave-offset*1+2, color=colors[i], linewidth=width, label=angles[i][0:2]+' Degrees', linestyle='-')
            offset +=1


    fig.tight_layout() # NOTE: prevents clipping of plot
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box') # NOTE: make plot square
    #pl.legend(fontsize=SMALL_SIZE, loc='upper right', framealpha=.7)




    pl.show()
