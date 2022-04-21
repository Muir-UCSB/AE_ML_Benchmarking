#Imports
import glob
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
import pandas as pd
import os
from scipy.integrate import simps


if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    os.chdir('C:/Research/Framework_Benchmarking')
    sig_len = 1024

    fname_raw = '220331_20deg_4mm'

    raw = glob.glob("./Data/220331_Muir/"+fname_raw+".txt")[0]



    v0, ev = read_ae_file2(raw, channel_num=0, sig_length=sig_len) # B1025

    #os.chdir("./test_13_FFT_bank/")
    dt = 10**-7 #s
    Low = 200*10**3 #Hz
    High = 1000*10**3 #Hz



    y=v0[67]
    pl.plot(y)
    pl.show()
    w,z = fft(dt, y, low_pass=Low, high_pass=High)
    w=w/1000



    SMALL_SIZE = 10
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('FFT Power (arb. units)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.set_xlim(Low/1000, High/1000)
    ax1.set_xlabel('Frequency (kHz)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)





    plot1 = ax1.plot(w, z/np.max(z), color=color1, linewidth=width)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box') # NOTE: makes plot square
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(.25))
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))
    pl.show()
