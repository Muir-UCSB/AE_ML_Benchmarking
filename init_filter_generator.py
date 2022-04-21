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

    fname_raw = '220404_40deg_4mm'

    raw = glob.glob("./Data/220404_Furst/"+fname_raw+".txt")[0]



    v0, ev = read_ae_file2(raw, channel_num=0, sig_length=sig_len) # B1025


    #os.chdir("./test_13_FFT_bank/")
    dt = 10**-7 #s
    Low = 200*10**3 #Hz
    High = 1000*10**3 #Hz


    for i in range(len(v0)):

        y=v0[i]
        pl.title('ev'+str(i+1))
        pl.plot(y)
        pl.show()
