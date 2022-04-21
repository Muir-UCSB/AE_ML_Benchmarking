#Imports
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker
from ae_measure2 import *
import os

'''
Example file
'''



if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    mypath = 'C:/Research/Framework_Benchmarking/Data/220407_Tulshibagwale/220407_26deg_4mm_waves.txt'



    v0, ev = read_ae_file2(mypath, 0)
    print(v0)



    for i, wave in enumerate(v0):
        pl.plot(wave)
        pl.title('event '+ str(i+1))
        pl.show()
