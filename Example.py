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
    mypath = 'E:/Research/Framework_Benchmarking/Data/PLB_data.json'
    data = load_PLB(mypath)
    print(data.keys())




    '''
    Plot example wave
    '''
    index = 0


    waves = data['data']
    targets = data['target']
    angles = data['target_angle']

    example_wave = waves[index]
    example_target = targets[index]
    example_angle = angles[example_target]

    print(example_target, example_angle)



    SMALL_SIZE = 10
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 18
    width = 2.0

    fig, ax1 = pl.subplots()
    color1 = 'black'
    color2 = 'blue'
    color3 = 'red'

    ax1.set_ylabel('Voltage (arb. units)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize = MEDIUM_SIZE)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(.25))
    ax1.set_xlabel('Time ($\mu$s)', fontsize=MEDIUM_SIZE)
    ax1.tick_params(axis='x', labelsize=MEDIUM_SIZE)

    plot1 = ax1.plot(example_wave, color=color1, linewidth=width)

    fig.tight_layout() # NOTE: prevents clipping of plot
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box') # NOTE: make plot square
    pl.show()



    '''
    Extract signals from only 1 angle, say 20 degrees
    '''
    reference = waves[np.where(targets==2)]
    print(len(reference))
