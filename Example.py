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
    mypath = 'C:/Research/Framework_Benchmarking/Data/PLB_data.json'
    data = load_PLB(mypath)
    print(data.keys())




    '''
    Plot example wave
    '''
    index = 145


    waves = data['data']
    targets = data['target']
    angles = data['target_angle']
    energy = data['energy']
    print(angles)

    example_wave = waves[index]
    example_target = targets[index]
    example_energy=energy[index]
    example_angle = angles[example_target]

    print(example_target, example_angle, example_energy)






    '''
    Extract signals from only 1 angle, say 20 degrees
    '''
    reference = waves[np.where(targets==0)]
    test = waves[np.where(targets==2)]

    set = np.vstack((reference, test))
    print(np.shape(set))
