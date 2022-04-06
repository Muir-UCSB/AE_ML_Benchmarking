#Imports
import glob
import numpy as np
import pylab as pl
from ae_measure2 import *
import os
from os import listdir, chdir
from os.path import isfile, join
import json

'''
Script has very specific requirements. The data directory must ONLY contain
waveform files and filter files. The waveform files must be named in the format
date_angle_freeLeadLength_wave: XXXXXXX_XXdeg_Xmm_wave.txt. The filter files
must be of the same format XXXXXXX_XXdeg_Xmm_filter.csv. There can be no other
files in this directory.
'''


if __name__ == "__main__":

    '''
    Read-in and Setup
    '''
    sig_len = 1024
    data_directory = 'E:/Research/Framework_Benchmarking/Data/220405_data_files/'
    write_directory = 'E:/Research/Framework_Benchmarking/Data/'
    write_to = 'PLB_data.json'



    os.chdir(data_directory)
    onlyfiles = [f for f in listdir(data_directory) if isfile(join(data_directory, f))] # NOTE: gets file list in directory

    '''
    Setup holders
    '''
    waves = np.array([])
    labels = np.array([])
    target_angle = np.array([])
    target = np.array([])

    angles = [] # NOTE: generate angle list, setup for parsing through file tree
    for i, file in enumerate(onlyfiles):
        angles.append(file[7:12]) # NOTE: gets angle from file name, hard coded
    target_angle = np.unique(angles).tolist()

    for i, angle in enumerate(target_angle):
        newfiles = [f for f in onlyfiles if angle in f] # NOTE: gets list of files with XXangle in fname
        for j, file in enumerate(newfiles):
            if 'wave' in file:
                raw = file
            elif 'filter' in file:
                filter = file

        v0, ev = filter_ae(raw, filter, channel_num=0)
        label = np.ones(len(v0))*i

        target = np.hstack((target,label)) # NOTE: true label of waveform i
        waves = np.hstack((waves, np.ravel(v0))) # NOTE: ravel needed for dimension agreement, reshape later
        # NOTE: CONTINUED: waves are put back into normal form later

    target = target.astype(int).tolist()
    waves = waves.reshape(len(target), sig_len) # NOTE: don't flip order of
    waves = waves.tolist()

    PLB_data = {'data':waves, 'target':target, 'target_angle':target_angle}




    os.chdir(write_directory)
    with open(write_to, "w") as outfile:
        json.dump(PLB_data, outfile)
