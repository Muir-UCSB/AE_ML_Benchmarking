import numpy as np
import matplotlib as plt
import pandas
from ae_measure2 import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
import pywt
import os
import glob
import pylab as pl

os.chdir('C:/Research/Framework_Comparison')

sig_len = 1024
explained_var = 0.95
k = 5 # NOTE: number of clusters
kmeans = KMeans(n_clusters=k, n_init=200)

fname_raw = '210308-1_waveforms'
fname_filter = '210308-1_filter'


'''
Read-in and Setup
'''
raw = glob.glob("./Raw_Data/210308-1/"+fname_raw+".txt")[0]
filter = glob.glob("./Filtered_Data/210308-1/"+fname_filter+".csv")[0]





##Figure out wavelet decomp
v1, ev = filter_ae(raw, filter, channel_num=0)
try1 = v1[0]

wp = pywt.WaveletPacket(data=try1, wavelet = 'db2', mode = 'zero', maxlevel=3)
print([n.path for n in wp.get_leaf_nodes(True)])
pl.plot(wp['aaa'].data)
pl.show()
