from deOliviera_functions import *

#This is just a small test script I put together quickly
get3ddistancemap('210308-1_waveforms.txt','210308-1_filter.csv',0,6,6)
weight_vects = getweightvectors('210308-1_waveforms.txt','210308-1_filter.csv',0,6,6)

kmeans = KMeans(n_clusters=10, n_init=100,max_iter=300).fit(weight_vects)
labels = kmeans.labels_
#print(labels)

z = assigningclusters('210308-1_waveforms.txt','210308-1_filter.csv',0,6,6,labels)
#print(z)

separatedAEplot('210308-1_filter.csv',z,0)

