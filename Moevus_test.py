import numpy as np
import matplotlib as plt
import pandas
from Moevus_functions import *

list_of_feature_vectors1 = find_feature_vectors('test_13_waveforms.txt', '201015_test_13_filter.csv',10**-7,1, sig_length= 1024)
print(list_of_feature_vectors1)

pcacomponents, eigenvalues, variance_accounted = PCA_analysis(list_of_feature_vectors1)

print(pcacomponents)
print(eigenvalues)
print(variance_accounted)

rescaledpc = rescaling_pc(pcacomponents,eigenvalues)

print(rescaledpc)

DBIplot(rescaledpc)

