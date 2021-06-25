import numpy as np
import matplotlib as plt
import pandas
from ae_measure2 import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def find_feature_vectors(ae_file, filter_csv, dt, channel_number, sig_length = 1024):
    '''This will find the feature vectors from channel, specify the channel number starting with index 1 '''
    total_signal_length = sig_length/10 #microseconds
    start_string = 'Start_channel' +str(channel_number)
    start_energy = 'Energy_Ch'+ str(channel_number)
    list_of_feature_vectors = []

    waveform_channel = channel_number-1

    v1, ev = filter_ae(ae_file, filter_csv, waveform_channel, sig_length)


    df = pandas.read_csv(filter_csv)


    for i in range(len(ev)):

        start_time = df[start_string][i]


        energy = df[start_energy][i]


        ln_energy = np.log(energy)


        waveform_channel_array = np.array(v1[i])


        max_amplitude = np.max(waveform_channel_array)


        index_location_max = np.argmax(waveform_channel_array)


        max_location_time = index_location_max / 10

        # eventually remove 10... make it into words

        rise_time = max_location_time - start_time


        duration = total_signal_length - start_time


        decay_time = duration - rise_time


        risingpart = v1[i][:index_location_max]


        wr, zr = fft(dt, risingpart, low_pass=None, high_pass=None)
        risingfrequency = get_average_freq(wr, zr, low_pass=None, high_pass=None)


        decaypart = v1[i][index_location_max:]


        wd, zd = fft(dt, decaypart, low_pass=None, high_pass=None)
        decayfrequency = get_average_freq(wd, zd, low_pass=None, high_pass=None)


        wa, za = fft(dt, v1[i], low_pass=None, high_pass=None)
        avgfrequency = get_average_freq(wa, za, low_pass=None, high_pass=None)


        Rln = np.log(rise_time)
        RDln = np.log(rise_time / duration)
        ARln = np.log(max_amplitude / rise_time)
        ADTln = np.log(max_amplitude / decay_time)
        AFln = np.log(max_amplitude / avgfrequency)

        fvector = [Rln, avgfrequency, risingfrequency, ln_energy, RDln, ARln, ADTln, AFln]


        list_of_feature_vectors.append(fvector)

    return list_of_feature_vectors

def PCA_analysis(list_of_feature_vectors):
    '''Conducts PCA on a list of feature vectors'''

    pca = PCA(.95)
    principalComponents = pca.fit_transform(list_of_feature_vectors)

    eigvalues = pca.explained_variance_ratio_
    y = pca.explained_variance_ratio_.sum()


    return principalComponents, eigvalues, y

def rescaling_pc(principal_components, eigenvalues):
    rspc = []
    pctranspose = principal_components.T

    for i in range(len(pctranspose)):
        rescaledi = (pctranspose[i]) * np.sqrt(eigenvalues[i])
        rspc.append(rescaledi)

    final = np.array(rspc).T
    return final

def DBIplot(pclist):
    '''Takes a principal component list and generates a Davies-Bouldin Plot'''
    clustervector = list(range(2,11))
    #print(clustervector)
    dbscores = []
    for i in clustervector:
        kmeans = KMeans(n_clusters=i, random_state=0).fit(pclist)
        labels = kmeans.labels_
        #print(labels)

        davies_bouldin_score1 = davies_bouldin_score(pclist, labels)
        #print(davies_bouldin_score)
        dbscores.append(davies_bouldin_score1)
    #print(dbscores)
    plt.plot(clustervector, dbscores)
    plt.show()

