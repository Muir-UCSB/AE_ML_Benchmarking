import numpy as np
import matplotlib as plt
import pandas
from ae_measure2 import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
import pywt
import seaborn as sns

##Note: This file is not yet complete and was being used to figure out wavelet decomposition

def find_feature_vectors(ae_file, filter_csv, dt, channel_number, sig_length = 1024):
    '''This will find the feature vectors from channel, specify the channel number starting with index 0
    Inputs: ae_file(.txt): Voltage-data series
            filter_csv(.csv): filter_csv of important events
            dt(float) = sampling rate
            channel_number(int) = channel number wanting to be analyzed, starting with index 0
            sig_length(int) = signal length (default = 1024)
    Outputs: list_of_feature_vectors(array-like): a list of lists of feature vector values
    '''

    total_signal_length = sig_length/10 #note: converts signal length to microseconds
    start_string = 'Start_channel' +str(channel_number+1) #note: the .csv file starts with channel 1, not 0. Must add 1
    start_energy = 'Energy_ch'+ str(channel_number+1)
    list_of_feature_vectors = []

    waveform_channel = channel_number

    v1, ev = filter_ae(ae_file, filter_csv, waveform_channel, sig_length)
    df = pandas.read_csv(filter_csv)

    for i in range(len(ev)):

        #start_time = df[start_string][i] this is going to be changed

        energy = df[start_energy][i]
        ln_energy = np.log(energy)

        waveform_channel_array = np.array(v1[i])
        max_amplitude = np.max(waveform_channel_array)
        index_location_max = np.argmax(waveform_channel_array)
        max_location_time = index_location_max / 10

        threshold = .1 * max_amplitude  #Note: the threshold is taken as 10 percent of the maximum amplitude
        imin, imax = np.nonzero(v1[i] > threshold)[0][[0, -1]]
        start_time = imin / 10 #Note: converts index location to a start time (microseconds)
        end_time = imax / 10 #Note: converts index location to an end time (microseconds)

        rise_time = max_location_time - start_time
        duration = end_time - start_time #point of question: would the duration now be start to end time
        decay_time = duration - rise_time

        risingpart = v1[i][imin:index_location_max] #Note: this is the rising portion of the waveform
        wr, zr = fft(dt, risingpart, low_pass=None, high_pass=None)
        risingfrequency = get_freq_centroid(wr, zr, low_pass=None, high_pass=None)

        decaypart = v1[i][index_location_max:imax] #Note: this is the falling portion of the waveform
        wd, zd = fft(dt, decaypart, low_pass=None, high_pass=None)
        decayfrequency = get_freq_centroid(wd, zd, low_pass=None, high_pass=None)

        wa, za = fft(dt, v1[i], low_pass=None, high_pass=None)
        avgfrequency = get_freq_centroid(wa, za, low_pass=None, high_pass=None)

        Rln = np.log(rise_time)
        RDln = np.log(rise_time / duration)
        ARln = np.log(max_amplitude / rise_time)
        ADTln = np.log(max_amplitude / decay_time)
        AFln = np.log(max_amplitude / avgfrequency)

        fvector = [Rln, avgfrequency, risingfrequency, ln_energy, RDln, ARln, ADTln, AFln]

        list_of_feature_vectors.append(fvector)

    return list_of_feature_vectors

def PCA_analysis(list_of_feature_vectors):
    '''Conducts PCA on a list of feature vectors
    Inputs: list_of_feature_vectors (array-like): a list of lists of feature vector values
    Outputs: principalComponents (array-like): coordinates for a list of principal components
             eigvalues (array-like): array of eigen values that also corresponds to the variance accounted for by each principal component
             y (float): variance explained in total by the principal components
    '''
    pca = PCA(.95) #Note: determines the number of principal components to explain no less than 0.95 variance
    principalComponents = pca.fit_transform(list_of_feature_vectors)

    eigvalues = pca.explained_variance_ratio_ #variance accounted by each principal component
    y = pca.explained_variance_ratio_.sum()

    return principalComponents, eigvalues, y

def DBIandsilhoutteplot(pclist,experiment,channel):
    '''Takes a principal component list and generates a Davies-Bouldin and Silhouette Plot
    Inputs: pclist (array-like): Takes a list of principal components coordinates and generates Davies-Bouldin and Silhouette Plots
    Outputs: A plot is generated that is the Davies-Bouldin index (and Silhouette) vs. the number of clusters
    '''

    clustervector = list(range(2,11)) # list of number of clusters [2 3 4...]
    dbscores = []
    silhouettescores = []
    for i in clustervector:
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=300).fit(pclist)
        labels = kmeans.labels_

        davies_bouldin_score1 = davies_bouldin_score(pclist, labels)
        dbscores.append(davies_bouldin_score1)

        sil_score1 = silhouette_score(pclist, labels)
        silhouettescores.append(sil_score1)

    plt.plot(clustervector, dbscores, label = 'DBI')
    plt.ylabel('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters')

    ax2 = plt.twinx()
    ax2.plot(clustervector,silhouettescores, label = 'Silhouette', c = 'r')
    ax2.set_ylabel('Silhouette Index')
    plt.title(('DBI and Silhouette Plot ' + 'Experiment ' + str(experiment) + ' Channel ' + str(channel)))
    fig = plt.gcf()
    fig.legend(loc = 'lower left')
    plt.show()

##Figure out wavelet decomp
v1, ev = filter_ae('210308-1_waveforms.txt', '210308-1_filter.csv', 0, 1024)
try1 = v1[0]

wp = pywt.WaveletPacket(data=try1, wavelet = 'db2', mode = 'symmetric', maxlevel=3)
features = np.array([wp['aaa'].data ,wp['aad'].data,wp['ada'].data,wp['add'].data, wp['daa'].data,wp['dad'].data,wp['dda'].data,wp['ddd'].data])
print(features)

#prints a correlation matrix heatmap, used for the identification of lowest correlated nodes
corr = np.corrcoef(features)
print(corr)
sns.heatmap(corr,annot=True)
plt.show()
#print(np.linalg.norm(features))

## This section will acquire the energy percentage contained in each node... will be used for feature extraction
n = 3
re = []  #No. n Decomposition factor of all nodes in layer
for i in [node.path for node in wp.get_level(n, 'freq')]:
    re.append(wp[i].data)
#No. n Layer Energy Characteristics
energy = []
for i in re:
    energy.append(pow(np.linalg.norm(i,ord=None),2))
sumenergy = np.sum(energy)
print(sumenergy)
#for i in energy:
    #print(i)

normalizedE = energy/sumenergy*100
print(normalizedE) #Note: percentage of energy captured by each node... next need to figure out how to determine which nodes to take
''' 
currently not a working function 
def waveletdecomp(ae_file,filter_csv,channel_number):
    v1, ev = filter_ae(ae_file, filter_csv, channel_number, 1024)
    totaldecomp = [] #Note: Dummy variable that will contain all the decompositions for each waveform
    for i in range(len(ev)):
        wp = pywt.WaveletPacket(data=v1[i], wavelet = 'db2', mode = 'symmetric', maxlevel=3)
        features = np.array([wp['aaa'].data, wp['add'].data, wp['daa'].data, wp['dad'].data, wp['ddd'].data])
        #Note: The nodes selected may change for our data... find these using a correlation matrix. The
        #ones with least correlation should be selected.
        totaldecomp.append(features)
    nptotaldecomp = np.array(totaldecomp)
    return nptotaldecomp
x = waveletdecomp('210308-1_waveforms.txt','210308-1_filter.csv',0)
#print(x[0])
'''