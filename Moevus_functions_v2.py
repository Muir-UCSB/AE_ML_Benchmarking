import numpy as np
import matplotlib as plt
import pandas
from ae_measure2 import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

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

def rescaling_pc(principal_components, eigenvalues):
    ''' rescaling_pc rescales a list of principal components by respective eigenvalues
    Inputs: principal_components (array-like): An array of principal components determined by the PCA_analysis function
            eigenvalues (array-like): An array of eigenvalues that corresponds to the principal components
    Outputs: final (array-like): coordinates of rescaled principal components
    '''
    rspc = [] #holder variable that stands for rescaled principal components
    pctranspose = principal_components.T

    for i in range(len(pctranspose)):
        rescaledi = (pctranspose[i]) * np.sqrt(eigenvalues[i])
        rspc.append(rescaledi)

    final = np.array(rspc).T
    return final

def DBIplot(pclist):
    '''Takes a principal component list and generates a Davies-Bouldin Plot
    Inputs: pclist (array-like): Takes a list of principal components coordinates and generates a Davies-Bouldin Plot
    Outputs: A plot is generated that is the Davies-Bouldin index vs. the number of clusters
    '''

    clustervector = list(range(2,11))
    dbscores = []
    for i in clustervector:
        kmeans = KMeans(n_clusters=i, n_init=100).fit(pclist)
        labels = kmeans.labels_

        davies_bouldin_score1 = davies_bouldin_score(pclist, labels)
        dbscores.append(davies_bouldin_score1)
    plt.plot(clustervector, dbscores)
    plt.show()

def convergenceplot(pclist):
    '''Generates a convergence plot based on the number of initializations
    Inputs: pclist (array-like): Takes a list of principal components coordinates
    Outputs: Convergence plot based on the number of initializations
    '''
    inertia = []
    for i in range(1,50):
        kmeans = KMeans(n_clusters=4, n_init=i).fit(pclist)
        inertia.append(kmeans.inertia_)

    plt.plot(np.log(inertia))
    plt.show()

def iterationplot(pclist):
    '''Generates a convergence plot based on the number of initializations
    Inputs: pclist (array-like): Takes a list of principal components coordinates
    Outputs: Convergence plot based on the number of initializations
    '''
    inertia = []
    for i in range(1, 300):
        kmeans = KMeans(n_clusters=4, n_init=100, max_iter=i).fit(pclist)
        inertia.append(kmeans.inertia_)

    plt.plot(np.log(inertia))
    plt.show()

def cumulativeAEplot(filter_csv):
    ''' Uses a filter csv file and generates a cumulative AE plot not separated by cluster
    Inputs: filter_csv (.csv file): A filter csv file of important events
    Outputs: Cumulative AE vs. Stress plot
    '''
    stresses = []
    aecumulative = []
    df = pandas.read_csv(filter_csv)
    for i in range(len(df['Adjusted_Stress_MPa'])):
        stress = df['Adjusted_Stress_MPa'][i]
        stresses.append(stress)
        aevalue = i/len(df['Adjusted_Stress_MPa'])
        aecumulative.append(aevalue)
    print(stresses)
    print(aecumulative)
    plt.scatter(stresses,aecumulative)
    plt.show()

def separatedAEplot(filter_csv, labels,channel_number):
    ''' Generates an AE vs. Stress plots and separates the events by cluster
    Inputs: filter_csv (.csv file): A filter csv file of important events
            labels (array-like): cluster labels from the kmeans funtion
            channel_number (int): channel number starting with index of 0
    Outputs: An AE vs Stress plot where the clusters are separated by color
    Note: Depending on the labels you put, this function will only work for 5 clusters. More can be added by adding another
          stressesX variable, an elif statement in the for loop, another aeX block, and axX.scatter statement.
    '''
    stresses0 = []
    stresses1 = []
    stresses2 = []
    stresses3 = []
    stresses4 = []
    df = pandas.read_csv(filter_csv)
    for i in range(len(df['Adjusted_Stress_MPa'])):
        stress = df['Adjusted_Stress_MPa'][i]
        if labels[i] == 0:
            stresses0.append(stress)

        elif labels[i] == 1:
            stresses1.append(stress)

        elif labels[i] == 2:
            stresses2.append(stress)
        elif labels[i] == 3:
            stresses3.append(stress)
        elif labels[i] == 4:
            stresses4.append(stress)

    ae0 = range(len(stresses0))
    ae0 = np.array(ae0)
    aecumulative0 = ae0/len(df['Adjusted_Stress_MPa'])

    ae1 = range(len(stresses1))
    ae1 = np.array(ae1)
    aecumulative1 = ae1/len(df['Adjusted_Stress_MPa'])

    ae2 = range(len(stresses2))
    ae2 = np.array(ae2)
    aecumulative2 = ae2/len(df['Adjusted_Stress_MPa'])

    ae3 = range(len(stresses3))
    ae3 = np.array(ae3)
    aecumulative3 = ae3 / len(df['Adjusted_Stress_MPa'])

    ae4 = range(len(stresses4))
    ae4 = np.array(ae4)
    aecumulative4 = ae4 / len(df['Adjusted_Stress_MPa'])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(stresses0,aecumulative0, c='k')
    ax1.scatter(stresses1,aecumulative1, c ='b' )
    ax1.scatter(stresses2,aecumulative2, c = 'r')
    ax1.scatter(stresses3, aecumulative3, c = 'g')
    ax1.scatter(stresses4, aecumulative4, c='c')
    plt.xlabel('Stress (MPa)')
    plt.ylabel('Normalized AE Cumulative Score')
    plt.title([str(filter_csv),'Channel'+str(channel_number)])
    plt.show()





