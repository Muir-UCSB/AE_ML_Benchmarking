from ae_measure2 import *
import pandas
from minisom import MiniSom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from sklearn.decomposition import PCA

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

def getdistancemap(ae_file,filter_csv,channel_number,n_neurons,m_neurons):
    '''This will find the 2D distance map using the MiniSom package
    Inputs: ae_file(.txt): Voltage-data series
            filter_csv(.csv): filter_csv of important events
            channel_number (float): channel number ranging from 0 to 3
            n_neurons (float): number of rows in SOM
            m_neurons (float): number of columns in SOM
    Outputs: 2D distance map made using SOM
    '''
    featurevect1 = find_feature_vectors(ae_file, filter_csv, 10 ** -7, channel_number)
    npfeaturevect1 = np.array(featurevect1)

    # Initialization and training
    som = MiniSom(n_neurons, m_neurons, npfeaturevect1.shape[1], sigma=1.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=0)

    som.pca_weights_init(npfeaturevect1)
    som.train(npfeaturevect1, 1000, verbose=True)  # random training

    plt.figure(figsize=(9, 9))

    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    plt.title(['Distance Map',str(filter_csv),'Channel'+str(channel_number)])
    plt.show()

def get3ddistancemap(ae_file,filter_csv,channel_number,n_neurons,m_neurons):
    '''This will find the 3D distance map using the MiniSom package
    Inputs: ae_file(.txt): Voltage-data series
            filter_csv(.csv): filter_csv of important events
            channel_number (float): channel number ranging from 0 to 3
            n_neurons (float): number of rows in SOM
            m_neurons (float): number of columns in SOM
    Outputs: 3D distance map made using SOM
    '''
    featurevect1 = find_feature_vectors(ae_file, filter_csv, 10 ** -7, channel_number)
    npfeaturevect1 = np.array(featurevect1)

    # Initialization and training
    som = MiniSom(n_neurons, m_neurons, npfeaturevect1.shape[1], sigma=1.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=0)

    som.pca_weights_init(npfeaturevect1)
    som.train(npfeaturevect1, 1000, verbose=True)  # random training

    plt.figure(figsize=(9, 9))

    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    plt.title(['Distance Map', str(filter_csv), 'Channel' + str(channel_number)])

    points = [] #dummy variable that will hold XYZ coordinates
    for i in range(n_neurons):
        for j in range(m_neurons):
            points.append([i, j , som.distance_map().T[i][j]])

    nppoints= np.array(points)
    nppointsT = nppoints.T

    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_trisurf(nppointsT[0], nppointsT[1], nppointsT[2], cmap='bone_r',linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(['Distance Map', str(filter_csv), 'Channel' + str(channel_number)])
    plt.show()

def getweightvectors(ae_file,filter_csv,channel_number,n_neurons,m_neurons):
    '''This will find the weight vectors using the MiniSom package
    Inputs: ae_file(.txt): Voltage-data series
            filter_csv(.csv): filter_csv of important events
            channel_number (float): channel number ranging from 0 to 3
            n_neurons (float): number of rows in SOM
            m_neurons (float): number of columns in SOM
    Outputs: reshapez (list of lists): list of weight vectors
             Will also display a 2D SOM
    '''
    featurevect1 = find_feature_vectors(ae_file, filter_csv, 10 ** -7, channel_number)
    npfeaturevect1 = np.array(featurevect1)

    # Initialization and training
    som = MiniSom(n_neurons, m_neurons, npfeaturevect1.shape[1], sigma=1.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=0)

    som.pca_weights_init(npfeaturevect1)
    som.train(npfeaturevect1, 1000, verbose=True)  # random training

    plt.figure(figsize=(9, 9))

    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    z = som.get_weights()
    reshapez = np.reshape(z,(-1,len(z[0][0])))
    plt.colorbar()
    plt.title(['Distance Map', str(filter_csv), 'Channel' + str(channel_number)])
    return reshapez

def DBIplot(pclist):
    '''Takes a principal component list and generates a Davies-Bouldin Plot
    Inputs: pclist (array-like): Takes a list of principal components coordinates and generates a Davies-Bouldin Plot
    Outputs: A plot is generated that is the Davies-Bouldin index vs. the number of clusters
    '''

    clustervector = list(range(2,11))
    dbscores = []
    for i in clustervector:
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=300).fit(pclist)
        labels = kmeans.labels_

        davies_bouldin_score1 = davies_bouldin_score(pclist, labels)
        dbscores.append(davies_bouldin_score1)
    plt.figure()
    plt.plot(clustervector, dbscores)
    plt.show()

def convergenceplot(pclist):
    '''Generates a convergence plot based on the number of initializations
    Inputs: pclist (array-like): Takes a list of principal components coordinates
    Outputs: Convergence plot based on the number of initializations
    '''
    inertia = []
    for i in range(1,100):
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

def labeldistancemap(ae_file,filter_csv,channel_number,n_neurons,m_neurons, labellist):
    '''This will get the labeled 2D distance map by cluster
        Inputs: ae_file(.txt): Voltage-data series
                filter_csv(.csv): filter_csv of important events
                channel_number (float): channel number ranging from 0 to 3
                n_neurons (float): number of rows in SOM
                m_neurons (float): number of columns in SOM
                labellist (array-like): list of labels found using kmeans on weight vectors
        Outputs: 3D distance map made using SOM
    '''
    #reshape label list to fit the original grid
    reshapelabels= np.reshape(labellist,(n_neurons,m_neurons))
    featurevect1 = find_feature_vectors(ae_file, filter_csv, 10 ** -7, channel_number)
    npfeaturevect1 = np.array(featurevect1)

    # Initialization and training
    som = MiniSom(n_neurons, m_neurons, npfeaturevect1.shape[1], sigma=1.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=0)

    som.pca_weights_init(npfeaturevect1)
    som.train(npfeaturevect1, 1000, verbose=True)  # random training

    plt.figure(figsize=(9, 9))

    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    plt.colorbar()
    plt.title(['Distance Map',str(filter_csv),'Channel'+str(channel_number)])

    #Plot the clusters on the map
    currentx = .4
    currenty = .5
    for i in range(len(reshapelabels)):
        for j in range(len(reshapelabels[i])):
            plt.text(currentx,currenty, str(reshapelabels[i][j]), fontsize=14, color = 'red')
            currentx += 1
        currentx = .4
        currenty += 1

    plt.show()

def DBI2plot(pclist,weight_vects,experiment,channel):
    '''Takes a principal component list and generates a Davies-Bouldin Plot for the Moevus and de Oliveira frameworks
    Inputs: pclist (array-like): Takes a list of rescaled principal components coordinates and generates a Davies-Bouldin Plot
            weight_vects (array-like): Takes a list of weight vectors from the getweightvectors functuon and generates DBI Plot
    Outputs: A plot is generated that is the Davies-Bouldin index vs. the number of clusters for both frameworks
    '''

    clustervector = list(range(2,11)) #Note: list of clusters [2 3 4 ... 10]
    dbscores = []
    dbscores2 = []
    for i in clustervector:
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=300).fit(pclist)
        labels = kmeans.labels_
        davies_bouldin_score1 = davies_bouldin_score(pclist, labels)
        dbscores.append(davies_bouldin_score1)

    for i in clustervector:
        kmeans2 = KMeans(n_clusters=i, n_init=100, max_iter=300).fit(weight_vects)
        labels2 = kmeans2.labels_
        davies_bouldin_score2 = davies_bouldin_score(weight_vects, labels2)
        dbscores2.append(davies_bouldin_score2)

    plt.figure()
    plt.ylabel('Davies Bouldin Index')
    plt.xlabel('Number of Clusters')
    plt.title(('DBI Plot '+ 'Experiment ' +str(experiment)+ ' Channel '+ str(channel)))
    plt.plot(clustervector, dbscores, label = 'Moevus')
    plt.plot(clustervector, dbscores2, label = 'deOliveira')
    plt.legend()
    plt.show()

def determineSOMARI(ae_file, filter_csv, channelx, channely, clusters, n_neurons, m_neurons):
    ''' determineSOMARI takes in inputs and returns an Adjusted Rand Index Value between two channels using a self-organizing map
    Inputs: ae_file(.txt): Voltage-data series
            filter_csv(.csv): filter_csv of important events
            channelx (int): first channel wanted to compare (ranging from 0 to 3)
            channely (int): second channel wanted to compare (ranging from 0 to 3)
            clusters (int): number of clusters
            n_neurons (int): number of rows in SOM
            m_neurons (int): number of columns in SOM
    Outputs: ARI value (float): Adjusted Rand Index Value
    '''
    weightvect1 = getweightvectors(ae_file,filter_csv,channelx,n_neurons,m_neurons)
    kmeans1 = KMeans(n_clusters=clusters, n_init=100, max_iter=300).fit(weightvect1)
    labels1 = kmeans1.labels_

    weightvect2 = getweightvectors(ae_file, filter_csv, channely, n_neurons, m_neurons)
    kmeans2 = KMeans(n_clusters=clusters, n_init=100, max_iter=300).fit(weightvect2)
    labels2 = kmeans2.labels_

    ARI = metrics.adjusted_rand_score(labels1, labels2)
    return ARI

def assigningclusters(ae_file, filter_csv, channel_number, n_neurons, m_neurons, labellist):
    ''' This function is used to determine which event goes into which cluster. This is helpful for making cumulative AE plots
    Inputs: ae_file(.txt): Voltage-data series
            filter_csv(.csv): filter_csv of important events
            channel_number (float): channel number ranging from 0 to 3
            n_neurons (float): number of rows in SOM
            m_neurons (float): number of columns in SOM
            labellist: label list from kmeans of weight vectors. Assigns a label to a node.
    Outputs: Clusterlist (list): list of cluster labels that correspond to each AE event. Useful in Cumulative AE plots.
    '''
    featurevect1 = find_feature_vectors(ae_file, filter_csv, 10 ** -7, channel_number)
    npfeaturevect1 = np.array(featurevect1)
    reshapelabels = np.reshape(labellist, (n_neurons, m_neurons))

    # Initialization and training
    som = MiniSom(n_neurons, m_neurons, npfeaturevect1.shape[1], sigma=1.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=0)

    som.pca_weights_init(npfeaturevect1)
    som.train(npfeaturevect1, 1000, verbose=True)  # random training

    plt.figure(figsize=(9, 9))

    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background

    clusterlist = []
    # need to get cluster list
    for i in range(len(npfeaturevect1)):
        x = som.winner(npfeaturevect1[i])
        getlabel = reshapelabels[x[0]][x[1]]
        clusterlist.append(getlabel)

    plt.colorbar()
    plt.title(['Distance Map', str(filter_csv), 'Channel' + str(channel_number)])

    plt.show()
    return clusterlist

def separatedAEplot(filter_csv, labels,channel_number):
    ''' Generates an AE vs. Stress plots and separates the events by cluster
    Inputs: filter_csv (.csv file): A filter csv file of important events
            labels (array-like): cluster labels from the kmeans funtion
            channel_number (int): channel number starting with index of 0
    Outputs: An AE vs Stress plot where the clusters are separated by color
    Note: Depending on the labels you put, this function will only work for 10 clusters. More can be added by adding another
          stressesX variable, an elif statement in the for loop, another aeX block, and axX.scatter statement.
    '''
    stresses0 = []
    stresses1 = []
    stresses2 = []
    stresses3 = []
    stresses4 = []
    stresses5 = []
    stresses6 = []
    stresses7 = []
    stresses8 = []
    stresses9 = []
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
        elif labels[i] == 5:
            stresses5.append(stress)
        elif labels[i] == 6:
            stresses6.append(stress)
        elif labels[i] == 7:
            stresses7.append(stress)
        elif labels[i] == 8:
            stresses8.append(stress)
        elif labels[i] == 9:
            stresses9.append(stress)

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

    ae5 = range(len(stresses5))
    ae5 = np.array(ae5)
    aecumulative5 = ae5 / len(df['Adjusted_Stress_MPa'])

    ae6 = range(len(stresses6))
    ae6 = np.array(ae6)
    aecumulative6 = ae6 / len(df['Adjusted_Stress_MPa'])

    ae7 = range(len(stresses7))
    ae7 = np.array(ae7)
    aecumulative7 = ae7 / len(df['Adjusted_Stress_MPa'])

    ae8 = range(len(stresses8))
    ae8 = np.array(ae8)
    aecumulative8 = ae8 / len(df['Adjusted_Stress_MPa'])

    ae9 = range(len(stresses9))
    ae9 = np.array(ae9)
    aecumulative9 = ae9 / len(df['Adjusted_Stress_MPa'])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(stresses0,aecumulative0, c='tab:blue')
    ax1.scatter(stresses1,aecumulative1, c ='tab:orange')
    ax1.scatter(stresses2,aecumulative2, c = 'tab:green')
    ax1.scatter(stresses3, aecumulative3, c = 'tab:red')
    ax1.scatter(stresses4, aecumulative4, c='tab:purple')
    ax1.scatter(stresses5, aecumulative5, c='tab:brown')
    ax1.scatter(stresses6, aecumulative6, c='tab:pink')
    ax1.scatter(stresses7, aecumulative7, c='tab:gray')
    ax1.scatter(stresses8, aecumulative8, c='tab:olive')
    ax1.scatter(stresses9, aecumulative9, c='tab:cyan')
    plt.xlabel('Stress (MPa)')
    plt.ylabel('Normalized AE Cumulative Score')
    plt.title('Normalized AE Plot '+str(filter_csv)+' Channel '+str(channel_number))
    plt.show()

def DBIandARIplot(weight_vects,experiment,channel):
    '''Takes a weight_vector list and generates a Davies-Bouldin Plot and Average ARI plot
    Inputs: weight_vects (array-like): Takes a list of weight vectors and generates a Davies-Bouldin Plot
            experiment (integer): experiment number from 'Average_deOliveira_ARI.csv' file
            channel(int): channel number ranging from 0 to 3
    Outputs: A plot is generated that is the Davies-Bouldin index vs. the number of clusters and ARI vs. Number of Clusters
    '''
    df = pandas.read_csv('Average_deOliveira_ARI.csv')
    avgari = df[str(experiment)]
    npavgari = np.array(avgari)
    clustervector = list(range(2,11))
    dbscores2 = []

    for i in clustervector:
        kmeans2 = KMeans(n_clusters=i, n_init=100, max_iter=300).fit(weight_vects)
        labels2 = kmeans2.labels_

        davies_bouldin_score2 = davies_bouldin_score(weight_vects, labels2)
        dbscores2.append(davies_bouldin_score2)

    plt.figure()
    plt.plot(clustervector, dbscores2, label = 'deOliveira DBI')
    plt.ylabel('Davies Bouldin Index')
    plt.xlabel('Number of Clusters')
    plt.title(('DBI and ARI plot '+ 'Experiment ' +str(experiment)+ ' Channel '+ str(channel)))

    ax2 = plt.twinx()
    ax2.plot(clustervector, npavgari, label = 'Average ARI', c = 'r')
    ax2.set_ylabel('Adjusted Rand Index')
    fig = plt.gcf()
    fig.legend(loc = 'lower left')
    plt.show()
