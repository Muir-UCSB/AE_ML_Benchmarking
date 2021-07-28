from deOliviera_functions import *
import time

##Starting again to figure out bayesian optimization
#Acquiring feature vectors
channel_number = 0
features = find_feature_vectors('210308-1_waveforms.txt','210308-1_filter.csv',10**-7, channel_number)
npfeaturevect1 = np.array(features)
data = npfeaturevect1

#Parameter settings
n_neurons = 8
m_neurons = 8
input_len = data.shape[1] #Note: number of dimensions
sigma = 1
learning_rate = 0.5
iterations = 640000

#initialization
som = MiniSom(n_neurons,m_neurons,input_len,sigma = sigma, learning_rate = learning_rate)
som.random_weights_init(data)

#training
start_time = time.time()
som.train_random(data,iterations)
elapsed_time = time.time() - start_time
print(elapsed_time, "seconds")

#plotting
from pylab import plot,axis,show,pcolor,colorbar,bone

plt.figure(figsize = (9,9))
bone()
pcolor(som.distance_map().T)
colorbar()
plt.title('Test Plot')
axis([0,som._weights.shape[0],0,som._weights.shape[1]])
show()

#more imports
from hyperopt import fmin, tpe, hp
import time

def train_som(n_neurons, m_neurons, input_len,sigma,learning_rate):
    ''' Training function to condense code'''
    # initialization
    som = MiniSom(n_neurons, m_neurons, input_len, sigma=sigma, learning_rate=learning_rate) #add a random seed?
    som.random_weights_init(data)

    # training
    start_time = time.time()
    som.train_random(data, iterations)
    elapsed_time = time.time() - start_time
    print(elapsed_time, "seconds")
    return som

def plot_som(som):
    ''' Plot function to condense code'''
    plt.figure(figsize=(9, 9))
    bone()
    pcolor(som.distance_map().T)
    colorbar()
    plt.title('Test Plot')
    axis([0, som._weights.shape[0], 0, som._weights.shape[1]])
    show()

#set hyperparameters
rows_data = data.shape[0]
n_neurons = int(np.sqrt(5*np.sqrt(rows_data)))
x=n_neurons
m_neurons = n_neurons
y=m_neurons
input_len = data.shape[1] #Note: dimensions of input space
sigma = 0.003 #Note: Intentionally poorly set hyperparameters
learning_rate = 5
iterations = 1000
print ("n_neurons is {}".format(n_neurons))

som = train_som(n_neurons,m_neurons,input_len,sigma,learning_rate)
print("n_neurons: {}\nm_neurons: {}\ninput_len: {}\nsigma {}\nlearning_rate: {}".format(n_neurons,m_neurons,input_len,sigma,learning_rate))

plot_som(som)

#tuning sigma by reducing quant error
start_time = time.time()
best = fmin(
    fn=lambda sig: MiniSom(x=x,
                             y=x,
                             input_len=input_len,
                             sigma=sig,
                             learning_rate=learning_rate
                             ).quantization_error(data),
    space=hp.uniform("sig", 0.001, n_neurons/2.01),
    algo=tpe.suggest,
    max_evals=200)
#Note: The hyperopt.fmin function will minimize an objective function given a space of features to manipulate
#The objective function is the quantization error, we convert it to a function by converting it to a lambda function.
#tpe.suggest is tree-of-parzen estimatiors: models P(x|y) and P(y) where x represents hyper parameters and y is score
elapsed_time = time.time() - start_time
print(elapsed_time, "seconds")
print(best)

#hyperparameters
sigma = best['sig'] #updates hyperparameter of sigma
print("n_neurons: {}\nm_neurons: {}\ninput_len: {}\nsigma {}\nlearning_rate: {}".format(n_neurons,m_neurons,input_len,sigma,learning_rate))
som = train_som(n_neurons,m_neurons,input_len,sigma,learning_rate)
plot_som(som)

#more importing
from hyperopt import Trials, STATUS_OK
space={
        'sig': hp.uniform('sig',0.001,4),
        'learning_rate': hp.uniform('learning_rate',0.001,4)
}
#Note: To optimize 2 variables, needs to be in a dictionary. Here it is called space.
def som_fn(space):
    sig = space['sig']
    learning_rate = space['learning_rate']
    val = MiniSom(x=x,
                  y=x,
                  input_len=input_len,
                  sigma = sig,
                  learning_rate=learning_rate
                  ).quantization_error(data)
    print(val)
    return{'loss':val,'status':STATUS_OK}
trials = Trials()
best=fmin(fn=som_fn,
          space=space,
          algo=tpe.suggest,
          max_evals=10000,
          trials=trials)
print('best: {}'.format(best))

for i, trial in enumerate(trials.trials[:2]):
    print(i,trial)

sigma = best['sig']
learning_rate=best['learning_rate']
print("n_neurons: {}\nm_neurons: {}\ninput_len: {}\nsigma {}\nlearning_rate: {}".format(n_neurons,m_neurons,input_len,sigma,learning_rate))

som = train_som(x,y,input_len,sigma,learning_rate)
plot_som(som)