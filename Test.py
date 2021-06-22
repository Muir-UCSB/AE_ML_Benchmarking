import numpy as np
import matplotlib as plt
import pandas
from ae_measure2 import *

read_ae_file2('test_13_waveforms.txt', 0)

####
## reading CSV file for Feature calculation
####

df = pandas.read_csv('201015_test_13_filter.csv')
print(df)
'''
risetime = df['Start_channel1']
print(risetime)

logrisetime = np.log(risetime)
print(logrisetime)

x,y = filter_ae('test_13_waveforms.txt','201015_test_13_filter.csv',1,)
print(x)
print(y)

amplitude = x
maxamp = np.max(amplitude)
print(maxamp)

counts = len(y)
print(counts)

logcounts = np.log(counts)
print(logcounts)

# I am commenting this out for now as it is likely not needed
'''
total_signal_length = 102.4 #microseconds
dt = 10**-7 #samplingrate
start_time = df['Start_channel1'][0]
print('start_time' , start_time)

energy = df['Energy_Ch1'][0]
print('energy' , energy)

ln_energy = np.log(energy)
print('ln_energy' , ln_energy)

(sig, ev) = read_ae_file2('test_13_waveforms.txt', 0, sig_length=1024)
print(sig)


#I am going to start with event 45 to get a base case and eventually automate the process
'''
max_amplitude = np.max(sig[0][44])
print(max_amplitude)
'''

waveform_channel_array = np.array(sig[44])
print(waveform_channel_array)

max_amplitude = np.max(waveform_channel_array)
print(max_amplitude)

index_location_max = np.argmax(waveform_channel_array)
print(index_location_max)

max_location_time = index_location_max/10
print(max_location_time)
#eventually remove 10... make it into words

rise_time = max_location_time - start_time
print(rise_time)

duration = total_signal_length - start_time
print(duration)
# Go back after writing and remove floating numbers... make it a parameter in a function

decay_time = duration - rise_time
print(decay_time)

risingpart = sig[44][:index_location_max]

plt.plot(risingpart)
plt.show()

wr, zr = fft(dt,risingpart, low_pass= None, high_pass= None)
risingfrequency = get_average_freq(wr,zr, low_pass= None, high_pass=None)
print(risingfrequency)

decaypart = sig[44][index_location_max:]
plt.plot(decaypart)
plt.show()

wd,zd = fft(dt,decaypart,low_pass=None, high_pass=None)
decayfrequency = get_average_freq(wd,zd,low_pass=None, high_pass=None)
print(decayfrequency)

wa,za = fft(dt,sig[44], low_pass=None, high_pass=None)
avgfrequency = get_average_freq(wa,za,low_pass=None,high_pass=None)
print(avgfrequency)

Rln = np.log(rise_time)
RDln = np.log(rise_time/duration)
ARln = np.log(max_amplitude/rise_time)
ADTln = np.log(max_amplitude/decay_time)
AFln = np.log(max_amplitude/avgfrequency)

fvector = [Rln, avgfrequency, risingfrequency, ln_energy, RDln, ARln, ADTln, AFln ]
print(fvector)

