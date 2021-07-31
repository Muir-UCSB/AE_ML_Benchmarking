'''
Author: Caelin Muir
Contact: muir@ucsb.edu
Version: 201008

Completed library for HHT. Needed because the instantaneous frequency function in
PyHHT is dubious
'''

#Imports
import numpy as np
import pylab as pl
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import binned_statistic
from PyEMD import EEMD
from PyEMD import EMD


def instantaneous_frequency(signal, duration, fs):
    '''
    :param signal - array like: Original signal, input is an IMF
    :param duration - float: Duration of signal
    :param fs - float: Frequency of sampling of signal

    :return inst: instantaneous_frequency, see scipy.signal.hilbert documentation for calculation
    :return envelope: Amplitude of envelope. Corresponds to a(t) in eqn. 14 of Huang et al. 1999

    '''
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = (np.diff(instantaneous_phase) /(2.0*np.pi) * fs)
    thresh_indicies = inst_freq < 0
    inst_freq[thresh_indicies] = 0


    return inst_freq, amplitude_envelope

def plot_hilbert(signal, duration, fs):
    '''
    Plots envelope and instantaneous frequency

    :param inst_freq:
    :param amplitude_envelope:
    :param t array-like: Time

    Plots input signal with envelope, and instantaneous frequency
    '''
    inst_freq, amplitude_envelope = instantaneous_frequency(signal, duration, fs)
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    fig = pl.figure()

    ax0 = fig.add_subplot(211)
    ax0.plot(t, signal, label='signal')
    ax0.plot(t, amplitude_envelope, label='envelope')
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Amplitude (arb. units)")
    ax0.legend()

    ax1 = fig.add_subplot(212)
    ax1.plot(t[1:], inst_freq)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Instantaneous Frequency (arb. units)")
    pl.show()


def get_Imfs(signal):
    imf = EMD().emd(signal)
    return imf


def get_eImfs(signal, num_trials=100):
    eemd = EEMD(trials=num_trials) # Assign EEMD to `eemd` variable
    emd = eemd.EMD
    emd.extrema_detection="parabol"
    eIMFs = eemd.eemd(signal) # Execute EEMD on signal1
    return eIMFs


def get_spectrogram(imfs, duration, fs, t, plot=False, lower=0, upper=1.6,):
    '''
    Makes Hilbert-Huang spectrogram using PyEEMD

    :param (np.array): IMFs of original signal
    :param duration (float): Duration of signal in microseconds
    :param fs (float): Frequency of sampling of signal in MHz
    :param t (np.array): 1d array of equally spaced time values

    :return spectrogram: 3xn array of the spectrogram s.t. [[time, instantaneous_frequency, amplitude]]
    '''


    spectrogram = []
    for imf in imfs:
        inst_freq, amplitude_envelope = instantaneous_frequency(imf, duration, fs)
        for i, freq in enumerate(inst_freq):
            point = [t[i+1], freq, amplitude_envelope[i+1]] # inst_freq only defined for
                                                        # t[1:], not t[0:]
            spectrogram.append(point)
    spectrogram = np.array(spectrogram)


    time = spectrogram[:,0]
    frequency = spectrogram[:,1]
    amplitude = spectrogram[:,2]

    if plot == True:
        # need to find a way to fix this
        norm_amp = amplitude/(max(amplitude)-max(amplitude)*.1)

        fig1, ax1 = pl.subplots()
        tcf = ax1.tricontourf(time, frequency, norm_amp, 1000) #1000 is resolution of triangles
        pl.ylim(lower, upper)
        ax1.set_xlabel('Time ($\mu$s)')
        ax1.set_ylabel('Frequency (MHz)')
        x = pl.colorbar(tcf, ticks=range(2), label = 'Intensity (arb. units)')
        pl.show()


    return spectrogram


def get_marginal_spectrum(spectrogram, smooth = False, dw = .02):
    '''
    :param spectrogram (np.array): spectrogram from get_spectrogram function
    :param smooth (bool): If true, sums powers within frequency width dw
    :param dw (float): Frequency width to sum over in MHz

    :return unique_freq (np.array): values of unsmoothed frequencies in spectrogram
    :return power (np.array): values of unsmoothed powers corresponding to unique_freq
    :return ave_freq (np.array): Values of average frequency of a bin
    :return total_power (np.array): Values of summed powers for all frequencies in a bin

    '''

    spectrogram = spectrogram[spectrogram[:,1].argsort()].T # sorts according to second column transposes
    time = spectrogram[0]
    frequency = spectrogram[1]
    amplitude = spectrogram[2]
    unique_freq = np.unique(spectrogram[1])

    power = []
    for freq in unique_freq:
        power.append(np.sum(amplitude[np.where(frequency==freq)]))
    #power = np.square(np.array(power))

    if smooth == True:
        range = max(unique_freq) - min(unique_freq)
        num_bins = int(range/dw)
        total_power, bin_edge, _ = binned_statistic(unique_freq, power, statistic=sum, bins=num_bins)
        ave_freq = (bin_edge-dw/2)[1:] # sums

        return ave_freq, total_power

    return unique_freq, power

def band_pass(freq, power, low_pass=None, high_pass=None):
    '''
    :param freq (np.array): list of frequencies
    :param power (np.array): list of powers corresponding to frequency
    :param low_pass (float): all frequencies below this will be cut off
    param high_pass (float): all frequencies above this will be cut off
    '''
    if low_pass is not None:
        power = power[np.where(freq > low_pass)]
        freq = freq[np.where(freq > low_pass)]
    if high_pass is not None:
        power = power[np.where(freq < high_pass)]
        freq = freq[np.where(freq < high_pass)]

    return freq, power
