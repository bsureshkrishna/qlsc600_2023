import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, convolve
from scipy.stats import gamma
import math
import sklearn

def plot_signal_spectrum(signal, fs, plotornot = 1):
    # Compute the FFT
    fft_result = np.fft.fft(signal)

    # Compute the frequency bins
    freqs = np.fft.fftfreq(len(signal), d=1/fs)

    # Compute the magnitude (absolute value of the FFT)
    magnitude = np.abs(fft_result)

    if(plotornot==1):
        # Plotting
        sorted_indices = np.argsort(freqs)
        magnitude=magnitude[sorted_indices]
        freqs=freqs[sorted_indices]
        plt.plot(freqs, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title('Spectrum')

    return freqs, magnitude

def amplitude_from_spl(spl):
    amplitude = np.power(10,(spl/20))*20*1e-6 #amplitude in pascals, since reference of 0dB = 20 micropascals
    return amplitude

def generate_tone(t,spl,frequency, phasevalue=0):
    #supply amplitdue in sound pressure level (decibels), frequency in hz
    amplitude = amplitude_from_spl(spl) #amplitude in micropascals, since reference of 0dB = 20 micropascals
    signal = amplitude*np.sin(2.0*np.pi*frequency*t+phasevalue) #a simple sine-wave to use as signal
    return signal

def nonlinearity(amplitudes, x0=10, Rmax=100, sigma=40):
    x0=-1.0*amplitude_from_spl(x0)
    if sigma<=0:
      sigma=sigma+0.1
    if Rmax<sigma:
      Rmax=sigma+10
    km=x0**2*((Rmax/sigma)-1) #km is defined in terms of sr.
    x = np.array(amplitudes)  # Ensure x is a NumPy array
    return np.where(x < x0, 0, Rmax * (x - x0)**2 / ((x - x0)**2 + km))
  
def timeplot(signal,fs,colorvar='blue'):
    plt.plot((1/fs)*np.arange(0,len(signal)),signal,color=colorvar)

def generate_gammatone(timeax,f=440,b=50,n=4,a=1,phi=0):
    gammapart=a*timeax**(n-1)*np.exp(-2.0*np.pi*b*timeax)
    gammapart=2*gammapart/np.trapz(gammapart)
    cospart=np.cos(2.0*np.pi*f*timeax) # a hack normalization to make convolution output "preserve" amplitude
    gammatone_impresp=cospart*gammapart #impulse response of gammatone filter
    return gammatone_impresp

def generate_samtone(t,spl,frequency, phasevalue=0, amfrequency=10):
    #supply amplitdue in sound pressure level (decibels), frequency in hz
    amplitude = amplitude_from_spl(spl) #amplitude in micropascals, since reference of 0dB = 20 micropascals
    signal = amplitude*(1+np.sin(2*np.pi*amfrequency*t-(np.pi)/2))*np.sin(2.0*np.pi*frequency*t+phasevalue) #a simple sine-wave to use as signal
    return signal
