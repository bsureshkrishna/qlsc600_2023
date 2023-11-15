import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import gamma
import math

def plot_signal_spectrum(signal, fs, plotornot = 1):
    # Compute the FFT
    fft_result = np.fft.fft(signal-np.mean(signal))

    # Compute the frequency bins
    freqs = np.fft.fftfreq(len(signal), d=1/fs)

    # Compute the magnitude (absolute value of the FFT)
    magnitude = np.abs(fft_result)

    if(plotornot==1):
        # Plotting
        plt.plot(freqs, magnitude)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title('Spectrum')

    return freqs, magnitude
