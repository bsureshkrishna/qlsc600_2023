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

def generate_samtone(t,spl,frequency, phasevalue=0, amfrequency=10):
    #supply amplitdue in sound pressure level (decibels), frequency in hz
    amplitude = amplitude_from_spl(spl) #amplitude in micropascals, since reference of 0dB = 20 micropascals
    signal = amplitude*(1+np.sin(2*np.pi*amfrequency*t-(np.pi)/2))*np.sin(2.0*np.pi*frequency*t+phasevalue) #a simple sine-wave to use as signal
    return signal

def timeplot(signal,fs,colorvar='blue'):
    plt.plot((1/fs)*np.arange(0,len(signal)),signal,color=colorvar)

def generate_gammatone(timeax,f=440,b=50,n=4,a=1,phi=0):
    gammapart=a*timeax**(n-1)*np.exp(-2.0*np.pi*b*timeax)
    gammapart=2*gammapart/np.trapz(gammapart)
    cospart=np.cos(2.0*np.pi*f*timeax) # a hack normalization to make convolution output "preserve" amplitude
    gammatone_impresp=cospart*gammapart #impulse response of gammatone filter
    return gammatone_impresp

#set filter parameters
a=1
n=4
b=50
f=440
phi=0
fs=50000
t=np.arange(0,1,1/fs) #1 seconds

#set signal parameters

spl=38 #in decibels
signal=generate_tone(t,spl,f)

gammapart=a*t**(n-1)*np.exp(-2.0*np.pi*b*t) #change b to change (reduce) the temporal extent; b acts like the bandwidth of the filter.
gammapart=2*gammapart/np.trapz(gammapart)
sinepart=np.sin(2.0*np.pi*f*t) # a hack normalization to make convolution output "preserve" amplitude
gammatone_impresp=sinepart*gammapart #impulse response of gammatone filter

fig, axs = plt.subplots(3, 2, layout='constrained')
plt.sca(axs[0,0]); plt.plot(t,gammapart);
plt.xlabel('Time (s)'); plt.ylabel('Magnitude'); plt.xlim(0,0.05)
plt.sca(axs[0,1]); plot_signal_spectrum(gammapart,fs)
plt.xlim(-1000,1000)
plt.sca(axs[1,0]); plt.plot(t,sinepart);
plt.xlabel('Time (s)'); plt.ylabel('Magnitude'); plt.xlim(0,0.05)
plt.sca(axs[1,1]); plot_signal_spectrum(sinepart,fs)
plt.xlim(-1000,1000)
plt.sca(axs[2,0]); plt.plot(t,gammatone_impresp)
plt.xlim(0,0.05); plt.xlabel('Time (s)'); plt.ylabel('Magnitude')
plt.sca(axs[2,1]); plot_signal_spectrum(gammatone_impresp,fs)
plt.xlim(-1000,1000)
plt.show()

gammatone_output = convolve(signal,gammatone_impresp)

frequencies=np.arange(100,900,100)
output_across_frequencies=np.zeros(len(frequencies))
for i in range(len(frequencies)):
    temp=generate_tone(t,spl,frequencies[i])
    output_across_frequencies[i]=np.max(convolve(temp,gammatone_impresp))

fig, axs = plt.subplots(1, 2, layout='constrained')
plt.sca(axs[0]); timeplot(signal,fs,'green'); timeplot(gammatone_output,fs,'blue')
plt.xlim(0,0.2); plt.xlabel('Time (s)'); plt.ylabel('Amplitude')

plt.sca(axs[1]); plt.plot(frequencies,output_across_frequencies)
plt.xlabel('Frequency'); plt.ylabel('Amplitude')

plt.show()

def lowpass_create(b=2000, duration=0.01):
    #rough bandwidth, 1/b is rough temporal extent of impulse response
    #set filter parameters
    a=1
    n=4
    #10 milliseconds will do, since impulse respnse will be 0 after that for all plausible values..
    #10 ms duration is roughly a cutoff of 100 hz, btu real cutoffs wold be well above 1000 hz, so the
    #impulse response willl be much briefer in druation (why).
    #reducing the duration will sbstnatially speed up convolution operations with this impulse response. why ?
    lowpass_time=np.arange(0,duration,1/fs)
    lowpass_impresp=a*np.power(lowpass_time,(n-1))*np.exp(-2.0*np.pi*b*lowpass_time)
    lowpass_impresp=lowpass_impresp/np.trapz(lowpass_impresp)
    return lowpass_impresp

b=2000 #can vary this and see how bandwidth changes
lowpass_impresp=lowpass_create(b)
fig, axs = plt.subplots(1, 2, layout='constrained')
plt.sca(axs[0]); timeplot(lowpass_impresp,fs); plt.xlim([0,0.01]); plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
plt.sca(axs[1]); a,b = plot_signal_spectrum(lowpass_impresp,fs); plt.xlim(-5000,5000)
plt.show()

def simulate_inhomogeneous_poisson(times, rate_function, numtrials=1):
    # Example usage:
    # times = np.linspace(0, 10, 10000)  # Define the time points
    # rate_function = np.sin(2.0*np.pi*0.5*times) * 25 + 25  # Define the rate function, ensuring it is non-negative
    # events = simulate_inhomogeneous_poisson(times, rate_function)
    # Now 'events' contains the event times of the inhomogeneous Poisson process
    inhomogeneous_poisson_events_times=[]
    spike_counts=np.zeros(numtrials)
    for trial in np.arange(0,numtrials):
      # Integrate the rate function to get the compensator (cumulative rate function)
      compensator = np.cumsum(rate_function) * (times[1] - times[0])
      # Generate a standard Poisson process with rate 1
      n_events = np.random.poisson(compensator[-1])
      standard_poisson_events = np.sort(np.random.uniform(0, compensator[-1], n_events))

      # Apply the inverse of the cumulative rate function to the event times
      # We find where each standard Poisson event would fall in the compensator
      inhomogeneous_poisson_events = np.searchsorted(compensator, standard_poisson_events, side='left')
      inhomogeneous_poisson_events_times.append([times[inhomogeneous_poisson_events]])
      spike_counts[trial]=len(inhomogeneous_poisson_events)

    return spike_counts, inhomogeneous_poisson_events_times

import copy

def fullmodel(signal, fs, gammatone_centerfreq=440, gammatone_b=50, lowpass_b=2000, spikesout=0, numtrials=20, Rmult=3000):
    #this takes in any signal and returns either an auditory nerve rate-function or spike-train
    gammatone_timeax=np.arange(0,0.5,1/fs) #500 ms duration should be more than enough.
    gammatone_impresp=generate_gammatone(gammatone_timeax,f=gammatone_centerfreq, b=gammatone_b)
    gammatone_output=np.convolve(signal,gammatone_impresp)
    rectified_output=copy.deepcopy(gammatone_output); rectified_output[rectified_output<=0]=0
    lowpass_impresp=lowpass_create()
    lowpass_output=np.convolve(rectified_output,lowpass_impresp)
    if spikesout==0:
        return lowpass_output, rectified_output, gammatone_output
    else:
        return simulate_inhomogeneous_poisson((1/fs)*np.arange(0,len(lowpass_output)), Rmult*lowpass_output, numtrials)

spl=50 #in decibels
freqs=[440,2000,8000] #in hz
t=np.arange(0,1,1/fs) #1 seconds

fig, axs = plt.subplots(len(freqs), 3, layout='constrained')

for idx,f in enumerate(freqs):
  signal=generate_tone(t,spl,f)
  lowpass_output, rectified_output, gammatone_output = fullmodel(signal,fs, gammatone_centerfreq=f) #not getting spike-train output
  plt.sca(axs[idx,0]); timeplot(gammatone_output,fs); plt.xlim([0,0.02])
  plt.sca(axs[idx,1]); timeplot(rectified_output,fs); plt.xlim([0,0.02])
  plt.sca(axs[idx,2]); timeplot(lowpass_output,fs); plt.xlim([0,0.02])
  if idx==0:
    plt.sca(axs[idx,0]);plt.title('Gammatone output');plt.annotate('440 Hz',(0.001,0.005))
    plt.sca(axs[idx,1]);plt.title('Rectified output')
    plt.sca(axs[idx,2]);plt.title('Low-pass filter output')
  if idx==1:
    plt.sca(axs[idx,0]);plt.xlabel('Time (s)');plt.annotate('2000 Hz',(0.001,0.005))
  if idx==2:
    plt.sca(axs[idx,0]);plt.xlabel('Time (s)');plt.annotate('8000 Hz',(0.001,0.005))
    plt.sca(axs[idx,1]);plt.xlabel('Time (s)');
    plt.sca(axs[idx,2]);plt.xlabel('Time (s)');


plt.show()

spl=50 #in decibels
t=np.arange(0,1,1/fs) #1 seconds

fig, axs = plt.subplots(2, 2, layout='constrained')

f=3000;
signal=generate_tone(t,spl,f)
lowpass_output, rectified_output, gammatone_output = fullmodel(signal,fs, gammatone_centerfreq=f) #not getting spike-train output
plt.sca(axs[0,0]); timeplot(signal,fs); plt.xlim([0,0.02])
plt.sca(axs[0,1]); timeplot(lowpass_output,fs); plt.xlim([0,0.02])

signal=generate_samtone(t,spl,f)
lowpass_output, rectified_output, gammatone_output = fullmodel(signal,fs, gammatone_centerfreq=f) #not getting spike-train output
plt.sca(axs[1,0]); timeplot(signal,fs); plt.xlim([0,0.2])
plt.sca(axs[1,1]); timeplot(lowpass_output,fs); plt.xlim([0,0.2])

plt.show()

spl=80 #in decibels
freqs=np.arange(4200,5800,100)
freqs=np.arange(1000,2000,100)
t=np.arange(0,1,1/fs) #1 seconds

spikecounts=np.zeros(len(freqs))
spikecounts_sd=np.zeros(len(freqs)) #dont do spikeconts = spikeconts_sd.. python copy is weird !
spikecounts_var=np.zeros(len(freqs))

for i in range(len(freqs)):
  signal=generate_tone(t,spl,freqs[i])
  lowpass_output, nonlinearity_output, gammatone_output = fullmodel(signal,fs)
  #sc, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=5000, gammatone_b=600)
  sc, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=1500, gammatone_b=200)

  spikecounts[i]=np.mean(sc)
  spikecounts_sd[i]=np.std(sc) #/np.sqrt(len(sc))
  spikecounts_var[i]=np.var(sc)


fig, axs = plt.subplots(1, 2, layout='constrained')
plt.sca(axs[0]); plt.errorbar(freqs,spikecounts,spikecounts_sd); plt.xlabel('Frequency (Hz)'); plt.ylabel('spike-count')
plt.sca(axs[1]); axs[1].scatter(spikecounts,spikecounts_var); axs[1].axline((0,0),slope=1); plt.xlabel('Mean spike-count'); plt.ylabel('Variance of spike-count')
plt.show()

spls=np.arange(40,90,5) #in decibels
f=5000
t=np.arange(0,1,1/fs) #1 seconds

spikecounts=np.zeros(len(spls))
spikecounts_sd=np.zeros(len(spls)) #dont do spikeconts = spikeconts_sd.. python copy is weird !
spikecounts_var=np.zeros(len(spls))

for i in range(len(spls)):
  signal=generate_tone(t,spls[i],f)
  lowpass_output, nonlinearity_output, gammatone_output = fullmodel(signal,fs)
  #sc, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=5000, gammatone_b=600)
  sc, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=f, gammatone_b=600)

  spikecounts[i]=np.mean(sc)
  spikecounts_sd[i]=np.std(sc) #/np.sqrt(len(sc))
  spikecounts_var[i]=np.var(sc)

#fig, axs = plt.subplots(1, 2, layout='constrained')
#plt.sca(axs[0]);
plt.errorbar(spls,spikecounts,spikecounts_sd); plt.xlabel('SPL (dB)'); plt.ylabel('spike-count')
#plt.sca(axs[1]);
plt.show()

from sklearn.metrics import roc_curve, auc

spls=np.arange(40,90,10) #in decibels
increments=np.arange(1,20,3) #spl increments; COARSE RIGHT NOW. MAKE finer for question 4
f=5000
t=np.arange(0,1,1/fs) #1 seconds

for i in range(len(spls)):
  signal=generate_tone(t,spls[i],f)
  lowpass_output, nonlinearity_output, gammatone_output = fullmodel(signal,fs)
  #sc, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=5000, gammatone_b=600)
  sc, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=f, gammatone_b=600)

  percor=np.zeros(len(increments))

  for i1 in range(len(increments)):
      signal=generate_tone(t,spls[i]+increments[i1],f)
      lowpass_output, nonlinearity_output, gammatone_output = fullmodel(signal,fs)
      #sc, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=5000, gammatone_b=600)
      sc1, spiketrain = fullmodel(signal,fs, spikesout=1, gammatone_centerfreq=f, gammatone_b=600)

      spikevalues=np.concatenate([sc,sc1])
      spikelabels=np.concatenate([np.zeros(len(sc)),np.ones(len(sc1))])
      # Compute ROC curve and area
      fpr, tpr, thresholds = roc_curve(spikelabels, spikevalues)
      percor[i1] = auc(fpr, tpr)

  plt.plot(increments,percor);
  plt.xlabel('Increments (dB)')
  plt.ylabel('2-i 2-AFC Percentage Correct')
  plt.title('Increment detection performance')
  #plt.legend(loc="lower right")


plt.show()

