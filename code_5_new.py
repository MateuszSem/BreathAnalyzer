import urllib
import scipy.io.wavfile
import pydub
import numpy as np
import matplotlib.pyplot as plt
from numpy import fft as fft

rate,audData=scipy.io.wavfile.read("snoring1.wav")

#wav length
audData.shape[0] / rate
audData.shape[1]
channel1=audData[:,0] #left channel for stereo sound

if np.issubdtype(channel1.dtype, np.integer):
    type_info = np.iinfo(channel1.dtype)
else:
    type_info = np.finfo(channel1.dtype)

max_amp = float(type_info.max)
channel1 = channel1 / max_amp


channel2=audData[:,1] #right channel for stereo sound
audData.dtype
scipy.io.wavfile.write("snoring11.wav", rate, channel1)

mono=np.sum(audData.astype(float), axis=1)/2
scipy.io.wavfile.write("monosnoring.wav", rate, mono) #idea to make it mono rather than stereo.
np.sum(channel1.astype(float)**2) #energy
1.0/(2*(channel1.size)+1)*np.sum(channel1.astype(float)**2)/rate #power



# for numb in channel1:
    # if numb <= 0.05:
#        print ('Apnea')
#        continue
#    if numb >= 0.05 and numb <= 4000:
#        print ('breath')
#        continue
#    if  numb >= 4000:
#        print ('Snoring')
#        continue
#    print (numb)


    
time = np.arange(0, float(audData.shape[0]), 1) / rate #in seconds
plt.figure(1)
plt.subplot(211)
plt.plot(time, channel1, linewidth=0.04,) #plot of amplitude with more visible lines. 
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.subplot(212)
# plt.plot(time, channel2, linewidth=0.04,)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')




fourier=fft.fft(channel1)
plt.plot(time, fourier,linewidth=0.04, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')


n = len(channel1)
fourier = fourier[0:int(((n)/2))] #problem 1


fourier = fourier / float(n) #scaling process Magnitude won't be dependent on length

freqArray = np.arange(0, (n/2), 1.0) * (rate*1.0/n); #freq at each point

plt.plot(freqArray/1000, 10*np.log10(fourier),linewidth=0.04) #problem 2
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')




plt.figure(2, figsize=(8,6))
plt.subplot(211)
Pxx, freqs, bins, im = plt.specgram(channel1, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
plt.subplot(212)
Pxx, freqs, bins, im = plt.specgram(channel2, Fs=rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity (dB)')

plt.figure(3, figsize=(8,6))
plt.hist(channel1,bins='auto')
plt.title('histogram with bins that were auto assigned.')
plt.show()