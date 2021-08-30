# BreathAnalyzer

To be added

The recorded sounds
were first highpass filtered with a Butterworth filter of order
5 and cutoff frequency of 200Hz to remove low frequency
noises including motion artifacts and heart sounds. The
filtered sounds were segmented into windows of 20ms in
duration with 75% overlap between the adjacent windows.
In each window, the logarithm of the variance of the signal,
LogV ar, was calculated. The median of the LogV ar values
of all windows was used as a threshold to classify the
windows into sound or silent windows.
