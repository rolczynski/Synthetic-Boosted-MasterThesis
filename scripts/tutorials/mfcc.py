"""Tutorials
https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
https://github.com/danijel3/PyHTK/blob/master/python-notebooks/HTKFeaturesExplained.ipynb
"""
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from python_speech_features import sigproc
from python_speech_features import get_filterbanks
from python_speech_features import mfcc
from python_speech_features import lifter
from matplotlib import pyplot as plt

fs, audio = wav.read('doc/figures/sample.wav')
plt.plot(audio, color='black')
"""
Because audio is a non stationary process, the FFT will produce distortions. To overcome this 
we can assume that the audio is a stationary process for a short periods of time. Each audio 
frame will be the same size as the FFT. Also we want the frames to overlap. 
"""
signal = sigproc.preemphasis(audio, coeff=0.97)
winlen = 0.025 * fs
winstep = 0.010 * fs
raw_frames = sigproc.framesig(signal, winlen, winstep, np.hamming)
plt.imshow(raw_frames.T, aspect='auto', origin='lower')
plt.show()
"""
Chunk of data is repeated in time. That typically results in a major 
discontinuity at the edges of the chunk. Narrow windows width results
lower mas (circles over) especially for low freq. Windows make sure 
that the data at the edges are zero, so there is no discontinuity.

However multiplication in the time domain is convolution in the frequency 
domain and that results in widening of spectral lines and also in side lobes.
Trade offs between main lobe width and side lobe spacing and height 
"""
winfunc = np.hamming
function_values = np.expand_dims(winfunc(winlen), axis=0)
win = np.repeat(function_values, repeats=len(raw_frames), axis=0)
frames = raw_frames * win
plt.imshow(win.T, aspect='auto', origin='lower')
plt.show()
plt.imshow(frames.T, aspect='auto', origin='lower')
plt.show()
"""
Now we will convert the audio, which is currently in the time domain, 
to frequency domain. The FFT assumes the audio to be periodic and continues. 
By framing the signal we assured the audio to be periodic. To make the audio 
continues, we apply a window function on every frame. 
"""
NFFT = 512
pspec = sigproc.powspec(frames, NFFT)
plt.imshow(pspec.T, aspect='auto', origin='lower')
plt.show()
"""
Here we compute the MEL-spaced filterbank and then pass the framed audio through them. 
That will give us information about the power in each frequency band. The filters can be 
constructed for any frequency band but for our example we will look on the entire sampled band.

What spessial with the MEL-spaced filterbank is the spacing between the filters which 
grows exponentially with frequency. The filterbank can be made for any frequency band. 
Here we will compute the filterbank for the entire frequency band.
"""
NFILT = 26
fb = get_filterbanks(NFILT, NFFT, samplerate=fs, lowfreq=0, highfreq=fs/2)
plt.plot(fb.T)
plt.show()
feat = np.dot(pspec, fb.T)
feat = np.where(feat == 0, np.finfo(float).eps, feat)  # if feat is zero, we get problems with log
feat = np.log(feat)  # compute the filterbank energies
plt.imshow(feat.T, aspect='auto', origin='lower')
plt.show()
"""
The final step in generating the MFCC is to use the Discrete Cosine Transform (DCT). 
We will use the DCT-III. This type of DCT will extract high frequency and low frequency changes in the the signal.
"""
numcep = 26
cepstral_coefficents = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
plt.imshow(cepstral_coefficents.T, aspect='auto', origin='lower')
plt.show()
cepstral_coefficents = lifter(cepstral_coefficents, L=22)
plt.imshow(cepstral_coefficents.T, aspect='auto', origin='lower')
plt.show()
"""
And one-liner.
"""
features = mfcc(audio, fs, winlen=0.025, winstep=0.01, numcep=26, nfilt=26, nfft=512, appendEnergy=False, winfunc=np.hamming)
plt.imshow(features.T, aspect='auto', origin='lower')
plt.show()
