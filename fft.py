import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftshift

plt.style.use('seaborn-poster')


# sampling rate
sr = 2000
# sampling interval
ts = 2.0/sr
t = np.arange(-1,1,ts)

j = complex(0,1)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)#*np.exp(-j*1*t)
"""
freq = 4
x += np.sin(2*np.pi*freq*t)

freq = 7   
x += 0.5* np.sin(2*np.pi*freq*t)
"""
#j = complex(0,1)
#x = np.exp(-0.5*(t)**2)*np.exp(-j*3500*t)
#plt.figure(figsize = (8, 6))
#plt.plot(t, x, 'r')
#plt.ylabel('Amplitude')

#plt.show()


#from numpy.fft import fftshift, ifftshift

X = fftshift(fft(x))
#X = fft(x)

N = len(X)
print(N)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
#plt.xlim(-10, 10)

plt.subplot(122)
plt.plot(t, x, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
#plt.xlim(-1, 1)
plt.tight_layout()
plt.show()
