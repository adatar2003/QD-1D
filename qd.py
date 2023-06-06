import numpy as np
import math
import numpy.polynomial.hermite as Herm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline
import csv
#------- read potential energy from file -----------------------------

string = 'data.csv'
pos_x, V_x = np.loadtxt(string, unpack = True, usecols = (0,1))
#print (pos_x, V_x)

def potential(xval_):
    v = CubicSpline(pos_x, V_x, bc_type='natural')
    v_ = v(xval_)
    return v_


#----- initial conditions and definitions ---------------------------
m = 0.94117647*1837.1530
hbar = 1.0
omega = 0.024 #np.sqrt(ke/m)
ke = m*omega*omega

n_states = 80  #same as in hm_dvr code

tsteps = 1000  #no. of time steps
dt = 0.02418884 #fs 0.001
xmax = 20.0
res = 512     #512 not 500 because it is power of 2
dx  = 2.0*xmax/res

xgrid = np.arange(-xmax,xmax+dx,dx)

k0 = -1.0*np.pi/(dx)
print(k0)
dk = 2.0*np.pi/(res*dx)

kgrid = np.zeros(len(xgrid))
for i in range(len(xgrid)):
    k = k0 + i*dk
    kgrid[i] = k

x0 = 1.88
vgrid = np.zeros(len(xgrid))

for i in range(len(xgrid)):
    xval = xgrid[i]
    vgrid[i] = potential(xval)

f = open("overlp.txt", "a")

#vgrid = 0.5*ke* (xgrid - x0)**2

#----------for harmonic oscillator basis and overlap with wavefn-----
def hermite(x, n):
    xi = np.sqrt(m*omega/hbar)*x
    herm_coeffs = np.zeros(n+1)
    herm_coeffs[n] = 1
    return Herm.hermval(xi, herm_coeffs)
  
def stationary_state(x, n):
    xi = np.sqrt(m*omega/hbar)*x
    prefactor = 1.0/np.sqrt(2.**n * math.factorial(n)) * (m*omega/(np.pi*hbar))**(0.25)
    phi = prefactor * np.exp(-1.0*xi**2 / 2) * hermite(x, n)
    #print(phi)
    return phi

def overlap_integral(xgrid, wfc, c):

    overlap = 0.0
    """for n in range(n_states):
        x = xgrid[0] - 1.88
        overlap += c[n]*wfc[0]*stationary_state(x, n)"""

    #x = xgrid[len(xgrid)] - 1.88
    #overlap += wfc[len(xgrid)]*stationary_state(x, m, w, hbar, n)
    
    for i in range(1, len(xgrid) - 1):
        for n in range(n_states):
            x = xgrid[i] - 1.88
            overlap += 2*c[n]*wfc[i]*stationary_state(x, n)
        
    overlap *= dx/2
        
    return overlap

#------- for time evolution ------------------------------------------
class propogateK:
    def __init__(self, y):
        j = complex(0,1)
        self.opr = np.exp(-0.5*j*dt*hbar*y*y/m)
        
class propogateV:
    def __init__(self, y):
        j = complex(0,1)
        self.opr = np.exp(-0.5*j*dt*y/hbar)

cj = np.loadtxt('ground_0.txt', unpack = True)
class wavefn:
    def __init__(self, xgrid):
        wfc0 = 1.88
        j = complex(0,1)
        a_ = 1.0
        k0_ = -40.21238596594935
        wfc = np.zeros(len(xgrid))
        #self.wavefc = 0.5*np.exp(-0.5*(xgrid - wfc0)**2 - j*xgrid*(k0_+0.5))
        for i in range(len(xgrid)):
            x = xgrid[i] - wfc0
            for i1 in range(n_states):
                wfc[i] += cj[i1]*stationary_state(x, i1) #- j*x*(k0_+0.5)

            #wfc[i] -= np.exp(-j*k0_*x)
        self.wavefc = wfc 

        
#------------------------------------------------------------------------------------------------------------        
np.set_printoptions(threshold=np.inf)

wfn = wavefn(xgrid)
wfc = wfn.wavefc #wfc = np.exp(-((xgrid - wfc0)**2)/2) Gaussian wavepacket

#wfc = np.abs(wfc)/np.sum(np.abs(wfc))

f.write("Ground state wfc\n")

for j in range(7):
    #overlap = 0.0
    c = np.loadtxt('core-exc.txt', unpack = True, usecols = (j))
    #for i in range(n_states):
    overlap = overlap_integral(xgrid, wfc, c)
        #overlap  += integral
    print(j, overlap, file=f)
"""
for i in range(10):
    overlap = overlap_integral(xgrid, wfc, i)
    print(i, overlap, file=f)
"""
#----for plots---
fig, ax = plt.subplots()
#---------------
density = np.abs(wfc)**2
plt.plot(xgrid,vgrid,label="Potential")
plt.subplot(1,2,1)
plt.xlim(-5,10)
plt.ylim(0,2)
plt.plot(xgrid,0.1*vgrid,label="Potential")
plt.plot(xgrid,np.abs(wfc),linestyle='--',label="Initial Density")
#plt.plot(xgrid,np.imag(wfc),linestyle='dotted',label="Initial Density")
plt.xlabel('Position (a.u.)')
plt.legend(loc='upper center')

wfc1 = np.fft.fft(wfc)
plt.subplot(1,2,2)
plt.xlim(-15,15)
plt.plot(kgrid,np.abs(wfc1),linestyle='dotted',label="Initial Density")
#plt.plot(kgrid,np.imag(wfc1),linestyle='dotted',label="Initial Density")
plt.xlabel('momentum (a.u.)')


Rp = propogateV(vgrid)
Kp = propogateK(kgrid)


t = 0
for i in range (1,tsteps):
    
    wfc = wfc * Rp.opr

    wfc1 = np.fft.fft(wfc)
    wfc1 = wfc1 * Kp.opr
    wfc = np.fft.ifft(wfc1)

    wfc = wfc * Rp.opr

    if i % 100 == 0:
        wfc1 = np.fft.fft(wfc)
        
        plt.subplot(121)
        plt.xlim(-5,10)
        plt.ylim(0,2)
        line1 = plt.plot(xgrid,np.abs(wfc),color='r',linestyle='solid',label="Initial Density")
        #line2 = plt.plot(xgrid,np.imag(wfc),color='b',linestyle='dashed',label="Initial Density")
        plt.ylabel('real space')
        
        plt.subplot(122)
        plt.xlim(-15,15)
        line3 = plt.plot(kgrid,np.abs(wfc1),color='r',linestyle='dotted')
        #line4 = plt.plot(kgrid,np.imag(wfc1),color='b',linestyle=(0,(1,10)))
        plt.ylabel('momentum space')
        plt.xlabel('momentum (a.u.)')
        plt.tight_layout()
        plt.pause(0.1)
        line1.pop(0).remove()
        #line2.pop(0).remove()
        line3.pop(0).remove()
        #line4.pop(0).remove()
        

        kinetic = np.sum(0.5*np.conj(wfc1)*wfc1*kgrid*kgrid/m)/np.sum(np.abs(wfc1)**2)
        potential = np.real(np.sum(vgrid*np.conj(wfc)*wfc))/np.sum(np.abs(wfc)**2)
        print('kinetic',kinetic,'potential',potential,'total',kinetic+potential)
    t = t + dt    

print(t)
norm_wfc = np.abs(wfc)/np.sum(np.abs(wfc))
f.write("Excited state wfc\n")

#c[:,0], c[:,1], c[:,2], c[:,3], c[:,4], c[:,5], c[:,6] = np.loadtxt('core-exc.txt', unpack = True, usecols =(0,1,2,3,4,5,6))

for j in range(n_states):
    #overlap = 0.0
    c = np.loadtxt('core-exc.txt', unpack = True, usecols = (j))
    #for i in range(n_states):
    overlap = overlap_integral(xgrid, np.abs(wfc), c)
        #overlap  += integral
    print(j, overlap, file=f)

f.close()

plt.show()
