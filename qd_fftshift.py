import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#------- for time evolution ------------------------------------------
class propogateK:
    def __init__(self,dt,y,m,hbar):
        j = complex(0,1)
        self.opr = np.exp(-0.5*j*dt*hbar*y*y/m)
        
class propogateV:
    def __init__(self,dt,y,hbar):
        j = complex(0,1)
        self.opr = np.exp(-0.5*j*dt*y/hbar)
        
class wavefn:
    def __init__(self,xgrid,m,omega,hbar,k0):
        wfc0 = 0.0
        j = complex(0,1)
        a_ = 1.0
        k0_ = -39.0
        #self.wavefc = ((a_ * np.sqrt(np.pi))**(-0.5) * np.exp(-0.5 * ((xgrid - wfc0) * 1. / a_) ** 2 + 1j *xgrid* k0_))
        #self.wavefc = 0.5*np.exp(-0.5*(xgrid - wfc0)**2 - 0.5*j*xgrid*k0_)
        self.wavefc = (m*omega/(np.pi*hbar))**(1/4)*np.exp(-0.5*m*omega/hbar*(xgrid - wfc0)**2)*np.exp(-j*k0_*xgrid)
        
np.set_printoptions(threshold=np.inf)
#----- initial conditions and definitions ---------------------------
m = 1.0
hbar = 1.0
ke = 1.0
omega = np.sqrt(ke/m)

tsteps = 100000  #no. of time steps
dt = 0.001
xmax = 20.0
res = 512     #512 not 500 because it is power of 2
dx  = 2.0*xmax/res

xgrid = np.arange(-xmax,xmax+dx,dx)

k0 = -1.0*np.pi/(dx)
dk = 2.0*np.pi/(res*dx)
print(k0)
kgrid = np.zeros(len(xgrid))
for i in range(len(xgrid)):
    k = k0 + i*dk
    kgrid[i] = k

#print(kgrid)

x0 = 0.0
vgrid = np.zeros(len(xgrid))
"""
for i in range(len(xgrid)):
    if xgrid[i] > 3.0 and xgrid[i] < 3.5:
        vgrid[i] = 55.0
"""
vgrid = 0.5*ke* (xgrid - x0)**2

wfn = wavefn(xgrid,m,omega,hbar,k0)
wfc = wfn.wavefc #wfc = np.exp(-((xgrid - wfc0)**2)/2) Gaussian wavepacket

#----for plots---
fig, ax = plt.subplots()
plt.xlim(-5,5)
plt.ylim(0,1)
#---------------
density = np.abs(wfc)**2
plt.plot(xgrid,vgrid,label="Potential")
plt.subplot(1,2,1)
plt.xlim(-5,5)
plt.ylim(0,2)
plt.plot(xgrid,0.1*vgrid,label="Potential")
plt.plot(xgrid,np.abs(wfc),linestyle='--',label="Initial Density")
#plt.plot(xgrid,np.imag(wfc),linestyle='dotted',label="Initial Density")
plt.xlabel('Position (a.u.)')
plt.legend(loc='upper center')

wfc1 = np.fft.fftshift(wfc)
plt.subplot(1,2,2)
#plt.xlim(-5,5)
plt.plot(kgrid,np.abs(wfc1),linestyle='dotted',label="Initial Density")
#plt.plot(kgrid,np.imag(wfc1),linestyle='dotted',label="Initial Density")
plt.xlabel('momentum (a.u.)')


Rp = propogateV(dt,vgrid,hbar)
Kp = propogateK(dt,kgrid,m,hbar)


t = 0
for i in range (1,tsteps):
    
    wfc = wfc * Rp.opr

    wfc1 = np.fft.fftshift(wfc)
    wfc1 = wfc1 * Kp.opr
    wfc = np.fft.ifftshift(wfc1)

    wfc = wfc * Rp.opr

    if i % 100 == 0:
        wfc1 = np.fft.fftshift(wfc)
        
        plt.subplot(1,2,1)
        plt.xlim(-5,5)
        plt.ylim(0,2)
        line1 = plt.plot(xgrid,np.abs(wfc),color='r',linestyle='solid',label="Initial Density")
        #line2 = plt.plot(xgrid,np.imag(wfc),color='b',linestyle='dashed',label="Initial Density")
        plt.ylabel('real space')
        
        plt.subplot(1,2,2)
        #plt.xlim(-5,5)
        line3 = plt.plot(kgrid,np.abs(wfc1),color='r',linestyle=(0,(5,10)))
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


plt.show()
