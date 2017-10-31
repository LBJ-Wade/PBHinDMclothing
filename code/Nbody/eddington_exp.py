#Eddington distribution function for the exponential truncated density profile
# rho ~ exp(-r/r_0) at large radii

import numpy as np
from scipy.interpolate import interp1d

fdat = np.loadtxt("Distribution_exp.dat")
func = interp1d(fdat[:,0], fdat[:,1], kind='linear', bounds_error=False, fill_value='extrapolate')

G_N = 4.302e-3 #(pc/solar mass) (km/s)^2
M_PBH = 30.0
r_tr = 0.0063*(M_PBH**(1.0/3.0))

A = 3*M_PBH/(8*np.pi*r_tr**3)
B = G_N*M_PBH/r_tr

#NB: x = r/r_tr

#Density profile
def rhoDM_scalar(x):
    if (x < 1):
        return A*x**(-3.0/2.0)
    elif (x >= 1):
        return A*np.exp(3.0/2.0)*np.exp(-x*3.0/2.0)

rhoDM = np.vectorize(rhoDM_scalar)

def Psi_tr():
    return G_N*(54.0*M_PBH + 120.0*A*np.pi*r_tr**3)/(27.0 *r_tr)

#Relative potential (Psi = 0 at infinity, Psi = infinity at r = 0)
def Psi_scalar(x):
    if (x < 1):
        return Psi_tr() + B*(1.0 + 1.0/x - 2.0*x**0.5)
    elif (x >= 1):
        return G_N*(54.0*M_PBH + 8*A*np.pi*r_tr**3*(29.0 - 2.0*np.exp(-1.5*(x-1.0))*(4 + 3*x)))/(27.0*r_tr*x)
        
Psi = np.vectorize(Psi_scalar)

#Maximum speed at a given radius x = r/r_tr
def vmax(x):
    return np.sqrt(2.0*Psi(x))

    
#Speed distribution f(v) at a given radius r
def f_scalar(r, v):
    x = r/r_tr
    if (v >= vmax(x)):
        return 0.0
    else:
        return 4.0*np.pi*(v**2)*func(Psi(x) - 0.5*v**2)/rhoDM(x)

f = np.vectorize(f_scalar)