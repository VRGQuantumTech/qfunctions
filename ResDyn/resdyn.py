# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:26:35 2023

@author: Victor Rollano

Resonator dynamics simulation (ResDyn).
This library comprises functions to simulate the resonator behaviour solving
the Langevin differential equations.
       
# v0.1 - In development.
         Functions to simulate a naked resonator finished.
         Developing resonator/spin dynamic functions.
        
"""

import numpy as np
import matplotlib.pyplot as plt

def wave_in(t, A):
    return 1j*A

def out(t, A, wc, wd, k):
    d = (1j*(wc - wd) + k)
    b = 1j*k*A / d
    b = b*(np.exp(-1*d*t) - 1) + 1j*A
    
    return b

def abs_out(t, A, wc, wd, k):
    
    return abs(out(t, A, wc, wd, k))

def decay(t, t0, A, wc, wd, k):
    d = (1j*(wc - wd) + k)
    b = 1j*k*A / d
    b = b * np.exp(-1*d*(t - t0))
    
    return b

def abs_decay(t, t0, A, wc, wd, k):
    
    return abs(decay(t, t0, A, wc, wd, k))
#%%

t = np.linspace(0, 1e-4, 1000) # s
tp = np.linspace(1e-4, 2e-4, 1000) # s
wc = 5.0e9 # Hz
k = 2*np.pi*10e3 # rad/s

wd = np.linspace(wc - 1e6, wc + 1e6, 1000) # Hz
A = np.zeros((len(wd), len(t)))
             
for i in range(len(wd)):
    A[i,:] = np.random.normal(1, 0.05, len(t))

data = np.zeros((len(wd), len(t) + len(tp)))

for i in range(len(wd)):
    data[i,:len(t)] = abs_out(t, A[i,:], wc, wd[i], k)
    data[i,len(tp):] = abs_decay(tp, t[-1], np.mean(A), wc, wd[i], k)

time =  np.append(t,tp)
plt.figure()
plt.pcolor(time*1e6, (wd - wc)*1e-6, data, cmap = 'plasma')
plt.tick_params(direction = 'in')
plt.xlabel('$time$ $(\mu s)$', fontsize=14.0)
plt.ylabel('$\Delta = \omega_c - \omega_d$ $(MHz)$', fontsize=14.0)
cb = plt.colorbar()
cb.set_label(label = '$<b_{out}>$ $(V)$')
plt.show()

#%% 

"""
def cavsp_dynamics(state, t, wc, ws, wd, kc, ki, g, gamma,eps):
    
    a, sm, sz  = state
    
    da = -1j*(wc - wd)*a - (kc + ki)*a + g*sm - np.sqrt(kc)*1j*eps
    
    dsm = -1j*(ws/2 - wd)*sm - 2*g*sm*a*np.exp(-1j*wd*t) - gamma*sm
    
    dsz = -1j*sz + g*(a * np.conj)"""
