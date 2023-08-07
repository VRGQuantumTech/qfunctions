# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:48:14 2023

@author: Victor

v1.0 - Qubit dynamics simulation (QuDyn).
This library comprises functions to simulate the qubit temporal evolution
solving the Langevin differential equations. 

"""

import numpy as np
import glob
from scipy.integrate import odeint as ode
from PIL import Image

def rabi(t, wd, wq, Ω, Ψ0 = np.array([[1,0]]), α = 0.):
        
    Δ = wq - wd # detuning
    Ωr = np.sqrt(Ω**2 + Δ**2) # Generalized Rabi frequency
    if Ωr != 0: 
        a = (Ψ0[0][0]*np.cos(0.5*Ωr*t) + 1j*(Δ*Ψ0[0][0]/(Ωr) -
            (Ω*Ψ0[0][1]/Ωr)*np.exp(1j*α))*np.sin(0.5*Ωr*t)) # probability amplitude of |0>
        b = (Ψ0[0][1]*np.cos(0.5*Ωr*t) - 1j*(Δ*Ψ0[0][1]/(Ωr) +
            (Ω*Ψ0[0][0]/Ωr)*np.exp(1j*α))*np.sin(0.5*Ωr*t)) # probability amplitude of |1>
    else:
        
        a = np.array([Ψ0[0][0] for i in range(len(t))])
        b = np.array([Ψ0[0][1] for i in range(len(t))])
        
    return a, b, Ωr


def dephase(t, wq, Ψ0 = np.array([[1,0]])):
    
    a = Ψ0[0][0]*np.exp(-1j*wq*t/2)
    b = Ψ0[0][1]*np.exp(1j*wq*t/2)
    
    return a, b

def rabi_amp(t, wd, wq, Ω, Ψ0 = np.array([[1,0]]), α = 0.):
    Δ = wq - wd # detuning
    Ωr = np.sqrt(Ω**2 + Δ**2)
    P = ((abs(Ψ0[0][1])**2)*(np.cos(Ωr*t/2))**2 +
    (1/Ωr**2)*((abs(Ψ0[0][1])*Δ)**2 + (abs(Ψ0[0][0])*Ω)**2 +
     2*Δ*Ω*np.real(Ψ0[0][0]*np.conjugate(Ψ0[0][1])*np.exp(1j*α)))*(np.sin(Ωr*t/2))**2 -
    (2*Ω/Ωr)*np.imag(Ψ0[0][0]*np.conjugate(Ψ0[0][1])*np.exp(1j*α))*np.cos(Ωr*t/2)*np.sin(Ωr*t/2))
    return P

def rabi_decay_ode(state, t, ɣ1, ɣ2, Ω, wd, wq, α):
    
    x, y, z = state
    Δ = wq - wd # detuning
    
    dx = - ɣ1 * x - Ω * np.real(np.exp(-1j*α)) * y - Ω * np.imag(np.exp(-1j*α)) * z + ɣ1
    
    dy = Ω * np.real(np.exp(1j*α)) * x - (ɣ1 / 2 + ɣ2) * y + Δ * z - (Ω / 2) * np.real(np.exp(1j*α))
    
    dz = - Ω * np.imag(np.exp(1j*α)) * x - Δ * y  - (ɣ1 / 2 + ɣ2) * z + (Ω / 2) * np.imag(np.exp(1j*α))
    
    return [dx,dy,dz]

def rabi_decay(t, wd, wq, ɣ1, ɣ2, Ω, ΨΨ, α = 0.):
    
    Δ = wq - wd # detuning
    Ωr = np.sqrt(Ω**2 + Δ**2)
    
    y0 = [ΨΨ[0,0], np.imag(ΨΨ[0,1]), np.real(ΨΨ[0,1])]
    p = (ɣ1, ɣ2, Ω, wd, wq, α)
    result = ode(rabi_decay_ode, y0, t, p)

    rho00 = result[:,0]

    rho01_real = result[:,2]
    rho01_imag = result[:,1]
    
    rho01 = rho01_real + 1j*rho01_imag

    return rho00, rho01, Ωr

def tobsphere(a, b):
    
    φ = np.arctan2(np.imag(b),np.real(b))
    
    if type(φ) == np.ndarray:
        for i in range(len(φ)):
            if φ[i] < 0: φ[i] = φ[i] + 2*np.pi
    else:
        if φ < 0: φ = φ + 2*np.pi
        
    θ =  2 * np.arctan2(abs(b),abs(a))
    
    r = np.sqrt(abs(a)**2 + abs(b)**2)
    print(r)
    xp = r * np.cos(φ) * np.sin(θ)
    yp = r * np.sin(φ) * np.sin(θ)
    zp = r * np.cos(θ)
    
    return [xp, yp, zp]

def densitytobsphere(rho00, rho01):
    
    z = 2*rho00 - 1
    x = 2*np.real(rho01)
    y = 2*np.imag(rho01)
    
    return [x, y, z]

def density(rho00, rho01):
    
    return np.array([[rho00, rho01], [np.conj(rho01), 1-rho00]])
  
def check_phase_continuity(b_phase):
    if type(b_phase) == np.ndarray:
        index = [i+2 for i in range(len(b_phase)-2)]
        for i in index:
            if np.sign(b_phase[i]) != np.sign(b_phase[i-1]):
                if (abs(b_phase[i] - b_phase[i-1]) > 
                    5*abs(b_phase[i-1] - b_phase[i-2])):
                    b_phase[i] = -1*b_phase[i]
    return b_phase

def make_gif(frame_folder, name = 'ramseygif.gif', image_name = '*.png'):
    frames = [Image.open(image) for image in 
              sorted(glob.glob(f"{frame_folder}/" + image_name))]

    frame_one = frames[0]
    frame_one.save(frame_folder + name, format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
