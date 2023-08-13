# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:48:14 2023

@author: Victor

Qubit dynamics simulation (QuDyn).
This library comprises functions to simulate the qubit behaviour solving
the Langevin differential equations.
       
# v1.1 - Documented some functions.
         Reordered functions to improve clarity.

"""

import numpy as np
import glob
from scipy.integrate import odeint as ode
from PIL import Image

def rabi_decay_ode(state, t, ɣ1, ɣ2, Ω, wd, wq, α):
    
    """
    System of differential equations for the temporal evolution of the
    two-level system. It is obtained from Langevin master equation so it takes
    decay and dephase effects into account. 
    
    Parameters
    -------
    state: geometric coordinates of the spin vector in the Bloch sphere.
    t: numpy array or float. In seconds.
       If numpy array, the function returns the TLS state at each time so the
       array is the time over which the TLS evolves. 
       If float, it is the time at which one wants to calculate the TLS state.
    
    ɣ1: float. In Hertz. Decay rate of the TLS.
    ɣ2: float. In Hertz. Dephase rate of the TLS.
    Ω: numpy array or float. In Hertz
       Coupling strength between the microwave driving signal and the TLS.
       If numpy array is provided the function returns the TLS state for each
       element of the array.
    
    wd: float. In Hertz. Frequency of the microwave driving signal.
    wq: float. In Hertz. Frequency of the TLS.
    
    α: float. Radians. Relative phase between the microwave driving signal and
       the TLS.
    
    Returns
    -------
    state differential: [dx, dy, dz] - Differential of the TLS state to be
                        integrated with odeint() method from scipy.integrate.
    """
    
    x, y, z = state
    Δ = wq - wd # detuning
    
    dx = - ɣ1 * x - Ω * np.real(np.exp(-1j*α)) * y - Ω * np.imag(np.exp(-1j*α)) * z + ɣ1
    
    dy = Ω * np.real(np.exp(1j*α)) * x - (ɣ1 / 2 + ɣ2) * y + Δ * z - (Ω / 2) * np.real(np.exp(1j*α))
    
    dz = - Ω * np.imag(np.exp(1j*α)) * x - Δ * y  - (ɣ1 / 2 + ɣ2) * z + (Ω / 2) * np.imag(np.exp(1j*α))
    
    return [dx,dy,dz]

def rabi_decay(t, wd, wq, ɣ1, ɣ2, Ω, ΨΨ, α = 0.):
    
    """
    Function to calculate state of a two-level system when a microwave
    driving signal is applied. It is obtained from Langevin master equation so
    it takes decay and dephase effects into account. The function takes the
    differential state returned by rabi_decay_ode() and integrates it with
    odeint() method from scipy.integrate.
    
    Parameters
    -------
    t: numpy array or float. In seconds.
       If numpy array, the function returns the TLS state at each time so the
       array is the time over which the TLS evolves. 
       If float, it is the time at which one wants to calculate the TLS state.
       
    wd: float. In Hertz. Frequency of the microwave driving signal.
    wq: float. In Hertz. Frequency of the TLS.
    
    ɣ1: float. In Hertz. Decay rate of the TLS.
    ɣ2: float. In Hertz. Dephase rate of the TLS.
    
    Ω: numpy array or float. In Hertz
       Coupling strength between the microwave driving signal and the TLS.
       If numpy array is provided the function returns the TLS state for each
       element of the array.
       
    ΨΨ: two dimensional numpy array. No units. It is the intial state of the
        TLS expressed as a density matrix.
    
    α: float. Radians. Relative phase between the microwave driving signal and
       the TLS.
    
    Returns
    -------
    rho00: numpy array or float. No units. (0,0) component of the density
           matrix for the final state. It represents the probability amplitude
           for |0> state. Then, the probability amplitude for |1> can be
           obtained as 1 - rho00.
    
    rho01: complex numpy array or complex float. Transversal component of the
           density matrix for the final state. It represent the losses ocurred
           over temporal evolution due to decay and dephase effects.
           
    Ωr: numpy array or float. In Hertz. Generalised Rabi Frequency.
        if Ω is a numpy array then a will be a numpy array.
    """
    
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

def rabi(t, wd, wq, Ω, Ψ0 = np.array([[1,0]]), α = 0.):
    
    """
    Function to calculate the final state of an ideal two-level system
    when a microwave driving signal is applied. No decay nor dephase are
    considered.
    
    Parameters
    -------
    t: numpy array or float. In seconds.
       If numpy array, the function returns the TLS state at each time so the
       array is the time over which the TLS evolves. 
       If float, it is the time at which one wants to calculate the TLS state.
       
    wd: float. In Hertz. Frequency of the microwave driving signal.
    wq: float. In Hertz. Frequency of the TLS.
    Ω: numpy array or float. In Hertz
       Coupling strength between the microwave driving signal and the TLS.
       If numpy array is provided the function returns the TLS state for each
       element of the array.
       
    Ψ0: two dimensional numpy array. No units. It is the intial state of the
        TLS.
    
    α: float. Radians. Relative phase between the microwave driving signal and
       the TLS.
    
    Returns
    -------
    a: numpy array or float. No units. Probability amplitude for |0> state.
       if t or Ω are numpy arrays then a will be a numpy array.
    b: numpy array or float. No units. Probability amplitude for |1> state.
       if t or Ω are numpy arrays then a will be a numpy array.
    Ωr: numpy array or float. Hertz. Generalised Rabi Frequency.
        if Ω is a numpy array then a will be a numpy array.
    """
        
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

def rabi_amp(t, wd, wq, Ω, Ψ0 = np.array([[1,0]]), α = 0.):
    Δ = wq - wd # detuning
    Ωr = np.sqrt(Ω**2 + Δ**2)
    P = ((abs(Ψ0[0][1])**2)*(np.cos(Ωr*t/2))**2 +
    (1/Ωr**2)*((abs(Ψ0[0][1])*Δ)**2 + (abs(Ψ0[0][0])*Ω)**2 +
     2*Δ*Ω*np.real(Ψ0[0][0]*np.conjugate(Ψ0[0][1])*np.exp(1j*α)))*(np.sin(Ωr*t/2))**2 -
    (2*Ω/Ωr)*np.imag(Ψ0[0][0]*np.conjugate(Ψ0[0][1])*np.exp(1j*α))*np.cos(Ωr*t/2)*np.sin(Ωr*t/2))
    return P



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