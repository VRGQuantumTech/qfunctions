# -*- coding: utf-8 -*-
"""
QUBIT OPERATION THEORY (QuDyn example)
Created on Fri Mar 31 08:38:50 2023

This script is an example on how to use QuDyn library.

Important parameters:
    
Δ: Frequency detuning between qubit and microwave driving signal
wq: Qubit frequency
wd: Microwave driving signal
Ω: Effective amplitude of the driving signal
ɣ1: Qubit decay rate
ɣ2: Qubit dephase rate
Ψ0: Qubit initial state
Δt: Duration of the driving signal

This example covers:
    
    1. Decay vs waiting time after sending a driving signal to change the qubit
    state from |0> to |1>.
    2. Rabi time sweep. This consists on increasing the duration of the driving
    signal and measuring the resulting state of the qubit. The amplitude of the
    signal is fixed.
    3. Rabi amplitude sweep. This consists on incresing the duration of the
    driving signal and measuring the resulting state of the qubit. The duration
    of the signal is fixed.

@author: Victor Rollano
"""
import os
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import qudyn
from loadpath import pth
from matplotlib.lines import Line2D


#%% DECAY VS TIME PLOTTING IN THE BLOCH SPHERE - GIF

Δ = 0.0e6
wq = 5.00e9
wd = wq + Δ
Ω = 40e6
ɣ1 = 1/(10e-6) 
ɣ2 = 1/1e-6 
Ψ0 = np.array([[1,0]], dtype = complex)/np.sqrt(1)
ΨΨ = Ψ0*Ψ0.transpose()
Δt = 40e-6
t_cal = np.linspace(0, 100e-9, 1000)

rho00, rho01, _ = qudyn.rabi_decay(t_cal, wd, wq, ɣ1, ɣ2, Ω, ΨΨ = ΨΨ)

plt.plot(t_cal, rho00)
plt.plot(t_cal, 1-rho00)

t_pi = np.pi/Ω

plt.axvline(t_pi)
plt.show()

t0 = np.linspace(0, t_pi, 50)
t1 = np.linspace(0, Δt, 200)

rho00 = np.zeros((len(t1)), dtype=complex)
rho01 = np.zeros((len(t1)), dtype=complex)
rho00_0 = np.zeros((len(t1)), dtype=complex).tolist()
rho00_1 = np.zeros((len(t1)), dtype=complex).tolist()
rho01_0 = np.zeros((len(t1)), dtype=complex).tolist()
rho01_1 = np.zeros((len(t1)), dtype=complex).tolist()

for j in range(len(t1)):
    
    rho00_0[j], rho01_0[j], _ = qudyn.rabi_decay(t0, wd, wq, ɣ1, ɣ2, Ω, ΨΨ = ΨΨ)
    taux = np.linspace(0, t1[j], 50)
    rho = qudyn.density(rho00_0[j][-1], rho01_0[j][-1])
    rho00_1[j], rho01_1[j], _ = qudyn.rabi_decay(taux, wd, wq, ɣ1, ɣ2, 0, ΨΨ = rho)
    rho00[j] = rho00_1[j][-1]
    rho01[j] = rho01_1[j][-1]

envelope = np.exp(-ɣ1*t1)

path_to_save = pth.folder()
path_to_save = (path_to_save + 
    '/DecayGif_deltatime%.2fus'%(Δt*1e6) + 
    '_detuning%.2fMHZ_coupling%.2fMHz_decay%.2fMHz_dephase%.2f/'%(
    Δ*1e-6, Ω*1e-6, ɣ1*1e-6, ɣ2*1e-6))

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
    
for i in range(len(rho00)-1):
    
    fig = plt.figure(constrained_layout = True, figsize=(14,5), dpi = 100)
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(t1*1e6, abs(1-rho00), color = 'royalblue')
    ax1.scatter(t1[i]*1e6,abs(1-rho00[i]), color = 'royalblue', s=30)
    ax1.plot(t1*1e6, envelope, color = 'grey', ls = '--')
    ax1.tick_params(direction = 'in')
    ax1.set_xlabel('$Decay$ $time$ $(\mu s)$', fontsize=18.0)
    ax1.set_ylabel(r'$\mathcal{P}$ ' + r'$_{\| 1 \rangle}$', fontsize=20.0)
    ax1.set_xlim((0, t1.max()*1e6))
    
    ax2 = fig.add_subplot(1,2,2, projection = '3d')
    bloch = qt.Bloch(fig = fig, axes = ax2)
    bloch.view = [-60, 30]
    bloch.make_sphere()
    bloch.frame_color = 'black'
    bloch.sphere_color = 'white'
    bloch.zlpos = [1.2,-1.4]
    left, bottom, width, height = [0.98, 0.05, 0.05, 0.9]
    bloch.point_color = ['tomato', 'goldenrod', 'blue']
    bloch.point_marker = ['o', 'o', 'o']

    bloch.add_points(qudyn.densitytobsphere(np.array(rho00_0[i]),
                    np.array(rho01_0[i])), alpha = 0.5)
    bloch.add_points(qudyn.densitytobsphere(np.array(rho00_1[i]),
                    np.array(rho01_1[i])), alpha = 0.5)
    bloch.add_points(qudyn.densitytobsphere(rho00[i+1],rho01[i+1]))
    bloch.render()
    ax2.set_box_aspect([1,1,1])
    text = ('Decay \n' +
            '$\Omega = %.2f$ $MHz$\n'%(Ω*1e-6) +
            '$\gamma_1 = %.2f$ $MHz$\n'%(ɣ1*1e-6) +
            '$\gamma_2 = %.2f$ $MHz$'%(ɣ2*1e-6))
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', label='$\pi-pulse$', markersize=10),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='goldenrod', label='$wait$ $time$', markersize=10),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='$measured$ $state$', markersize=10),
                  Line2D([0], [0], linestyle='--', color='gray', label='$Envelope$')]

    ax2.legend(handles=legend_elements, bbox_to_anchor=(0.150, 0.1))
    
    plt.figtext(0.525, 0.8, text, fontsize = 13, bbox = dict(facecolor='white', edgecolor='black', boxstyle='round', alpha = 0.15))
    if i+1<10:plt.savefig(path_to_save + '/ds00%i.png'%(i+1), dpi=100, bbox_inches='tight')
    elif i+1<100 and i+1>=10:plt.savefig(path_to_save + '/ds0%i.png'%(i+1), dpi=100, bbox_inches='tight')
    else: plt.savefig(path_to_save + '/ds%i.png'%(i+1), dpi=100, bbox_inches='tight')
    #plt.show()
    plt.close()

qudyn.make_gif(path_to_save, name = 'decayfig.gif', image_name = 'ds*.png')    
#%% RABI TIME SWEEP PLOTTING IN THE BLOCH SPHERE - GIF

Δ = 0.0e6
wq = 5.00e9
wd = wq + Δ
Ω = 40e6
ɣ1 = 1/(10e-6) # Qubit decay rate
ɣ2 = 1/1.0e-6 # Qubit dephase rate
Ψ0 = np.array([[1,0]], dtype = complex)/np.sqrt(1)
ΨΨ = Ψ0*Ψ0.transpose()
Δt = np.linspace(0, 1e-6, 400)

rho00_1 = np.zeros((len(Δt)), dtype=complex).tolist()
rho01_1 = np.zeros((len(Δt)), dtype=complex).tolist()
rho00 = np.zeros((len(Δt)), dtype=complex)
rho01 = np.zeros((len(Δt)), dtype=complex)

for j in range(len(Δt)):
    t_pulse = np.linspace(0, Δt[j], 200)
    rho00_1[j], rho01_1[j], _ = qudyn.rabi_decay(t_pulse, wd, wq, ɣ1, ɣ2, Ω, ΨΨ = ΨΨ)
    rho00[j] = rho00_1[j][-1]
    rho01[j] = rho01_1[j][-1]

envelope = 0.5 + 0.5*np.exp(-(0.5*ɣ1 + 0.5*ɣ2)*Δt)

path_to_save = pth.folder()
path_to_save = (path_to_save + 
    '/RabiTimeGif' + 
    '_detuning%.2fMHZ_coupling%.2fMHz_decay%.2fMHz_dephase%.2f/'%(
    Δ*1e-6, Ω*1e-6, ɣ1*1e-6, ɣ2*1e-6))

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
    
for i in range(len(rho00)-1):
    
    fig = plt.figure(constrained_layout = True, figsize=(14,5), dpi = 100)
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(Δt*1e6, abs(1-rho00), color = 'royalblue')
    ax1.scatter(Δt[i]*1e6,abs(1-rho00[i]), color = 'royalblue', s=30)
    ax1.plot(Δt*1e6, envelope, color = 'grey', ls = '--')
    ax1.tick_params(direction = 'in')
    ax1.set_xlabel('$Pulse$ $time$ $(\mu s)$', fontsize=18.0)
    ax1.set_ylabel(r'$\mathcal{P}$ ' + r'$_{\| 1 \rangle}$', fontsize=20.0)
    ax1.set_xlim((0, Δt.max()*1e6))
    
    ax2 = fig.add_subplot(1,2,2, projection = '3d')
    bloch = qt.Bloch(fig = fig, axes = ax2)
    bloch.view = [-60, 30]
    bloch.make_sphere()
    bloch.frame_color = 'black'
    bloch.sphere_color = 'white'
    bloch.zlpos = [1.2,-1.4]
    left, bottom, width, height = [0.98, 0.05, 0.05, 0.9]
    bloch.point_color = ['tomato','royalblue']
    bloch.point_marker = ['o', 'o' ]
    
    bloch.add_points(qudyn.densitytobsphere(np.array(rho00_1[i]),
                    np.array(rho01_1[i])), alpha = 0.5)
    bloch.add_points(qudyn.densitytobsphere(rho00[i],rho01[i]))
    
    bloch.render()
    ax2.set_box_aspect([1,1,1])
    text = ('Rabi\n' + '$\Delta = %.2f$ $MHz$\n'%(Δ*1e-6) +
            '$\Omega = %.2f$ $MHz$\n'%(Ω*1e-6) +
            '$\gamma_1 = %.2f$ $MHz$\n'%(ɣ1*1e-6) +
            '$\gamma_2 = %.2f$ $MHz$'%(ɣ2*1e-6))
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', label='$fixed$ $amplitude$ $pulse$', markersize=10),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='$measured$ $state$', markersize=10),
                  Line2D([0], [0], linestyle='--', color='gray', label='$Envelope$')]

    ax2.legend(handles=legend_elements, bbox_to_anchor=(0.150, 0.1))
    
    plt.figtext(0.525, 0.8, text, fontsize = 13, bbox = dict(facecolor='white', edgecolor='black', boxstyle='round', alpha = 0.15))
    if i+1<10:plt.savefig(path_to_save + '/rbs00%i.png'%(i+1), dpi=100, bbox_inches='tight')
    elif i+1<100 and i+1>=10:plt.savefig(path_to_save + '/rbs0%i.png'%(i+1), dpi=100, bbox_inches='tight')
    else: plt.savefig(path_to_save + '/rbs%i.png'%(i+1), dpi=100, bbox_inches='tight')

    plt.close()

qudyn.make_gif(path_to_save, name = 'rabitimefig.gif', image_name = 'rbs*.png')

#%% RABI AMPLITUDE SWEEP PLOTTING IN THE BLOCH SPHERE - GIF

Δ = 0.0e6
wq = 5.00e9
wd = wq + Δ
Ω = np.linspace(0, 40e6, 100)
ɣ1 = 1/(10e-6) # Qubit decay rate
ɣ2 = 1/1.0e-6 # Qubit dephase rate
Ψ0 = np.array([[1,0]], dtype = complex)/np.sqrt(1)
ΨΨ = Ψ0*Ψ0.transpose()
Δt = 0.1e-6

rho00_1 = np.zeros((len(Ω)), dtype=complex).tolist()
rho01_1 = np.zeros((len(Ω)), dtype=complex).tolist()
rho00 = np.zeros((len(Ω)), dtype=complex)
rho01 = np.zeros((len(Ω)), dtype=complex)

for i in range(len(Ω)):
    t_pulse = np.linspace(0, Δt, 200)
    rho00_aux, rho01_aux, _ = qudyn.rabi_decay(t_pulse, wd, wq, ɣ1, ɣ2, Ω[i], ΨΨ = ΨΨ)
    rho00[i] = rho00_aux[-1]
    rho01[i] = rho01_aux[-1]

#envelope = 0.5 + 0.5*np.exp(-(0.5*ɣ1 + 0.5*ɣ2)*Δt)

path_to_save = pth.folder()
path_to_save = (path_to_save + 
    '/RabiAmpGif' + 
    '_detuning%.2fMHZ_coupling%.2fMHz_decay%.2fMHz_dephase%.2f/'%(
    Δ*1e-6, Δt*1e-6, ɣ1*1e-6, ɣ2*1e-6))

if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)
    
for i in range(len(rho00)-1):
    
    fig = plt.figure(constrained_layout = True, figsize=(14,5), dpi = 100)
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(Ω*1e-6, abs(1-rho00), color = 'royalblue')
    ax1.scatter(Ω[i]*1e-6,abs(1-rho00[i]), color = 'royalblue', s=30)
    #ax1.plot(Ω*1e-6, envelope, color = 'grey', ls = '--')
    ax1.tick_params(direction = 'in')
    ax1.set_xlabel('$Pulse$ $amplitude$ $(MHz)$', fontsize=18.0)
    ax1.set_ylabel(r'$\mathcal{P}$ ' + r'$_{\| 1 \rangle}$', fontsize=20.0)
    ax1.set_xlim((0, Ω.max()*1e-6))
    
    ax2 = fig.add_subplot(1,2,2, projection = '3d')
    bloch = qt.Bloch(fig = fig, axes = ax2)
    bloch.view = [-60, 30]
    bloch.make_sphere()
    bloch.frame_color = 'black'
    bloch.sphere_color = 'white'
    bloch.zlpos = [1.2,-1.4]
    left, bottom, width, height = [0.98, 0.05, 0.05, 0.9]
    bloch.point_color = ['tomato','royalblue']
    bloch.point_marker = ['o', 'o' ]
    
    bloch.add_points(qudyn.densitytobsphere(np.array(rho00_1[i]),
                    np.array(rho01_1[i])), alpha = 0.5)
    bloch.add_points(qudyn.densitytobsphere(rho00[i],rho01[i]))
    
    bloch.render()
    ax2.set_box_aspect([1,1,1])
    text = ('Rabi\n' + '$\Delta = %.2f$ $MHz$\n'%(Δ*1e-6) +
            '$\Delta t = %.2f$ $\mu s$\n'%(Δt*1e6) +
            '$\gamma_1 = %.2f$ $MHz$\n'%(ɣ1*1e-6) +
            '$\gamma_2 = %.2f$ $MHz$'%(ɣ2*1e-6))
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', label='$fixed$ $amplitude$ $pulse$', markersize=10),
                  Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', label='$measured$ $state$', markersize=10),
                  Line2D([0], [0], linestyle='--', color='gray', label='$Envelope$')]

    ax2.legend(handles=legend_elements, bbox_to_anchor=(0.150, 0.1))
    
    plt.figtext(0.525, 0.8, text, fontsize = 13, bbox = dict(facecolor='white', edgecolor='black', boxstyle='round', alpha = 0.15))
    if i+1<10:plt.savefig(path_to_save + '/rbas00%i.png'%(i+1), dpi=100, bbox_inches='tight')
    elif i+1<100 and i+1>=10:plt.savefig(path_to_save + '/rbas0%i.png'%(i+1), dpi=100, bbox_inches='tight')
    else: plt.savefig(path_to_save + '/rbas%i.png'%(i+1), dpi=100, bbox_inches='tight')

    plt.close()

qudyn.make_gif(path_to_save, name = 'rabiampfig.gif', image_name = 'rbas*.png')
