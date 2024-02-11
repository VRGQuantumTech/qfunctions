# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 05:50:16 2022

Transmission as a function of frequency (trace) fit (FitTracelib).
This library comprises functions to fit transmission data.
       
# v1.0 - Documented some functions.
         Reordered functions to improve clarity.
         
@author: Victor Rollano
"""
import csv
import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt
import math as mt
import pandas as pd
import json
from loadpath import pth
from scipy.signal import savgol_filter
from util import plot as uplt


plt.close('all')


#%% TRACE CLASS

class Trace():
    
    """
    Class with trace object and trace methods.

    """
    
    def __init__(self, f = None, t = None, power = None,
                 HALF_position = None, LER_position = None, units = None,
                 field = None, trace_type = None, model = None):
        
        
        """
        Trace object with transmission data and other information. Initializes
        at None. Load method or fake_trace method are required to proper initia
        lization.
        
        Parameters
        -------
        f: frequency array. Numpy array.
        t: transmission array. Numpy array.
        power: power value of the input signal. Float.
        HALF_position: FHWM position in frequency array. Int.
        LER_position: resonance frequency position in frequency array. Int.
        units: units of the trace array. String.
        field: in case the trace has been acquired with an external magnetic 
               field applied. Float.
        model: specifies the type of model to use in fit.
        
        Returns
        -------
        
        self: trace object.
        """
    
        self.f = f
        self.t = t
        self.power = power
        self.HALF_position = HALF_position
        self.LER_position = LER_position
        self.units = units
        self.field = field
        self.trace_type = trace_type
        self.model = model

        
    def fake_trace(self, params, noise_amp, model = 'parallel', flim = None):
        
        self.model = model
        
        if flim == None: [params[7]*0.5, params[7]*1.5]
        
        self.f = np.linspace(flim[0], flim[1], 10000)
        
        noise_i = np.random.normal(0, noise_amp, size=len(self.f))
        noise_r = np.random.normal(0, noise_amp, size=len(self.f))
        
        if model == 'parallel': self.t = lorentzian(self.f, params)
        elif model == 'inline': self.t = in_lorentzian(self.f, params)
        self.t = (np.real(self.t) + noise_r + 1j*(np.imag(self.t) + noise_i))
        
        return self
    
    def json_load_trace(self):
        path, file, dirpath = pth.file()

        with open(path, 'r') as f:
            data = json.load(f)
        
        if 'powers' in data.keys():
            self.f = np.linspace(data['freq_start'], data['freq_stop'], data['num_points'])
            self.t = np.array(data['realpart']) + 1j*np.array(data['imagpart'])
            self.power = np.array(data ['powers'])
        
        else:
            self.f = np.linspace(data['freq_start'], data['freq_stop'], data['num_points'])
            self.t = np.array(data['realpart']) + 1j*np.array(data['imagpart'])
            self.p = None
        
        if len(self.t.shape) == 2:
            t_aux = np.zeros(self.t.shape[0], dtype = complex)
            for i in range(self.t.shape[0]):
                t_aux[i] = self.t[i][0]
            self.t = t_aux
        
        print(self.f.shape)
        if len(self.f.shape) == 2:
            f_aux = np.zeros(self.f.shape[0])
            for i in range(self.f.shape[0]):
                f_aux[i] = self.f[i][0]
            self.f = f_aux
                
        return self
    
    def json_sonnet_load_trace_mod(self):
        
        path, file, dirpath = pth.file()
            
        # reading csv file
        with open(path, 'r') as csvfile:
            # creating a csv reader object
            data = csv.reader(csvfile)
            next(data)
            
            rows = []
            # extracting each data row one by one
            for row in data:
                rows.append(row)
            
            freq = []
            amp = []
            phase = []
            
            i = 0
            for element in rows[2:]:
                print(i)
                if element == []:
                    break
                freq.append(float(element[0]))
                amp.append(float(element[5]))
                phase.append(float(element[6]))
                i+=1
        
        freq = np.array(freq)
        amp = np.array(amp)
        phase = np.array(phase)

        self.f = freq*1e9
        self.t = amp * np.exp(1j*phase*np.pi/180)
              
        return self
    
    def cab_load_trace(self):
        
        path, file, dirpath = pth.file()
        
        df = pd.read_csv(path, delimiter='\t')
            
        self.f = np.array(df[df.columns[0]])
        self.t = np.array(df[df.columns[1]] + 1j*df[df.columns[2]])

        self.p = float(file.split('LER1_')[1].split('dBm')[0])
        
        return self
    
    def yebes_load_trace(self):
        
        path, file, dirpath = pth.file()
        
        np.loadtxt(path, dtype=str, comments = '!', delimiter = '\t')
        
        aux_str = np.loadtxt(path, dtype=str, comments = '!', delimiter = '\t')
        
        t = []
        
        f0 = float(aux_str[0].split(' ')[6])
        f1 = float(aux_str[-1].split(' ')[6])
        
        self.f = np.linspace(f0, f1, len(aux_str))*1e9
        
        min_char = 11
        for line in aux_str:           
            check = False
            i = 0
            while not check:
                try:
                    t.append(float(line.split(' ')[min_char + i]))
                    check = True
                except:
                    i+=1
        
        self.t = 10**(np.array(t)/20)
        self.p = None
        
        return self
    
    def sonnet_load_trace(self):
        
        path, file, dirpath = pth.file()
            
        # reading csv file
        with open(path, 'r') as csvfile:
            # creating a csv reader object
            data = csv.reader(csvfile)
            next(data)
            
            rows = []
            # extracting each data row one by one
            for row in data:
                rows.append(row)

            freq = np.zeros(len(rows[6:]))
            amp = np.zeros(len(rows[6:]))
            phase = np.zeros(len(rows[6:]))
            i = 0
            for element in rows[6:]:
                freq[i] = element[0]
                amp[i] = element[5]
                phase[i] = element[6]
                i+=1
        
        self.f = freq*1e9
        self.t = amp * np.exp(1j*phase*np.pi/180)
        
        return self
    
    def load_magnetic_trace(self, name = None):
        
        path, files, dirpath = pth.files()
        
        self.field = np.zeros(len(path))
        
        for i in range(len(files)): # Loop for get each name
            self.field[i] = float(files[i].split('_')[-1])
        
        with open(path[0], 'r') as f:
            data = json.load(f)
            self.f = np.linspace(data['freq_start'], data['freq_stop'], data['num_points'])
            self.t = np.array(data['realpart']) + 1j*np.array(data['imagpart'])
            self.t = np.array([self.t])
            
        for i in range(1,len(path)):
            with open(path[i], 'r') as f:
                data = json.load(f)
                self.t = np.append(self.t,
                    np.array([np.array(data['realpart']) + 1j*np.array(data['imagpart'])]), 
                    axis = 0)
                
        self.field, self.t = _order_mag_array(self.field, self.t)
     
        return self


    def do_fit(self, x0, unwrap = False, inv = False, residual = False, units = None):
        
        if np.iscomplexobj(self.t):
            self.do_fit_complex(x0, unwrap = unwrap, inv = False)
        else:
            self.do_fit_amp(x0, inv = inv, residual = residual, units = units)
            
        return
    
    def do_fit_amp(self, x0, inv = False, residual = False, units = None):
        
        x0[0] = _guess_amplitude(self.t)
        x0[7], self.LER_position = _guess_resonance(self.f, self.t)
        x0[5], self.HALF_position = _guess_kappa_coupling(
                                    self.LER_position, self.f, self.t)
        x0[4] = _guess_kappa_internal(self.LER_position, 
                                      self.f, self.t)
        
        self.plot()
        
        results_amp = fit_res_com(
            self.f, self.t, x0, mod='amp', jac='2-point', model = self.model, inv = inv
            )
        
        fit_plot([results_amp], residual = residual)
        
        return
    
    def do_fit_complex(self, x0, unwrap = False, inv = False):
        
        x0[0] = _guess_amplitude(self.t)
        x0[7], self.LER_position = _guess_resonance(self.f, self.t)
        x0[5], self.HALF_position = _guess_kappa_coupling(
                                    self.LER_position, self.f, self.t)
        x0[4] = _guess_kappa_internal(self.LER_position, 
                                      self.f, self.t)
        
        self.plot()
        #x0 = _guess_best_quadrant(x0, ω , t)
                
        results_amp = fit_res_com(
            self.f, self.t, x0, mod='amp', jac='2-point', model = self.model, inv = inv
            )
            
        results_phase = fit_res_com(
            self.f, self.t, results_amp.xf, mod='phase', jac='2-point', 
            model = self.model, unwrap = unwrap
            )
    
        results_real = fit_res_com(self.f, self.t, results_amp.xf,
                                   mod='real', jac='2-point', model = self.model
                                   )
        
        results_imag = fit_res_com(self.f, self.t, results_amp.xf, mod='imag',
                                   jac='2-point', model = self.model
                                   )
            
        fit_plot([results_amp, results_phase])
        
        plot_iq_amph(results_amp, results_phase)
        
        fit_plot([results_real, results_imag])
        
        plot_iq_reim(results_real, results_imag)
    
        return


    def plot(self, units = None, guess_FHWM = False):
        
        if (self.units != None and units != None and units != self.units):
            Warning('Two different vertical units provided. Using function units.')
            
        if guess_FHWM:
            _ , self.LER_position = _guess_resonance(self.f, self.t)
            _ , self.HALF_position = _guess_kappa_coupling(
                                        self.LER_position, self.f, self.t)
        
        t, y_label, half, resonance = uplt.set_power_unit(
                                    units, self.t,
                                    self.HALF_position, self.LER_position)
        
        f, funit, forder = uplt.guess_magnitude_order(self.f, 'Hz')
        
        plt.figure(dpi = 1200)
        plt.plot(
            f, t,'k',
            label = 'measurement', lw=1.5, zorder=1
            )
        if self.HALF_position != None:
            plt.scatter(
                f[self.HALF_position],
                half,
                color = 'red'
                )
        if self.LER_position != None:
            plt.scatter(
                f[self.LER_position],
                resonance,
                color = 'royalblue'
                )
        
        plt.xlabel(f'Frequency {(funit)}' , fontsize = 14)
        plt.ylabel(y_label, fontsize = 14)
        plt.xlim((f.min(), f.max()))
        plt.tick_params(direction = 'in')
        plt.show()
        plt.close()
        
        return


#%% LOADING FUCTIONS

def _order_mag_array(b,t):
    t = t.transpose()
    i = 0

    while i < (len(b) - 1):
        if b[i] > b[i + 1]:
            aux = b[i]
            b[i] = b[i + 1]
            b[i + 1] = aux
            aux = t[:,i]
            t[:,i] = t[:,i + 1]
            t[:,i + 1] = aux
            i = 0
        else: i+=1
    
    return b,t

#%% FIT SUPPORT FUNCTIONS

def _find_minimum(ω, t, a):
    # Finding LER resonance position and RESONANCE FREQUENCY: ##################
    LER_position = 0
    stop = False
    t_amp = abs(t - a)

    while stop == False:
        if t_amp[LER_position] == t_amp.max():
            stop = True
        else:
            LER_position = LER_position + 1
    
    return ω[LER_position], LER_position

def _find_maximum(ω, t):
    # Finding LER maximum position and RESONANCE FREQUENCY: ##################
    LER_position = 0
    stop = False
    t_amp = abs(t)**2
    while stop == False:
        if t_amp[LER_position] == t_amp.max():
            stop = True
        else:
            LER_position = LER_position + 1
    
    return ω[LER_position], LER_position

def _set_fixed(x0, mod = 'amp'):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    if mod == 'amp':
        fixed = np.zeros(2)
        fixed[0] = x0[1] #dt
        fixed[1] = x0[2] #φ
        x1 = np.zeros(6)
        x1[0] = x0[0] #a
        x1[1] = x0[3] #ar
        x1[2] = x0[4] #κi
        x1[3] = x0[5] #κc
        x1[4] = x0[6] #Δφ
        x1[5] = x0[7] #ωr
        
    elif mod == 'phase':
        fixed = np.zeros(2)
        fixed[0] = x0[0] #a
        fixed[1] = x0[7] #ωr

        x1 = np.zeros(6)
        x1[0] = x0[1] #dt
        x1[1] = x0[2] #φ
        x1[2] = x0[3] #ar
        x1[3] = x0[4] #κi
        x1[4] = x0[5] #κc
        x1[5] = x0[6] #Δφ
        
    else:
        fixed = np.zeros(6)
        fixed[0] = x0[0] #a
        fixed[1] = x0[3] #ar
        fixed[2] = x0[4] #κi
        fixed[3] = x0[5] #κc
        fixed[4] = x0[6] #Δφ
        fixed[5] = x0[7] #ωr
        
        x1 = np.zeros(2)
        x1[0] = x0[1] #dt
        x1[1] = x0[2] #φ
    
    return x1,fixed

def _set_output(x1,fixed, mod ='amp'):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    if mod == 'amp':
        xf = np.zeros(8)
        xf[0] = x1[0] #a
        xf[1] = fixed[0] #dt
        xf[2] = fixed[1]%(2*np.pi) #φ
        xf[3] = x1[1] #ar
        xf[4] = x1[2] #κi
        xf[5] = x1[3] #κc
        xf[6] = x1[4]%(2*np.pi) #Δφ
        xf[7] = x1[5] #ωr
    elif mod == 'phase':
        xf = np.zeros(8)
        xf[0] = fixed[0] #a
        xf[1] = x1[0] #dt
        xf[2] = x1[1]%(2*np.pi) #φ
        xf[3] = x1[2] #ar
        xf[4] = x1[3] #κi
        xf[5] = x1[4] #κc
        xf[6] = x1[5]%(2*np.pi) #Δφ
        xf[7] = fixed[1] #ωr
    else:
        xf = np.zeros(8)
        xf[0] = fixed[0] #a
        xf[1] = x1[0] #dt
        xf[2] = x1[1]%(2*np.pi) #φ
        xf[3] = fixed[1] #ar
        xf[4] = fixed[2] #κi
        xf[5] = fixed[3] #κc
        xf[6] = fixed[4]%(2*np.pi) #Δφ
        xf[7] = fixed[5] #ωr
    
    return xf
    
def _set_bounds(ω,x0,mod='amp'):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    ####### BOUNDS FOR FITTING PARAMETERS ##################
    a_upper_bound = 1.1
    a_lower_bound = 0.0
    
    dt_upper_bound = 1e-4
    dt_lower_bound = -1e-4
    
    φ_upper_bound = 2*mt.pi
    φ_lower_bound = 0
    
    ar_upper_bound = 1.00001
    ar_lower_bound = 0.99999
            
    κi_upper_bound = x0[4] + 100*x0[4]
    κi_lower_bound = x0[4] - x0[4]
    
    κc_upper_bound = x0[5] + 100*x0[5]
    κc_lower_bound = x0[5] - x0[5]
    
    Δφ_upper_bound = 2*mt.pi
    Δφ_lower_bound = -2*mt.pi
    
    ωr_upper_bound = ω.max()
    ωr_lower_bound = ω.min()
                
   
    if mod=='amp':
        #a, ar, κ, κc, Δφ, ωr
        upper_bound = np.array([a_upper_bound,ar_upper_bound,κi_upper_bound,
                                κc_upper_bound,Δφ_upper_bound,ωr_upper_bound])
        lower_bound = np.array([a_lower_bound,ar_lower_bound,κi_lower_bound,
                                κc_lower_bound,Δφ_lower_bound,ωr_lower_bound])
    elif mod=='phase':
        #a, ar, κ, κc, Δφ, ωr
        upper_bound = np.array([dt_upper_bound,φ_upper_bound,ar_upper_bound,
                                κi_upper_bound,κc_upper_bound,Δφ_upper_bound])
        lower_bound = np.array([dt_lower_bound,φ_lower_bound,ar_lower_bound,
                                κi_lower_bound,κc_lower_bound,Δφ_lower_bound])
    else:
        #dt, φ, κ, κc
        upper_bound = np.array([dt_upper_bound,φ_upper_bound])
        lower_bound = np.array([dt_lower_bound,φ_lower_bound])
    
    bounds = (lower_bound , upper_bound)

    return bounds

def _set_min_bounds(ω,x0,mod='amp'):
    #a, dt, κ, κc, Δφ, ωr
    if mod=='amp':
        #a, ar, κ, κc, Δφ, ωr
        bounds =((0, 1.0), (0,10), (0, None), (0, None), (0, 2*mt.pi), (ω.min(), ω.max()))
    elif mod=='phase':
        bounds =((0, 1.0e-6), (0, 2*mt.pi), (0, 10), (0, None), (0, None), (0, 2*mt.pi))
    else:
        #dt, κ, κc
        bounds =((0,None), (0, None))
        
    return bounds

def rescale(ω, t):
    ts = t/t.max()
    ωs = ω - ω.min()
    ωs = 2*ωs/ωs.max()
    
    return ωs, ts

def _guess_amplitude(t):
    return abs(t[0])
    
def _guess_phase_delay(ω, t, unwrap = False):
    if unwrap: ph = np.unwrap(np.arctan2(np.imag(t),np.real(t)))
    else: ph = np.arctan2(np.imag(t),np.real(t))
    return -1*((ph[-1] - ph[0])/(ω[-1] - ω[0]))

def _guess_phase_zero(ω, t, unwrap = False):
    dt = _guess_phase_delay(ω, t, unwrap = unwrap)
    ph = np.unwrap(np.arctan2(np.imag(t),np.real(t)))
    th = ph[0] + dt*ω[0]
    return th

def _guess_relative_amplitude():
    return 1.

def _guess_best_quadrant(x0, ω , t):
    
    x1 = x0.copy()
    q = [i*np.pi/4 for i in range(8)]
    r = []

    for i in q:
        x1[6]=i
        results = fit_res_com(ω, t, x1, mod='amp', jac='3-point', plot=False)
        r.append(sum(results.res))
     
    i = np.argmin(np.abs(r))

    x0[6] = q[i]
    
    print('Best phase initial condition is Δφ = %iπ/4'%i)
    
    return x0

def _guess_kappa_coupling(LER_position, ω, t):
    
    t_amp = abs(t)
    t_amp = savgol_filter(t_amp, 51, 3)
    
    t_half = np.sqrt((t_amp.max()**2 + t_amp.min()**2)/2)
        
    HALF_position = 0
    stop = False
    
    while stop == False:
        if t_amp[HALF_position] <= t_half:
            stop = True
        else:
            HALF_position = HALF_position + 1

    kappac_aprox = 2*(ω[LER_position] - ω[HALF_position])
    
    return kappac_aprox, HALF_position

def _guess_kappa_internal(LER_position, ω, t):
    kappa_c , _ = _guess_kappa_coupling(LER_position, ω, t)
    
    t_amp = abs(t)    
    t_min = t_amp[LER_position]
    
    kappai_aprox = 2 * kappa_c / t_min
    
    return kappai_aprox

def _guess_phano_phase(LER_position, ω, t):
    
    a = _guess_amplitude(t)
    
    t_amp = abs(t)    
    t_min = t_amp[LER_position]
    
    kappa_c, _ = _guess_kappa_coupling(LER_position, ω, t)
    kappa_i = _guess_kappa_internal(LER_position, ω, t)
    
    A = kappa_c / (kappa_i + kappa_c)
    
    phano =  (t_min - a**2 * (1 + A**2)) / (2 * a**2 * A)
    
    return np.arccos(phano)
    

def _guess_resonance(ω, t):
    # Finding LER resonance position and RESONANCE FREQUENCY: ##################
    
    LER_position = 0
    stop = False
    t_amp = abs(t)

    while stop == False:
        if t_amp[LER_position] == t_amp.min():
            stop = True
        else:
            LER_position = LER_position + 1
    
    return ω[LER_position], LER_position
            
def _calculate_residual(t,tprx):
    
    return t-tprx


#%% RESULT CLASS TO STORE DE RESULT OF THE FIT

class Results():
    def __init__(self, ω, t, tprx, xf, res, jac, fig, mod):
        self.ω = ω
        self.t = t
        self.tprx = tprx
        self.xf = xf
        self.res = res
        self.jac = jac
        self.fig = fig
        self.mod = mod 
        
#%% FIT FUNCTION

def fit_res_com(ω, t, x0, mod='amp', jac='3-point', verbose=0, gtol = 1e-15,
                xtol=1e-15, plot=True, model = 'parallel', unwrap = False,
                inv = False):
    
    """
    Function to fit transmission vs frequency data using scipy least squares
    method.
    
    Parameters
    ----------
    
    ω: numpy or float array. frequency data. in Hz.
    t: complex array. transmission data. linear. no units.
    mod: string. fit mode.
            possible values - 'amp', 'phase', 'real', 'imag'
    jac: string. least squares method.
            possible values - '2-point', '3-point', 'cs'
    verbose: int. level of information passed by the least squares function
        during the fit process.
    gtol: float. tolerance value for least squares function
    xtol: float. tolerance value for least squares function
    
    plot: boolean. if True, the function plots the results.
    model: string. selects resonator model.
            possible values - 'inline' for waveguide resonators
                            - 'parallel' for res coupled in parallel to TL
            WARNING! Only parallel is fully implemented in the model
    unwrap: boolean. if True, numpy unwrap function is applied to the phase.
            ADVICE: usually false for sonnet data and true for measurements.
                  
    Returns
    -------
    Results: Results class (in this library file)
        
    """
    
    #a, dt, φ, ar, κi, κc, Δφ, ωr
    
    x1,fixed = _set_fixed(x0, mod = mod)
    constat_arguments = {'fixed': fixed}

    if model == 'parallel':
        if mod == 'amp':
            fun_to_minimize = least_lorentzian_amp
            t = abs(t)
            x1[0] = _guess_amplitude(t)
            
            if inv:
                fun_to_minimize = least_invlorentzian_amp
                t = abs(1/t)
                
        elif mod == 'phase':
            fun_to_minimize = least_lorentzian_phase
            if unwrap: t = np.unwrap(np.arctan2(np.imag(t),np.real(t)))
            else: t = np.arctan2(np.imag(t),np.real(t))
        elif mod == 'real':
            fun_to_minimize = least_lorentzian_real        
            t = np.real(t)
        elif mod == 'imag':
            fun_to_minimize = least_lorentzian_imag
            t = np.imag(t)
        else:
            raise Exception('Fitting mode not recognized')
            return None
        
    elif model == 'inline':
        if mod == 'amp':
            fun_to_minimize = least_in_lorentzian_amp
            t = abs(t)
            x1[0] = _guess_amplitude(t)
        else:
            raise Exception('Fitting mode not recognized')
            return None
    else:
        raise Exception('Model not recognized')
        return None
    

    ##### FIND TRANSMISSION MINIMUM IN TRACE ###############
    if model == 'parallel':
        if mod == 'amp': x1[5], LER_position = _find_minimum(ω, t, x1[0])
        else: _, LER_position = _find_minimum(ω, t, x1[0])
    elif model == 'inline':
        if mod == 'amp': x1[5], LER_position = _find_maximum(ω, t, x1[0])
    
    ##### STABLISH FITTING BOUNDS ###############
    bounds = _set_bounds(ω,x0, mod=mod)
    print(bounds)
    print(x1)
    if mod == 'amp':
        result = sop.least_squares(fun_to_minimize, x1, jac=jac, 
                                   args = [ω, t], kwargs = constat_arguments,
                                   bounds = bounds, verbose = verbose, gtol=gtol,
                                   xtol=xtol)
    else:
        result = sop.least_squares(fun_to_minimize, x1, jac=jac, 
                                   args = [ω, t], kwargs = constat_arguments,
                                   bounds = bounds, verbose = verbose, gtol=gtol,
                                   xtol=xtol, ftol = 1e-15)
   
    xf = _set_output(result.x,fixed, mod = mod)
    res = result.fun
    jac = result.jac
    
        
    if mod == 'amp': tprx = abs(lorentzian(ω, xf))
    if mod == 'phase': tprx = np.unwrap(
            np.arctan2(np.imag(lorentzian(ω, xf)),np.real(lorentzian(ω, xf))))
    if mod == 'real': tprx = np.real(lorentzian(ω, xf))
    if mod == 'imag': tprx = np.imag(lorentzian(ω, xf))

    if plot == True: fig = plot_results(ω, t, tprx, xf, LER_position, mod)
    else: fig = None
    
    res = _calculate_residual(t,tprx)
    
    
    return Results(ω, t, tprx, xf, res, jac, fig, mod)


#%% DEPRECATED
# def min_res_com(ω, t, x0, mod='amp', method='Nelder-Mead', plot=True, model = 'parallel'):
#     #a, dt, ar, κi, κc, φ, ωr
    
#     x1,fixed = _set_fixed(x0, mod = mod)

#     if model == 'parallel':
#         if mod == 'amp':
#             fun_to_minimize = min_lorentzian_amp
#             t = abs(t)
#         elif mod == 'phase':
#             fun_to_minimize = min_lorentzian_phase
#             t = np.unwrap(np.arctan2(np.imag(t),np.real(t)))
#         elif mod == 'real':
#             fun_to_minimize = min_lorentzian_real        
#             t = np.real(t)
#         elif mod == 'imag':
#             fun_to_minimize = min_lorentzian_imag
#             t = np.imag(t)
#         else:
#             raise Exception('Fitting mode not recognized')
#             return None
#     elif model == 'inline':
#         if mod == 'amp':
#             fun_to_minimize = min_in_lorentzian_amp
#             t = abs(t)
#         elif mod == 'real':
#             fun_to_minimize = min_in_lorentzian_real
#             t = np.real(t) 
#         elif mod == 'imag':
#             fun_to_minimize = min_in_lorentzian_imag
#             t = np.imag(t)
#         else:
#             raise Exception('Fitting mode not recognized')
#             return None
#     else:
#         raise Exception('Model not recognized')
#         return None
        
    
#     ##### FIND TRANSMISSION MINIMUM IN TRACE ###############
#     if model == 'parallel':
#         if mod == 'amp': x1[5], LER_position = _find_minimum(ω, t, x1[0])
#         else: _, LER_position = _find_minimum(ω, t, x1[0])
#     elif model == 'inline':
#         if mod == 'amp': x1[5], LER_position = _find_maximum(ω, t, x1[0])
#         else: _, LER_position = _find_maximum(ω, t, x1[0])
    
#     ##### STABLISH FITTING BOUNDS ###############
#     bounds = _set_min_bounds(ω,x0, mod=mod)
    
#     if mod == 'amp':
#         result = sop.minimize(fun_to_minimize, x1, args = (ω, t, fixed),
#                             method=method, bounds = bounds)
       
#     else:
#         result = sop.minimize(fun_to_minimize, x1, args = (ω, t, fixed),
#                             method=method, bounds = bounds)
    
    
#     xf = _set_output(result.x,fixed, mod = mod)
#     res = result.fun
#     jac = None
    
#     if mod == 'amp': tprx = abs(lorentzian(ω, xf))
#     if mod == 'phase': tprx = np.unwrap(
#             np.arctan2(np.imag(lorentzian(ω, xf)),np.real(lorentzian(ω, xf))))
#     if mod == 'real': tprx = np.real(lorentzian(ω, xf))
#     if mod == 'imag': tprx = np.imag(lorentzian(ω, xf))

#     if plot == True: fig = plot_results(ω, t, tprx, xf, LER_position, mod)
#     else: fig = None
    
#     res = _calculate_residual(t,tprx)
    
#     return Results(ω, t, tprx, xf, res, jac, fig, mod)

#%% FUNCTIONS TO PLOT AND PRESENT THE RESULTS

def plot_results(ω, t, tprx, xf, LER_position, mod, plot = False):
    
    print('MINIMIZATION METHOD RESULTS (' + mod + '):')
    print('a = ' + str(format(xf[0], "E")))
    print('dt = ' + str(format(xf[1], "E")) + ' s')
    turns = mt.floor(xf[2]/(2*mt.pi))
    print('φ = ' + str(round(xf[2]%(2*np.pi), 4)) + ' rad')
    print('ar = ' + str(format(xf[3], "E")) + ' Hz')
    print('κi = ' + str(format(xf[4], "E")) + ' Hz')
    print('κc = ' + str(format(xf[5], "E")) + ' Hz')
    turns = mt.floor(xf[6]/(2*mt.pi))
    print('Δφ = ' + str(round(xf[6] - turns*2*mt.pi, 4)) + ' rad')
    print('ωr = ' + str(format(xf[7], "E")) + ' Hz')
    
    if plot:
        order = mt.floor(mt.log10(np.abs(ω.max())))
        
        fig = plt.figure('Result least squares')
        
        if order >= 9:
            plt.plot(ω*1e-9,t, 'k', label = 'measurement', lw=3.5, zorder=2)
            plt.scatter(ω[LER_position-1]*1e-9,t[LER_position], facecolors='white', edgecolors='k', zorder=1, s=100,label = 'S21 max')
            plt.plot(ω*1e-9,tprx, 'r--', label = 'fit', lw=1.5, zorder=3)
            plt.xlabel('Frequency (GHz)' , fontsize = 14, fontweight='bold')
            plt.xlim([ω.min()*1e-9, ω.max()*1e-9])
        
        if order >= 6 and order < 9:
            plt.plot(ω*1e-6,t, 'k', label = 'measurement', lw=2.5, zorder=2)
            plt.scatter(ω[LER_position]*1e-6,t[LER_position], facecolors='gray', edgecolors='k', zorder=1, label = 'S21 max')
            plt.plot(ω*1e-6,tprx, 'r--', label = 'fit', lw=1.5, zorder=3)
            plt.xlabel('Frequency (MHz)' , fontsize = 14, fontweight='bold')
            plt.xlim([ω.min()*1e-6, ω.max()*1e-6])
        
        if order >= 3 and order < 6:
            plt.plot(ω*1e-3,t, 'k', label = 'measurement', lw=2.5, zorder=2)
            plt.scatter(ω[LER_position]*1e-3,t[LER_position], facecolors='gray', edgecolors='k', zorder=1, label = 'S21 max')
            plt.plot(ω*1e-3,tprx, 'r--', label = 'fit', lw=1.5, zorder=3)
            plt.xlabel('Frequency (kHz)' , fontsize = 14, fontweight='bold')
            plt.xlim([ω.min()*1e-3, ω.max()*1e-3])
        
        if order >= 0 and order < 3:
            plt.plot(ω,t, 'k', label = 'measurement', lw=2.5, zorder=2)
            plt.scatter(ω[LER_position],t[LER_position], facecolors='gray', edgecolors='k', zorder=1, label = 'S21 max')
            plt.plot(ω,tprx, 'r--', label = 'fit', lw=1.5, zorder=3)
            plt.xlabel('Frequency (Hz)' , fontsize = 14, fontweight='bold')
            plt.xlim([ω.min(), ω.max()])
        
        
        plt.ylabel('Transmission' , fontsize = 14, fontweight='bold')
        
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        
        if mod == 'amp':
            plt.title('Amplitude')
            mod = 'Amplitude'
            
        if mod == 'real':
            plt.title('Real part')
            mod = 'Real Part'
            
        if mod == 'imag':
            plt.title('Imaginary part')
            mod = 'Imaginary Part'
            
        plt.tick_params(direction = 'in')
        plt.legend()
        plt.show()
        
        return fig
    
    return
  

def fit_plot(results = [], residual = False):
    
    ω = results[0].ω
    f = plt.figure(dpi = 2200)
    
    if not residual: dim = (len(results),1)
    else: dim = (len(results),2) 
    
    if len(results) == 1: figsize = (6,4)
    else: figsize = (7,11)
    
    if residual: figsize = (figsize[0] * 2, figsize[1])
    
    fig, ax =  plt.subplots(dim[0], dim[1], sharex=True, figsize=figsize, dpi = 1200)
    
    if len(results) == 1: ax = np.reshape(ax, dim)
    
    for i in range(len(results)):
        for j in range(dim[1]):
            if j == 0:
                ax[i,j].plot(ω*1e-9, results[i].t, 'ko', label = 'measurement', lw=3.5, zorder=2)
                ax[i,j].plot(ω*1e-9, results[i].tprx, 'r--', label = 'fit', lw=2.5, zorder=3)
                ax[i,j].set_ylabel('Transmission', fontsize=14.0, fontweight='bold')
            if j == 1:
                ax[i,j].plot(ω*1e-9, results[i].res, 'k', label = 'residual', lw=3.5, zorder=2)
                ax[i,j].set_ylabel('Residual', fontsize=14.0, fontweight='bold')
            
            ax[i,j].set_xlabel('Frequency (GHz)' , fontsize = 14, fontweight='bold')
            ax[i,j].tick_params(direction = 'in')
            ax[i,j].grid(True)
            ax[i,j].set_xlim([ω.min()*1e-9, ω.max()*1e-9])
            ax[i,j].set_title(results[i].mod)
            
    plt.show()
    plt.close()
    
    return f

def plot_iq_amph(results_amp, results_ph):
    
    t = results_amp.t * np.exp(1j*results_ph.t)
    tprx = results_amp.tprx * np.exp(1j*results_ph.tprx)
    
    f = plt.figure()
    fig, ax =  plt.subplots(1, 2, figsize=(11,11), sharex = True, sharey = True)
    ax[0].plot(np.real(t), np.imag(t), 'ko', label = 'measurement', lw=3.5, zorder = 2)
    ax[0].plot(np.real(tprx), np.imag(tprx), 'r--', label = 'fit', lw=1.5, zorder=3)
    ax[1].plot(np.real(t) - np.real(tprx), np.imag(t) - np.imag(tprx), 
                'ko', label = 'residual', lw=3.5, zorder=2)
    ax[0].set_title('IQ from Amplitude and Phase fits')
    ax[1].set_title('IQ residual from Amplitude and Phase fits')
    for i in range(2):
        ax[i].tick_params(direction = 'in')
        ax[i].grid(True)
        ax[i].set_aspect(1)
        ax[i].set_xlabel('I (a.u.)', fontsize = 14, fontweight='bold')
        ax[i].set_ylabel('Q (a.u.)', fontsize = 14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return f

def plot_iq_reim(results_real, results_imag):
    
    f = plt.figure()
    fig, ax =  plt.subplots(1, 2, figsize=(11,11), sharex = True, sharey = True)
    ax[0].plot(results_real.t, results_imag.t, 'ko', label = 'measurement', lw=3.5, zorder = 2)
    ax[0].plot(results_real.tprx, results_imag.tprx, 'r--', label = 'fit', lw=1.5, zorder=3)
    ax[1].plot(results_real.t - results_real.tprx, results_imag.t - results_imag.tprx, 
                'ko', label = 'residual', lw=3.5, zorder=2)
    ax[0].set_title('IQ fit from Real and Imaginary fits')
    ax[1].set_title('IQ residual from Real and Imaginary fits')
    for i in range(2):
        ax[i].tick_params(direction = 'in')
        ax[i].grid(True)
        ax[i].set_aspect(1)
        ax[i].set_xlabel('I (a.u.)', fontsize = 14, fontweight='bold')
        ax[i].set_ylabel('Q (a.u.)', fontsize = 14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return f
    

#%% MODEL FUNCTIONS PREPARED FOR 2D DATA FIT INCLUDING COUPLING TO SPIN

def M_2f (κi, κc, ωr, γ, G, Ω, ω):

    """
    Function to build the matrix of the resonator-spin coupling model
    
    Parameters
    ----------
    
    κi: float. internal loss rate of the resonator.
    κc: float. loss rate of the resonator due to the coupling to the TL.
    ωr: float. resonance frequency of the resonator.
    γ: float. internal loss rate of the spin.
    G: float. coupling rate between the spin and the resonator.
    ω: float. frequency at which the model is solved.
                  
    Returns
    -------
    Returns the matrix of the model.
    
    WARNING! This function is only used by other functions.
        
    """
    
    ω=ω
    κ = κi + κc
    return np.array([[1j*(ω - ωr) + κ , 1j*G], 
                     [1j*G, 1j*(ω - Ω) + γ]], dtype=complex)


def X1_2f(ac, dt, φ, ar, κi, κc, Δφ, ωr, γ, G, Ω, ω):
    
    """
    Function to build the matrix of the resonator-spin coupling model
    
    Parameters
    ----------
    
    ac: float. attenuation introduced by the setup.
    dt: float. slope of the phase delay.
    φ: float. phase delay at zero frequency.
    ar: float. fano amplitude assymetry. 
    κi: float. internal loss rate of the resonator.
    κc: float. loss rate of the resonator due to the coupling to the TL.
    Δφ: float. fano phase assymetry.
    ωr: float. resonance frequency of the resonator.
    γ: float. internal loss rate of the spin.
    G: float. coupling rate between the spin and the resonator.
    ω: numpy array. frequencies at which the model is solved.
                  
    Returns
    -------
    Returns the complex transmission
    
    NOTE:
        
    """
    
    ω=ω
    Mat= M_2f (κi, κc, ωr, γ, G, Ω, ω)
    inverse = np.linalg.inv(Mat)
    f1=np.array([-1j*np.sqrt(ar*κc*np.exp(1j*Δφ)),0.],  dtype=complex)
    xarray = np.matmul(inverse,f1)
    
    return ac*(np.exp(-1j*(ω*dt - φ)))*(1-1j*np.sqrt(ar*κc*np.exp(1j*Δφ))*xarray[0])

def lorentzian(ω, params):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    
    ac = params[0]
    dt = params[1]
    φ = params[2]
    ar = params[3]
    κi = params[4] 
    κc = params[5]
    Δφ = params[6]
    ωr = params[7]
    
    
    # t = ac*(np.exp(-1j*(ω*dt - φ)))*(1 - ar*κc*np.exp(1j*Δφ)/(1j*(ω - ωr) + κ))
    
    myfunction = np.vectorize(X1_2f)
    
    t = myfunction(ac, dt, φ, ar, κi, κc, Δφ, ωr, 0, 0, 1e9, ω)
    
    return t

#%% MODEL FUNCTIONS
''' These functions are used to model the resonator transmission
    lorentzian is for a resonator coupled in parallel to the TL
    in_lorentzian is for a resonator in the transmission line
    
    'least_' functions are adapted to the python least_square function sintax
    'min_' functions are adapted to the python minimize method sintax
'''

def in_lorentzian(ω, params):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    
    ac = params[0]
    dt = params[1]
    φ = params[2]
    ar = params[3]
    κi = params[4] 
    κc = params[5]
    Δφ = params[6]
    ωr = params[7]
    
    κ = κc + κi
    
    t = ac*(np.exp(-1j*(ω*dt - φ)))*(1 + ar*np.exp(1j*Δφ)*(κc/(1j*(ωr - ω) + κ)))
    
    return t

def least_lorentzian_amp(not_fixed,ω,t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = not_fixed[0] #a
    params[1] = fixed[0] #dt
    params[2] = fixed[1] #φ
    params[3] = not_fixed[1] #ar
    params[4] = not_fixed[2] #κi
    params[5] = not_fixed[3] #κc
    params[6] = not_fixed[4] #Δφ
    params[7] = not_fixed[5] #ωr
    
    return t - abs(lorentzian(ω, params))

def least_invlorentzian_amp(not_fixed,ω,t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = not_fixed[0] #a
    params[1] = fixed[0] #dt
    params[2] = fixed[1] #φ
    params[3] = not_fixed[1] #ar
    params[4] = not_fixed[2] #κi
    params[5] = not_fixed[3] #κc
    params[6] = not_fixed[4] #Δφ
    params[7] = not_fixed[5] #ωr
    
    return t - abs(1/lorentzian(ω, params))
    

def least_lorentzian_phase(not_fixed,ω,t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = not_fixed[2] #ar
    params[4] = not_fixed[3] #κi
    params[5] = not_fixed[4] #κc
    params[6] = not_fixed[5] #Δφ
    params[7] = fixed[1] #ωr
    
    return t - np.unwrap(np.arctan2(np.imag(lorentzian(ω, params)),
                                            np.real(lorentzian(ω, params))))

def least_in_lorentzian_amp(not_fixed,ω,t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = not_fixed[0] #a
    params[1] = fixed[0] #dt
    params[2] = fixed[1] #φ
    params[3] = not_fixed[1] #ar
    params[4] = not_fixed[2] #κi
    params[5] = not_fixed[3] #κc
    params[6] = not_fixed[4] #Δφ
    params[7] = not_fixed[5] #ωr
    
    return t - abs(in_lorentzian(ω, params))

def least_in_lorentzian_phase(not_fixed,ω,t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = not_fixed[0] #a
    params[1] = fixed[0] #dt
    params[2] = fixed[1] #φ
    params[3] = not_fixed[1] #ar
    params[4] = not_fixed[2] #κi
    params[5] = not_fixed[3] #κc
    params[6] = not_fixed[4] #Δφ
    params[7] = not_fixed[5] #ωr
    
    return t - np.arctan2(in_lorentzian(ω, params))

def min_lorentzian_amp(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = not_fixed[0] #a
    params[1] = fixed[0] #dt
    params[2] = fixed[1] #φ
    params[3] = not_fixed[1] #ar
    params[4] = not_fixed[2] #κi
    params[5] = not_fixed[3] #κc
    params[6] = not_fixed[4] #Δφ
    params[7] = not_fixed[5] #ωr
    
    return np.sum(np.abs(t - abs(lorentzian(ω, params)))**2)


def min_lorentzian_phase(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = not_fixed[1] #ar
    params[4] = not_fixed[2] #κi
    params[5] = not_fixed[3] #κc
    params[6] = not_fixed[4] #Δφ
    params[7] = fixed[1] #ωr
    
    return np.sum(np.abs(t - np.unwrap(np.arctan2(np.imag(lorentzian(ω, params)),
                                            np.real(lorentzian(ω, params)))))**2)

def min_in_lorentzian_amp(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = not_fixed[0] #a
    params[1] = fixed[0] #dt
    params[2] = fixed[1] #φ
    params[3] = not_fixed[1] #ar
    params[4] = not_fixed[2] #κi
    params[5] = not_fixed[3] #κc
    params[6] = not_fixed[4] #Δφ
    params[7] = not_fixed[5] #ωr
    
    return np.sum(np.abs(t - abs(lorentzian(ω, params)))**2)

def least_lorentzian_real(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = fixed[1] #ar
    params[4] = fixed[2] #κi
    params[5] = fixed[3] #κc
    params[6] = fixed[4] #Δφ
    params[7] = fixed[5] #ωr
   
    
    return t - np.real(lorentzian(ω, params))

def min_lorentzian_real(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = fixed[1] #ar
    params[4] = fixed[2] #κi
    params[5] = fixed[3] #κc
    params[6] = fixed[4] #Δφ
    params[7] = fixed[5] #ωr
    
    return np.sum(np.abs(t - np.real(lorentzian(ω, params)))**2)

def min_in_lorentzian_real(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = fixed[1] #ar
    params[4] = fixed[2] #κi
    params[5] = fixed[3] #κc
    params[6] = fixed[4] #Δφ
    params[7] = fixed[5] #ωr
    
    return np.sum(np.abs(t - np.real(in_lorentzian(ω, params)))**2)

def least_lorentzian_imag(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = fixed[1] #ar
    params[4] = fixed[2] #κi
    params[5] = fixed[3] #κc
    params[6] = fixed[4] #Δφ
    params[7] = fixed[5] #ωr
   
    
    return t - np.imag(lorentzian(ω, params))

def min_lorentzian_imag(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = fixed[1] #ar
    params[4] = fixed[2] #κi
    params[5] = fixed[3] #κc
    params[6] = fixed[4] #Δφ
    params[7] = fixed[5] #ωr
    
    return np.sum(np.abs(t - np.imag(lorentzian(ω, params)))**2)

def min_in_lorentzian_imag(not_fixed,ω,t,fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)
    
    params[0] = fixed[0] #a
    params[1] = not_fixed[0] #dt
    params[2] = not_fixed[1] #φ
    params[3] = fixed[1] #ar
    params[4] = fixed[2] #κi
    params[5] = fixed[3] #κc
    params[6] = fixed[4] #Δφ
    params[7] = fixed[5] #ωr
    
    return np.sum(np.abs(t - np.imag(in_lorentzian(ω, params)))**2)
