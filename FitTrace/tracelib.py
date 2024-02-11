# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 05:50:16 2022

Transmission as a function of frequency (trace) fit (FitTrace lib).
This library comprises functions to fit transmission data.

# v1.1 - Included do_nonlfit() method in Trace to fit resonator transmission 
             trace in a nonlinear regime due to kinetic inductance.
         Included funtions for non-linear regime fit from kinetic inductance
         Inlcuded function power_sweep_analysis() to fit several traces from
             a power sweep measurement (in linear regime).
         Improved results display method. Function 'plot_results' and
             'plot_nonl_results' changed to 'show_results' and 
             'show_nonl_results'.
         Improved coupling rate prediction in '_guess_kappa_coupling' function.
         Documented Trace class.
         Documented do_nonlfit() method from Trace class.
         
# v1.0 - Documented some functions.
         Reordered functions to improve clarity.
         
@author: Victor Rollano
"""

import csv
import numpy as np
import scipy.optimize as sop
import matplotlib
import matplotlib.pyplot as plt
import math as mt
import pandas as pd
import json
from loadpath import pth
from scipy.signal import savgol_filter
from util import plot as uplt
from util import conv


plt.close('all')


# %% TRACE CLASS

class Trace():

    """
    Class with trace object and trace methods.

    Attributes
    ----------
    f : numpy.ndarray
        Frequency array.
    t : numpy.ndarray
        Transmission array.
    LER_position : int
        Resonance frequency position in frequency array.
    HALF_position : int
        FHWM position in frequency array.
    power : float
        Power value (in dBm) of the input signal.
    temperature : float
        Temperature (in Kelvin) at which the trace was obtained
    field : float
        External magnetic field applied during trace acquisition.
    units : str
        Units of the tranmission array.
    trace_type : str
        Type of the trace.
    model : str
        Specifies the type of model to use in fit.
    file_name : str
        Name of the file containing the trace data.
    file_path : str
        Absolute path of the file containing the trace data.
    mod : str
        Mode of the loaded trace, can be 'amp' for amplitude or 'iq' for 
        in-phase/quadrature.

    Methods
    -------
    fake_trace():
        Generates a fake trace with specified parameters and noise amplitude.

    json_load_trace():
        Loads trace data from a JSON file.

    json_sonnet_load_trace_mod():
        Loads trace data with modulation from a JSON file formatted
        specifically for the Sonnet software.

    cab_load_trace(path=None):
        Loads trace data from a CAB-formatted file.

    yebes_load_trace():
        Loads trace data from a Yebes Observatory-formatted file.

    sonnet_load_trace():
        Loads trace data from a CSV file formatted specifically for the
        Sonnet software.

    load_magnetic_trace(name=None):
        Loads multiple trace datasets corresponding to different magnetic
        field strengths from JSON files.

    do_fit():
        Fits the trace data based on the provided initial parameters and options.

    do_fit_amp():
        Fits the amplitude of the trace data to a specified model.

    do_fit_complex():
        Fits the complex (amplitude and phase) data to a specified model.

    do_nonlfit():
        Performs a nonlinear fit of the trace to a specified model which takes
        into account nonlinearity physics due to the kinetic inductance of the
        superconductor

    plot():
        Plots the trace data with options for unit conversion and FWHM
        estimation visualization.
    """

    def __init__(self, f=None, t=None, LER_position=None,
                 HALF_position=None, power=None, temperature=None,
                 field=None, units=None, trace_type=None, model=None,
                 file_name=None, file_path=None, mod=None):
        """
        Trace object with transmission data and other information. Initializes
        at None. Load method or fake_trace method are required to proper initia
        lization.

        Parameters
        -------
        f: frequency array. Numpy array.
        t: transmission array. Numpy array.
        LER_position: resonance frequency position in frequency array. Int.
        HALF_position: FHWM position in frequency array. Int.
        power: power value of the input signal. Float.
        temperature: temperature value (in Kelvin) at which trace was obtained.
                     Float.
        field: in case the trace has been acquired with an external magnetic 
               field applied. Float.
        units: units of the trace array. String.
        model: specifies the type of model to use in fit.
        file_name: name of the file containing the trace data
        file_path: absolute path of the file containing the trace data
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
        self.file_name = file_name
        self.file_path = file_path
        self.mod = mod

    def fake_trace(self, params, noise_amp, model='parallel', flim=None):

        self.model = model

        if flim == None:
            [params[7]*0.5, params[7]*1.5]

        self.f = np.linspace(flim[0], flim[1], 10000)

        noise_i = np.random.normal(0, noise_amp, size=len(self.f))
        noise_r = np.random.normal(0, noise_amp, size=len(self.f))

        if model == 'parallel':
            self.t = lorentzian(self.f, params)
        elif model == 'inline':
            self.t = in_lorentzian(self.f, params)
        self.t = (np.real(self.t) + noise_r + 1j*(np.imag(self.t) + noise_i))

        return self

    def json_load_trace(self):
        path, file, dirpath = pth.file()

        with open(path, 'r') as f:
            data = json.load(f)

        if 'powers' in data.keys():
            self.f = np.linspace(data['freq_start'],
                                 data['freq_stop'], data['num_points'])
            self.t = np.array(data['realpart']) + 1j*np.array(data['imagpart'])
            self.power = np.array(data['powers'])

        else:
            self.f = np.linspace(data['freq_start'],
                                 data['freq_stop'], data['num_points'])
            self.t = np.array(data['realpart']) + 1j*np.array(data['imagpart'])
            self.p = None

        if len(self.t.shape) == 2:
            t_aux = np.zeros(self.t.shape[0], dtype=complex)
            for i in range(self.t.shape[0]):
                t_aux[i] = self.t[i][0]
            self.t = t_aux

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

                if element == []:
                    break
                freq.append(float(element[0]))
                amp.append(float(element[5]))
                phase.append(float(element[6]))
                i += 1

        freq = np.array(freq)
        amp = np.array(amp)
        phase = np.array(phase)

        self.f = freq*1e9
        self.t = amp * np.exp(1j*phase*np.pi/180)

        return self

    def cab_load_trace(self, path=None):

        if path is None:
            path, file, dirpath = pth.file()
        else:
            file = path.split('/')[-1]

        df = pd.read_csv(path, delimiter='\t')

        self.f = np.array(df[df.columns[0]])

        if np.mean(df[df.columns[2]]) == 0.0:
            self.t = np.array(df[df.columns[1]])
            if self.t.min() < 0:
                self.t = 10**(self.t/20)
            self.mod = 'amp'
        else:
            self.t = np.array(df[df.columns[1]] + 1j*df[df.columns[2]])
            self.mod = 'iq'

        self.file_name = file
        self.file_path = path

        if 'Pow' in file and 'dB' in file:
            self.power = float(file.split('ow')[1].split('dBm')[0])
        elif 'Pow' in file and 'dB' not in file:
            self.power = float(file.split('ow')[1].split('.')[0])
        elif 'pow' in file:
            self.power = float(file.split('ow')[1].split('dBm')[0])
        elif 'dB' in file:
            self.power = float(file.split('dB')[0].split('_')[-1])

        _, self.LER_position = _guess_resonance(self)

        return self

    def yebes_load_trace(self):

        path, file, dirpath = pth.file()

        np.loadtxt(path, dtype=str, comments='!', delimiter='\t')

        aux_str = np.loadtxt(path, dtype=str, comments='!', delimiter='\t')

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
                    i += 1

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
                i += 1

        self.f = freq*1e9
        self.t = amp * np.exp(1j*phase*np.pi/180)

        return self

    def load_magnetic_trace(self, name=None):

        path, files, dirpath = pth.files()

        self.field = np.zeros(len(path))

        for i in range(len(files)):  # Loop for get each name
            self.field[i] = float(files[i].split('_')[-1])

        with open(path[0], 'r') as f:
            data = json.load(f)
            self.f = np.linspace(data['freq_start'],
                                 data['freq_stop'], data['num_points'])
            self.t = np.array(data['realpart']) + 1j*np.array(data['imagpart'])
            self.t = np.array([self.t])

        for i in range(1, len(path)):
            with open(path[i], 'r') as f:
                data = json.load(f)
                self.t = np.append(self.t,
                                   np.array(
                                       [np.array(data['realpart']) + 1j*np.array(data['imagpart'])]),
                                   axis=0)

        self.field, self.t = _order_mag_array(self.field, self.t)

        return self

    def do_fit(self, x0, ar_bound=1.00001, unwrap=False, inv=False,
               residual=False, units=None, ar_fit=False, save=False,
               xlim=None, jac='2-point', mode='amp'):

        if np.iscomplexobj(self.t):
            if mode == 'amp':
                self.t = abs(self.t)
                self.mod = mode
                results = self.do_fit_amp(x0, inv=inv, ar_fit=ar_fit,
                                          residual=residual, units=units,
                                          save=save, xlim=xlim, jac=jac)
            else:
                results = self.do_fit_complex(x0, unwrap=unwrap, ar_fit=ar_fit,
                                              inv=False, save=save, xlim=xlim, jac=jac)
        else:
            results = self.do_fit_amp(x0, inv=inv, ar_fit=ar_fit,
                                      residual=residual, units=units,
                                      save=save, xlim=xlim, jac=jac)

        return results

    def do_fit_amp(self, x0, inv=False, residual=False,
                   units=None, ar_fit=False, save=False, xlim=None,
                   jac='2-point'):

        x0[0] = _guess_amplitude(self.t)
        x0[7], self.LER_position = _guess_resonance(self)
        x0[5], self.HALF_position = _guess_kappa_coupling(self, verbose=False)
        x0[4] = _guess_kappa_internal(self)

        self.plot()

        results_amp = fit_res_com(
            self, x0, mod='amp', jac=jac, model=self.model, inv=inv,
            ar_fit=ar_fit)

        fit_plot(self, [results_amp], residual=residual,
                 save=save, xlim=xlim)

        return results_amp

    def do_fit_complex(self, x0, unwrap=False, inv=False, ar_fit=False,
                       save=False, xlim=None, jac='2-point'):

        x0[0] = _guess_amplitude(self.t)
        x0[7], self.LER_position = _guess_resonance(self)
        x0[5], self.HALF_position = _guess_kappa_coupling(self, verbose=False)
        x0[4] = _guess_kappa_internal(self)

        self.plot()
        #x0 = _guess_best_quadrant(x0, ω , t)

        results_amp = fit_res_com(
            self, x0, mod='amp', jac=jac, model=self.model,
            inv=inv, ar_fit=ar_fit
        )

        results_phase = fit_res_com(
            self, results_amp.xf, mod='phase', jac=jac,
            model=self.model, unwrap=unwrap, ar_fit=ar_fit
        )

        results_real = fit_res_com(self, results_amp.xf,
                                   mod='real', jac='2-point',
                                   model=self.model, ar_fit=ar_fit
                                   )

        results_imag = fit_res_com(self, results_amp.xf, mod='imag',
                                   jac='2-point', model=self.model,
                                   ar_fit=ar_fit
                                   )

        fit_plot(self, [results_amp, results_phase],
                 save=save, xlim=xlim)

        plot_iq_amph(results_amp, results_phase)

        fit_plot(self, [results_real, results_imag],
                 save=save, xlim=xlim)

        plot_iq_reim(results_real, results_imag)

        return results_amp

    def do_nonlfit(self, x0, residual=False, units=None, bound_tol=100,
                   jac='2-point', fit=True, save=False, xlim=None):
        """
        Overview
        --------

        The do_nonlfit method is designed to perform a nonlinear fit on the
        transmission trace (S21) as a function of frequency for a
        superconducting resonator, particularly under high-power conditions
        where the kinetic inductance nonlinearity is dominant.
        It uses the fitting function `fit_nonl_res` to calculate the parameters
        that best describe the nonlinear transmission properties of the
        resonator based on input initial guesses and constraints. The method
        finishes by plotting the fit results and the measured trace as a
        function of frequency (and optionally also plots the residuals)
        with `fit_plot` function.

        Parameters
        ----------

        - **self** (_Trace_): The instance of the `Trace` object, representing
            the transmission trace data and metadata for the resonator.

        - **x0** (_numpy array_): Initial guess parameters as a numpy array,
            containing [a, ac, ar, κi, κc, Δφ, fr0].
            These parameters respectively represent:
          - **a**: Nonlinear parameter.
          - **ac**: Trace level. (fixed)
          - **ar**: Fano phase amplitude. (fixed)
          - **κi** (_ki_): Internal loss.
          - **κc** (_kc_): Coupling to the transmission line. (fixed)
          - **Δφ** (_Δφ_): Fano phase. 
          - **fr0**: Resonance frequency in a low power regime. (fixed)
          Note that the fitting process runs over on *a*, *ki*, and *Δφ*.

        - **residual** (_bool_, _optional_): If `True`, the fit residuals will
            be plotted. Default is `False`.

        - **units** (_optional_): Specifies the units of the transmission
            trace. Default assumes a linear scale if `None`.

        - **bound_tol** (_int_, _optional_): Tolerance bounds for the *ki* 
            (internal loss) parameter. Default is `100`.

        - **jac** (_str_, _optional_): Specifies the method to compute the
            Jacobian for the least squares fit. Recommended options are 
            '2-point' or '3-point'. Default is `'2-point'`.

        - **fit** (_bool_, _optional_): Determines whether to perform the fit.
            If False, the method assumes *x0* as the final parameters without
            performing a new fit. Default is True.

        - **save** (_bool_, _optional_): If `True`, saves the final fit plot
            as a .png file and the final fit parameters as a .dat file.
            Default is False.

        - **xlim** (_tuple_, _optional_): Specifies the limits for the plot
            on the horizontal axis. If None, the plot automatically adjusts to
            display the maximum and minimum frequencies in the dataset.
            Default is `None`.

        Returns
        -------

        - **results**: (_Results_) Results class from tracelib storing the data
            resulting from the fit. This typically includes the optimized
            values of the parameters involved in the fit, among possibly other
            related information derived from the fitting process (the jacobian
            and the residuals).

        Examples
        --------

        Below is a hypothetical usage example of the `do_nonlfit` function:

        # Assuming `trace` is an instance of Trace with loaded data
        initial_params = np.array([1e-6, 0.5, 0.2, 1e-3, 2e-3, 0.1, 6e9])
        results = trace.do_nonlfit(x0=initial_params, residual=True, save=True)
        """

        results = fit_nonl_res(self, x0, bound_tol=bound_tol,
                               jac=jac, fit=fit, save=save)

        fit_plot(self, [results], residual=residual, xlim=xlim, save=save)

        return results

    def plot(self, units=None, guess_FHWM=False):

        uplt.plottoinline()

        if (self.units != None and units != None and units != self.units):
            Warning('Two different vertical units provided. Using function units.')

        if guess_FHWM:
            _, self.LER_position = _guess_resonance(self)
            _, self.HALF_position = _guess_kappa_coupling(self, verbose=True)

        t, y_label, half, resonance = uplt.set_power_unit(self, units)

        f, funit, forder = uplt.guess_magnitude_order(self.f, 'Hz')

        plt.figure(dpi=1200)
        plt.plot(
            f, t, 'k',
            label='measurement', lw=1.5, zorder=1
        )
        if self.HALF_position != None:
            plt.scatter(
                f[self.HALF_position],
                half,
                color='red'
            )
        if self.LER_position != None:
            plt.scatter(
                f[self.LER_position],
                resonance,
                color='royalblue'
            )

        plt.xlabel(f'Frequency {(funit)}', fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xlim((f.min(), f.max()))
        plt.tick_params(direction='in')
        plt.show()
        plt.close()

        return

# %% LOADING FUCTIONS


def _order_mag_array(b, t):
    t = t.transpose()
    i = 0

    while i < (len(b) - 1):
        if b[i] > b[i + 1]:
            aux = b[i]
            b[i] = b[i + 1]
            b[i + 1] = aux
            aux = t[:, i]
            t[:, i] = t[:, i + 1]
            t[:, i + 1] = aux
            i = 0
        else:
            i += 1

    return b, t

# %% FIT SUPPORT FUNCTIONS


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


def _set_fixed(x0, mod='amp'):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    if mod == 'amp':
        fixed = np.zeros(2)
        fixed[0] = x0[1]  # dt
        fixed[1] = x0[2]  # φ
        x1 = np.zeros(6)
        x1[0] = x0[0]  # a
        x1[1] = x0[3]  # ar
        x1[2] = x0[4]  # κi
        x1[3] = x0[5]  # κc
        x1[4] = x0[6]  # Δφ
        x1[5] = x0[7]  # ωr

    elif mod == 'phase':
        fixed = np.zeros(2)
        fixed[0] = x0[0]  # a
        fixed[1] = x0[7]  # ωr

        x1 = np.zeros(6)
        x1[0] = x0[1]  # dt
        x1[1] = x0[2]  # φ
        x1[2] = x0[3]  # ar
        x1[3] = x0[4]  # κi
        x1[4] = x0[5]  # κc
        x1[5] = x0[6]  # Δφ

    else:
        fixed = np.zeros(6)
        fixed[0] = x0[0]  # a
        fixed[1] = x0[3]  # ar
        fixed[2] = x0[4]  # κi
        fixed[3] = x0[5]  # κc
        fixed[4] = x0[6]  # Δφ
        fixed[5] = x0[7]  # ωr

        x1 = np.zeros(2)
        x1[0] = x0[1]  # dt
        x1[1] = x0[2]  # φ

    return x1, fixed


def _set_output(x1, fixed, mod='amp'):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    if mod == 'amp':
        xf = np.zeros(8)
        xf[0] = x1[0]  # a
        xf[1] = fixed[0]  # dt
        xf[2] = fixed[1] % (2*np.pi)  # φ
        xf[3] = x1[1]  # ar
        xf[4] = x1[2]  # κi
        xf[5] = x1[3]  # κc
        xf[6] = x1[4] % (2*np.pi)  # Δφ
        xf[7] = x1[5]  # ωr
    elif mod == 'phase':
        xf = np.zeros(8)
        xf[0] = fixed[0]  # a
        xf[1] = x1[0]  # dt
        xf[2] = x1[1] % (2*np.pi)  # φ
        xf[3] = x1[2]  # ar
        xf[4] = x1[3]  # κi
        xf[5] = x1[4]  # κc
        xf[6] = x1[5] % (2*np.pi)  # Δφ
        xf[7] = fixed[1]  # ωr
    else:
        xf = np.zeros(8)
        xf[0] = fixed[0]  # a
        xf[1] = x1[0]  # dt
        xf[2] = x1[1] % (2*np.pi)  # φ
        xf[3] = fixed[1]  # ar
        xf[4] = fixed[2]  # κi
        xf[5] = fixed[3]  # κc
        xf[6] = fixed[4] % (2*np.pi)  # Δφ
        xf[7] = fixed[5]  # ωr

    return xf


def _set_bounds(f, x0, mod='amp', ar_fit=False):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    ####### BOUNDS FOR FITTING PARAMETERS ##################
    a_upper_bound = 10.0
    a_lower_bound = 0.0

    dt_upper_bound = 1e-4
    dt_lower_bound = -1e-4

    φ_upper_bound = 2*mt.pi
    φ_lower_bound = 0

    if ar_fit:
        ar_upper_bound = 2.
        ar_lower_bound = 0.5
    else:
        ar_upper_bound = 1.00000001
        ar_lower_bound = 0.99999999

    κi_upper_bound = x0[4] + 100*x0[4]
    κi_lower_bound = x0[4] - 0.01*x0[4]

    κc_upper_bound = x0[5] + 100*x0[5]
    κc_lower_bound = x0[5] - 0.01*x0[5]

    Δφ_upper_bound = 2*np.pi
    Δφ_lower_bound = -2*np.pi

    fr_upper_bound = f.max()
    fr_lower_bound = f.min()

    if mod == 'amp':
        #a, ar, κ, κc, Δφ, ωr
        upper_bound = np.array([a_upper_bound, ar_upper_bound, κi_upper_bound,
                                κc_upper_bound, Δφ_upper_bound, fr_upper_bound])
        lower_bound = np.array([a_lower_bound, ar_lower_bound, κi_lower_bound,
                                κc_lower_bound, Δφ_lower_bound, fr_lower_bound])
    elif mod == 'phase':
        #a, ar, κ, κc, Δφ, ωr
        upper_bound = np.array([dt_upper_bound, φ_upper_bound, ar_upper_bound,
                                κi_upper_bound, κc_upper_bound, Δφ_upper_bound])
        lower_bound = np.array([dt_lower_bound, φ_lower_bound, ar_lower_bound,
                                κi_lower_bound, κc_lower_bound, Δφ_lower_bound])
    else:
        #dt, φ, κ, κc
        upper_bound = np.array([dt_upper_bound, φ_upper_bound])
        lower_bound = np.array([dt_lower_bound, φ_lower_bound])

    bounds = (lower_bound, upper_bound)

    return bounds


def _set_min_bounds(ω, x0, mod='amp'):
    #a, dt, κ, κc, Δφ, ωr
    if mod == 'amp':
        #a, ar, κ, κc, Δφ, ωr
        bounds = ((0, 1.0), (0, 10), (0, None), (0, None),
                  (0, 2*mt.pi), (ω.min(), ω.max()))
    elif mod == 'phase':
        bounds = ((0, 1.0e-6), (0, 2*mt.pi), (0, 10),
                  (0, None), (0, None), (0, 2*mt.pi))
    else:
        #dt, κ, κc
        bounds = ((0, None), (0, None))

    return bounds


def rescale(ω, t):
    ts = t/t.max()
    ωs = ω - ω.min()
    ωs = 2*ωs/ωs.max()

    return ωs, ts


def _guess_amplitude(t):
    return abs(t[0])


def _guess_phase_delay(f, t, unwrap=False):
    if unwrap:
        ph = np.unwrap(np.arctan2(np.imag(t), np.real(t)))
    else:
        ph = np.arctan2(np.imag(t), np.real(t))
    return -1*((ph[-1] - ph[0])/(f[-1] - f[0]))


def _guess_phase_zero(f, t, unwrap=False):
    dt = _guess_phase_delay(f, t, unwrap=unwrap)
    ph = np.unwrap(np.arctan2(np.imag(t), np.real(t)))
    th = ph[0] + dt*f[0]
    return th


def _guess_relative_amplitude():
    return 1.


def _guess_best_quadrant(x0, ω, t):

    x1 = x0.copy()
    q = [i*np.pi/4 for i in range(8)]
    r = []

    for i in q:
        x1[6] = i
        results = fit_res_com(ω, t, x1, mod='amp', jac='3-point', plot=False)
        r.append(sum(results.res))

    i = np.argmin(np.abs(r))

    x0[6] = q[i]

    print('Best phase initial condition is Δφ = %iπ/4' % i)

    return x0


def _guess_kappa_coupling(trace, verbose=True):

    if trace.mod == 'iq':
        t_amp = abs(trace.t)
    else:
        t_amp = trace.t

    t_amp = savgol_filter(t_amp, 51, 3)
    t_half = (t_amp.max() + t_amp.min())/2

    #t_half = np.sqrt((t_amp.max()**2 + t_amp.min()**2)/2)

    HALF_position = 0
    stop = False

    while stop == False:
        if t_amp[HALF_position] <= t_half:
            stop = True
        else:
            HALF_position = HALF_position + 1

    kappac_aprox = (trace.f[trace.LER_position] - trace.f[HALF_position])/2

    if verbose:
        print(('\033[1;36m Aprox kc is %.2e Hz \033[0m'
               % (kappac_aprox)))

    return kappac_aprox, HALF_position


def _guess_kappa_internal(trace):
    kappa_c, _ = _guess_kappa_coupling(trace, verbose=False)

    if trace.mod == 'iq':
        t_amp = abs(trace.t)
    else:
        t_amp = trace.t
    t_min = t_amp[trace.LER_position]

    kappai_aprox = kappa_c*t_min/abs((1 - t_min))

    return kappai_aprox


def _guess_phano_phase(trace):

    a = _guess_amplitude(trace.t)

    if trace.mod == 'iq':
        t_amp = abs(trace.t)
    else:
        t_amp = trace.t

    t_min = t_amp[trace.LER_position]

    kappa_c, _ = _guess_kappa_coupling(trace)
    kappa_i = _guess_kappa_internal(trace)

    A = kappa_c / (kappa_i + kappa_c)

    phano = (t_min - a**2 * (1 + A**2)) / (2 * a**2 * A)

    return np.arccos(phano)


def _guess_resonance(trace):
    # Finding LER resonance position and RESONANCE FREQUENCY: ##################

    LER_position = 0
    stop = False
    if trace.mod == 'iq':
        t_amp = abs(trace.t)
    else:
        t_amp = trace.t

    while stop == False:
        if t_amp[LER_position] == t_amp.min():
            stop = True
        else:
            LER_position = LER_position + 1

    return trace.f[LER_position], LER_position


def _calculate_residual(t, tprx):

    return t-tprx


# %% RESULT CLASS TO STORE DE RESULT OF THE FIT

class Results():
    def __init__(self, ω, t, tprx, xf, res, jac, mod):
        self.ω = ω
        self.t = t
        self.tprx = tprx
        self.xf = xf
        self.res = res
        self.jac = jac
        self.mod = mod

# %% FIT FUNCTION


def fit_res_com(trace, x0, mod='amp', jac='3-point', verbose=0, gtol=1e-15,
                xtol=1e-15, plot=True, model='parallel', unwrap=False,
                inv=False, ar_fit=False, save = False):
    
    """
    Function to fit transmission vs frequency data using scipy least squares
    method.

    Parameters
    ----------

    trace (_Trace_): The instance of the `Trace` object, representing
        the transmission trace data and metadata for the resonator.
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
    ar_fit (_boolean_): If True, fit runs over the ar parameter (the fano
            amplitude). If False, the ar parameter remains constant at 1.0
            during fit.
    save (_boolean_): If True, fit results and plot will be saved.

    Returns
    -------
    Results: Results class (in this library file)

    """

    #a, dt, φ, ar, κi, κc, Δφ, ωr

    x1, fixed = _set_fixed(x0, mod=mod)
    constat_arguments = {'fixed': fixed}

    if model == 'parallel':
        if mod == 'amp':
            fun_to_minimize = least_lorentzian_amp
            if trace.mod == 'iq':
                t = abs(trace.t)
            else:
                t = trace.t
            x1[0] = _guess_amplitude(t)

            if inv:
                fun_to_minimize = least_invlorentzian_amp
                t = abs(1/t)

        elif mod == 'phase':
            fun_to_minimize = least_lorentzian_phase
            if unwrap:
                t = np.unwrap(np.arctan2(np.imag(trace.t), np.real(trace.t)))
            else:
                t = np.arctan2(np.imag(trace.t), np.real(trace.t))
        elif mod == 'real':
            fun_to_minimize = least_lorentzian_real
            t = np.real(trace.t)
        elif mod == 'imag':
            fun_to_minimize = least_lorentzian_imag
            t = np.imag(trace.t)
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
        if mod == 'amp':
            x1[5], LER_position = _find_minimum(trace.f, t, x1[0])
        else:
            _, LER_position = _find_minimum(trace.f, t, x1[0])
    elif model == 'inline':
        if mod == 'amp':
            x1[5], LER_position = _find_maximum(trace.f, t, x1[0])

    ##### STABLISH FITTING BOUNDS ###############
    bounds = _set_bounds(trace.f, x0, mod=mod, ar_fit=ar_fit)
    
    print('\033[1;36m Executing fit \033[0m')
    if mod == 'amp':
        result = sop.least_squares(fun_to_minimize, x1, jac=jac,
                                   args=[trace.f, t], kwargs=constat_arguments,
                                   bounds=bounds, verbose=verbose, gtol=gtol,
                                   xtol=xtol)
    else:
        result = sop.least_squares(fun_to_minimize, x1, jac=jac,
                                   args=[trace.f, t], kwargs=constat_arguments,
                                   bounds=bounds, verbose=verbose, gtol=gtol,
                                   xtol=xtol, ftol=1e-15)

    xf = _set_output(result.x, fixed, mod=mod)
    res = result.fun
    jac = result.jac
    print('\033[1;36m Fit ended \033[0m')

    if mod == 'amp':
        tprx = abs(lorentzian(trace.f, xf))
    if mod == 'phase':
        tprx = np.unwrap(
            np.arctan2(np.imag(lorentzian(trace.f, xf)), np.real(lorentzian(trace.f, xf))))
    if mod == 'real':
        tprx = np.real(lorentzian(trace.f, xf))
    if mod == 'imag':
        tprx = np.imag(lorentzian(trace.f, xf))

    show_results(trace, tprx, xf, mod, save = save)

    res = _calculate_residual(t, tprx)

    return Results(trace.f, t, tprx, xf, res, jac, mod)


# %% FUNCTIONS TO PLOT AND PRESENT THE RESULTS

def show_results(trace, tprx, xf, mod, save = False):

    turns = mt.floor(xf[2]/(2*mt.pi))
    Qi = xf[6]/(2*xf[3])
    Qc = xf[6]/(2*xf[4])

    lines = ['MINIMIZATION METHOD RESULTS:',
             'Name: ' + trace.file_name,
             'a = ' + str(format(xf[0], "E")),
             'dt = ' + str(format(xf[1], "E")) + ' s',
             'φ = ' + str(round(xf[2] % (2*np.pi), 4)) + ' rad',
             'ar = ' + str(format(xf[3], "E")),
             'ki = ' + str(format(xf[4], "E")) + ' Hz',
             'kc = ' + str(format(xf[5], "E")) + ' Hz',
             'Δφ = ' + str(round(xf[6] - turns*2*mt.pi, 4)) + ' rad',
             'fr = ' + str(format(xf[7], "E")) + ' Hz',
             'Qi = ' + str(format(Qi, "E")),
             'Qc = ' + str(format(Qc, "E")),
             'Pow = %.2f dBm' % trace.power
             ]

    for line in lines:
        print(line)

    partial_name = trace.file_path.split('.')[0]
    for part in trace.file_path.split('.')[1:-1]:
        partial_name = partial_name + '.' + part

    file_results_name = (partial_name + '_FITRESULTS.' +
                         trace.file_path.split('.')[-1])

    if save:
        print('\033[1;36m Saving fit parameters \033[0m')
        with open(file_results_name, "w", encoding="utf-8") as data_file:
            for line in lines:
                data_file.write(line + '\n')
            data_file.close()    

    return


def fit_plot(trace, results=[], residual=False, xlim=None, save=False):

    uplt.plottoinline()
    matplotlib.rcParams['axes.formatter.useoffset'] = False

    f = results[0].ω

    if not residual:
        dim = (len(results), 1)
    else:
        dim = (len(results), 2)

    if len(results) == 1:
        figsize = (6, 4)
    else:
        figsize = (7, 11)

    if residual:
        figsize = (figsize[0] * 2, figsize[1])

    fig, ax = plt.subplots(dim[0], dim[1], sharex=True,
                           figsize=figsize, dpi=700)

    if len(results) == 1 or dim[1] == 1:
        ax = np.reshape(ax, dim)

    for i in range(len(results)):
        for j in range(dim[1]):
            if j == 0:
                ax[i, j].plot(f*1e-9, results[i].t, 'ko', label='measurement',
                              lw=3.5, zorder=2)
                ax[i, j].plot(f*1e-9, results[i].tprx, 'r--',
                              label='fit', lw=2.5, zorder=3)
                ax[i, j].set_ylabel(
                    'Transmission', fontsize=14.0, fontweight='bold')
            if j == 1:
                ax[i, j].plot(f*1e-9, results[i].res, 'k', label='residual',
                              lw=3.5, zorder=2)
                ax[i, j].set_ylabel(
                    'Residual', fontsize=14.0, fontweight='bold')

            ax[i, j].set_xlabel('Frequency (GHz)',
                                fontsize=14, fontweight='bold')
            ax[i, j].tick_params(direction='in')
            ax[i, j].grid(True)
            ax[i, j].set_xlim([f.min()*1e-9, f.max()*1e-9])
            ax[i, j].set_title(results[i].mod)

    if xlim != None and type(xlim) == tuple:
        plt.xlim(xlim)

    if save:
        image_name = trace.file_path.split('.')[0]
        for part in trace.file_path.split('.')[1:-1]:
            image_name = image_name + '.' + part
        plt.savefig(image_name + '.png')

    plt.show()
    plt.close()

    return f


def plot_iq_amph(results_amp, results_ph):

    t = results_amp.t * np.exp(1j*results_ph.t)
    tprx = results_amp.tprx * np.exp(1j*results_ph.tprx)

    f = plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(11, 11), sharex=True, sharey=True)
    ax[0].plot(np.real(t), np.imag(t), 'ko',
               label='measurement', lw=3.5, zorder=2)
    ax[0].plot(np.real(tprx), np.imag(tprx), 'r--',
               label='fit', lw=1.5, zorder=3)
    ax[1].plot(np.real(t) - np.real(tprx), np.imag(t) - np.imag(tprx),
               'ko', label='residual', lw=3.5, zorder=2)
    ax[0].set_title('IQ from Amplitude and Phase fits')
    ax[1].set_title('IQ residual from Amplitude and Phase fits')
    for i in range(2):
        ax[i].tick_params(direction='in')
        ax[i].grid(True)
        ax[i].set_aspect(1)
        ax[i].set_xlabel('I (a.u.)', fontsize=14, fontweight='bold')
        ax[i].set_ylabel('Q (a.u.)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
    plt.close()

    return f


def plot_iq_reim(results_real, results_imag):

    f = plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(11, 11), sharex=True, sharey=True)
    ax[0].plot(results_real.t, results_imag.t, 'ko',
               label='measurement', lw=3.5, zorder=2)
    ax[0].plot(results_real.tprx, results_imag.tprx,
               'r--', label='fit', lw=1.5, zorder=3)
    ax[1].plot(results_real.t - results_real.tprx, results_imag.t - results_imag.tprx,
               'ko', label='residual', lw=3.5, zorder=2)
    ax[0].set_title('IQ fit from Real and Imaginary fits')
    ax[1].set_title('IQ residual from Real and Imaginary fits')
    for i in range(2):
        ax[i].tick_params(direction='in')
        ax[i].grid(True)
        ax[i].set_aspect(1)
        ax[i].set_xlabel('I (a.u.)', fontsize=14, fontweight='bold')
        ax[i].set_ylabel('Q (a.u.)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
    plt.close()

    return f


# %% MODEL FUNCTIONS PREPARED FOR 2D DATA FIT INCLUDING COUPLING TO SPIN

def M_2f(κi, κc, fr, γ, G, Ω, f):
    
    """
    Function to build the matrix of the resonator-spin coupling model

    Parameters
    ----------

    κi: float. internal loss rate of the resonator.
    κc: float. loss rate of the resonator due to the coupling to the TL.
    fr: float. resonance frequency of the resonator.
    γ: float. internal loss rate of the spin.
    G: float. coupling rate between the spin and the resonator.
    f: float. frequency at which the model is solved.

    Returns
    -------
    Returns the matrix of the model.

    WARNING! This function is only used by other functions.

    """

    f = f
    κ = κi + κc
    return np.array([[1j*(f - fr) + κ, 1j*G],
                     [1j*G, 1j*(f - Ω) + γ]], dtype=complex)


def X1_2f(ac, dt, φ, ar, κi, κc, Δφ, fr, γ, G, Ω, f):
    
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
    fr: float. resonance frequency of the resonator.
    γ: float. internal loss rate of the spin.
    G: float. coupling rate between the spin and the resonator.
    f: numpy array. frequencies at which the model is solved.

    Returns
    -------
    Returns the complex transmission

    NOTE:

    """

    f = f
    Mat = M_2f(κi, κc, fr, γ, G, Ω, f)
    inverse = np.linalg.inv(Mat)
    f1 = np.array([-1j*np.sqrt(ar*κc*np.exp(1j*Δφ)), 0.],  dtype=complex)
    xarray = np.matmul(inverse, f1)

    return ac*(np.exp(-1j*(f*dt - φ)))*(1-1j*np.sqrt(ar*κc*np.exp(1j*Δφ))*xarray[0])


def lorentzian(f, params):
    #a, dt, φ, ar, κ, κc, Δφ, fr

    ac = params[0]
    dt = params[1]
    φ = params[2]
    ar = params[3]
    κi = params[4]
    κc = params[5]
    Δφ = params[6]
    fr = params[7]

    # t = ac*(np.exp(-1j*(ω*dt - φ)))*(1 - ar*κc*np.exp(1j*Δφ)/(1j*(ω - ωr) + κ))

    myfunction = np.vectorize(X1_2f)

    t = myfunction(ac, dt, φ, ar, κi, κc, Δφ, fr, 0, 0, 1e9, f)

    return t


# %% MODEL FUNCTIONS
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

    t = ac*(np.exp(-1j*(ω*dt - φ))) * \
        (1 + ar*np.exp(1j*Δφ)*(κc/(1j*(ωr - ω) + κ)))

    return t


def least_lorentzian_amp(not_fixed, ω, t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = not_fixed[0]  # a
    params[1] = fixed[0]  # dt
    params[2] = fixed[1]  # φ
    params[3] = not_fixed[1]  # ar
    params[4] = not_fixed[2]  # κi
    params[5] = not_fixed[3]  # κc
    params[6] = not_fixed[4]  # Δφ
    params[7] = not_fixed[5]  # ωr

    return t - abs(lorentzian(ω, params))


def least_invlorentzian_amp(not_fixed, ω, t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = not_fixed[0]  # a
    params[1] = fixed[0]  # dt
    params[2] = fixed[1]  # φ
    params[3] = not_fixed[1]  # ar
    params[4] = not_fixed[2]  # κi
    params[5] = not_fixed[3]  # κc
    params[6] = not_fixed[4]  # Δφ
    params[7] = not_fixed[5]  # ωr

    return t - abs(1/lorentzian(ω, params))


def least_lorentzian_phase(not_fixed, ω, t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = not_fixed[2]  # ar
    params[4] = not_fixed[3]  # κi
    params[5] = not_fixed[4]  # κc
    params[6] = not_fixed[5]  # Δφ
    params[7] = fixed[1]  # ωr

    return t - np.unwrap(np.arctan2(np.imag(lorentzian(ω, params)),
                                    np.real(lorentzian(ω, params))))


def least_in_lorentzian_amp(not_fixed, ω, t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = not_fixed[0]  # a
    params[1] = fixed[0]  # dt
    params[2] = fixed[1]  # φ
    params[3] = not_fixed[1]  # ar
    params[4] = not_fixed[2]  # κi
    params[5] = not_fixed[3]  # κc
    params[6] = not_fixed[4]  # Δφ
    params[7] = not_fixed[5]  # ωr

    return t - abs(in_lorentzian(ω, params))


def least_in_lorentzian_phase(not_fixed, ω, t, fixed):
    #a, dt, φ, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = not_fixed[0]  # a
    params[1] = fixed[0]  # dt
    params[2] = fixed[1]  # φ
    params[3] = not_fixed[1]  # ar
    params[4] = not_fixed[2]  # κi
    params[5] = not_fixed[3]  # κc
    params[6] = not_fixed[4]  # Δφ
    params[7] = not_fixed[5]  # ωr

    return t - np.arctan2(in_lorentzian(ω, params))


def min_lorentzian_amp(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = not_fixed[0]  # a
    params[1] = fixed[0]  # dt
    params[2] = fixed[1]  # φ
    params[3] = not_fixed[1]  # ar
    params[4] = not_fixed[2]  # κi
    params[5] = not_fixed[3]  # κc
    params[6] = not_fixed[4]  # Δφ
    params[7] = not_fixed[5]  # ωr

    return np.sum(np.abs(t - abs(lorentzian(ω, params)))**2)


def min_lorentzian_phase(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = not_fixed[1]  # ar
    params[4] = not_fixed[2]  # κi
    params[5] = not_fixed[3]  # κc
    params[6] = not_fixed[4]  # Δφ
    params[7] = fixed[1]  # ωr

    return np.sum(np.abs(t - np.unwrap(np.arctan2(np.imag(lorentzian(ω, params)),
                                                  np.real(lorentzian(ω, params)))))**2)


def min_in_lorentzian_amp(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = not_fixed[0]  # a
    params[1] = fixed[0]  # dt
    params[2] = fixed[1]  # φ
    params[3] = not_fixed[1]  # ar
    params[4] = not_fixed[2]  # κi
    params[5] = not_fixed[3]  # κc
    params[6] = not_fixed[4]  # Δφ
    params[7] = not_fixed[5]  # ωr

    return np.sum(np.abs(t - abs(lorentzian(ω, params)))**2)


def least_lorentzian_real(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = fixed[1]  # ar
    params[4] = fixed[2]  # κi
    params[5] = fixed[3]  # κc
    params[6] = fixed[4]  # Δφ
    params[7] = fixed[5]  # ωr

    return t - np.real(lorentzian(ω, params))


def min_lorentzian_real(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = fixed[1]  # ar
    params[4] = fixed[2]  # κi
    params[5] = fixed[3]  # κc
    params[6] = fixed[4]  # Δφ
    params[7] = fixed[5]  # ωr

    return np.sum(np.abs(t - np.real(lorentzian(ω, params)))**2)


def min_in_lorentzian_real(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = fixed[1]  # ar
    params[4] = fixed[2]  # κi
    params[5] = fixed[3]  # κc
    params[6] = fixed[4]  # Δφ
    params[7] = fixed[5]  # ωr

    return np.sum(np.abs(t - np.real(in_lorentzian(ω, params)))**2)


def least_lorentzian_imag(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = fixed[1]  # ar
    params[4] = fixed[2]  # κi
    params[5] = fixed[3]  # κc
    params[6] = fixed[4]  # Δφ
    params[7] = fixed[5]  # ωr

    return t - np.imag(lorentzian(ω, params))


def min_lorentzian_imag(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = fixed[1]  # ar
    params[4] = fixed[2]  # κi
    params[5] = fixed[3]  # κc
    params[6] = fixed[4]  # Δφ
    params[7] = fixed[5]  # ωr

    return np.sum(np.abs(t - np.imag(lorentzian(ω, params)))**2)


def min_in_lorentzian_imag(not_fixed, ω, t, fixed):
    #a, φ, dt, ar, κ, κc, Δφ, ωr
    params = np.zeros(8)

    params[0] = fixed[0]  # a
    params[1] = not_fixed[0]  # dt
    params[2] = not_fixed[1]  # φ
    params[3] = fixed[1]  # ar
    params[4] = fixed[2]  # κi
    params[5] = fixed[3]  # κc
    params[6] = fixed[4]  # Δφ
    params[7] = fixed[5]  # ωr

    return np.sum(np.abs(t - np.imag(in_lorentzian(ω, params)))**2)


# %% FUNCTIONS FOR FITTING NON LINEAR REGIME TRANSMISSION

def fit_nonl_res(trace, x0, bound_tol=1000, jac='3-point', verbose=0,
                 gtol=1e-15, xtol=1e-15, plot=True, fit=True, save=False):

    # params = (a, ac, ar, κi, κc, Δφ, fr0)
    fun_to_minimize = least_nonl_transmission
    
    # with non linear regime fit only amplitude of the transmission is used
    t = abs(trace.t)

    ##### FIND TRANSMISSION MINIMUM IN TRACE ###############
    _, LER_position = _find_minimum(trace.f, t, x0[1])

    ##### STABLISH FITTING BOUNDS ###############
    bounds = _set_nonl_bounds(trace.f, x0, bound_tol=bound_tol)

    # x1 = (a, κi, Δφ)
    x1 = [x0[0], x0[3], x0[5]]
    # fixed = (ac, ar, κc, fr0)
    fixed = [x0[1], x0[2], x0[4], x0[6]]
    xf = x0
    constat_arguments = {'fixed': fixed}
    if fit:

        print('\033[1;36m Executing fit \033[0m')

        result = sop.least_squares(fun_to_minimize, x1, jac=jac,
                                   kwargs=constat_arguments,
                                   args=[trace.f, t],
                                   bounds=bounds,
                                   verbose=verbose,
                                   gtol=gtol, xtol=xtol)

        xf[0] = result.x[0]
        xf[3] = result.x[1]
        xf[5] = result.x[2]

        res = result.fun
        jac = result.jac

        print('\033[1;36m Fit ended \033[0m')

    else:
        print('\033[1;36m No fit execution \033[0m')
        res, jac = (None, None)

    tprx = abs(nonl_transmission(trace.f, xf))

    show_nonl_results(trace, tprx, xf, mod='amp', save=save)

    res = _calculate_residual(t, tprx)

    return Results(trace.f, t, tprx, xf, res, jac, mod='amp')


def _solver(y0, a):

    from util import Solver

    A = 4
    B = - 4*y0
    C = 1
    D = - y0 - a

    return Solver.cubic(A, B, C, D)


def _check_roots(roots):
    if np.isrealobj(roots):
        return roots, True
    else:
        return roots[0].real, False


''' ############# MODELS ##################### '''


def nonl_transmission(f, params):

    # params = (a, ac, ar, κi, κc, Δφ, fr0)

    a = params[0]
    ki = params[3]
    kc = params[4]
    fr0 = params[6]

    Qi = fr0/(2*ki)
    Qc = fr0/(2*kc)

    Ql = (1/Qc + 1/Qi)**(-1)
    y0 = Ql * (f - fr0)/fr0

    t = []

    for i in range(len(y0)):
        roots = _solver(y0[i], a)
        roots, isreal = _check_roots(roots)

        if isreal:
            fr = Ql*f[i] / (roots[1] + Ql)

        else:
            fr = Ql*f[i] / (roots + Ql)

        params[6] = fr
        t.append(abs(nonl_lorentzian(f[i], params[1:])))

    return np.array(t)


def least_nonl_transmission(not_fixed, f, t, fixed):

    # params = (a, ac, ar, κi, κc, Δφ, fr0)

    params = np.zeros(7)

    params[0] = not_fixed[0]
    params[1] = fixed[0]
    params[2] = fixed[1]
    params[3] = not_fixed[1]
    params[4] = fixed[2]
    params[5] = not_fixed[2]
    params[6] = fixed[3]

    return t - nonl_transmission(f, params)


def nonl_lorentzian(f, params):
    ''' Non linear Kinetic Inductance regime '''

    # params = (a, ac, ar, κi, κc, Δφ, fr0)

    ac = params[0]
    ar = params[1]
    κi = params[2]
    κc = params[3]
    Δφ = params[4]
    fr = params[5]

    κ = κc + κi

    t = ac*(1 - ar*np.exp(1j*Δφ)*(κc/(1j*(fr - f) + κ)))

    return t


def _set_nonl_bounds(f, x0, bound_tol=1000):

    a_upper_bound = 20.
    a_lower_bound = 0.

    κi_upper_bound = x0[3] + bound_tol*x0[3]
    κi_lower_bound = x0[3] - bound_tol*x0[3]

    phase_upper_bound = 2*np.pi
    phase_lower_bound = 0

    #a, κ, κc, Δφ, fr
    upper_bound = np.array([a_upper_bound, κi_upper_bound, phase_upper_bound])
    lower_bound = np.array([a_lower_bound, κi_lower_bound, phase_lower_bound])

    bounds = (lower_bound, upper_bound)

    return bounds


def show_nonl_results(trace, tprx, xf, mod, save = False):

    turns = mt.floor(xf[5]/(2*mt.pi))
    Qi = xf[6]/(2*xf[3])
    Qc = xf[6]/(2*xf[4])

    lines = ['MINIMIZATION METHOD RESULTS:',
             'Name: ' + trace.file_name,
             'a = ' + str(format(xf[0], "E")),
             'ac = ' + str(format(xf[1], "E")),
             'ar = ' + str(format(xf[2], "E")),
             'ki = ' + str(format(xf[3], "E")) + ' Hz',
             'kc = ' + str(format(xf[4], "E")) + ' Hz',
             'Δφ = ' + str(round(xf[5] - turns*2*mt.pi, 4)) + ' rad',
             'fr0 = ' + str(format(xf[6], "E")) + ' Hz',
             'Qi = ' + str(format(Qi, "E")),
             'Qc = ' + str(format(Qc, "E")),
             'Pow = %.2f dBm' % trace.power
             ]

    for line in lines:
        print(line)

    partial_name = trace.file_path.split('.')[0]
    for part in trace.file_path.split('.')[1:-1]:
        partial_name = partial_name + '.' + part

    file_results_name = (partial_name + '_FITRESULTS.' +
                         trace.file_path.split('.')[-1])

    if save:
        print('\033[1;36m Saving fit parameters \033[0m')
        with open(file_results_name, "w", encoding="utf-8") as data_file:
            for line in lines:
                data_file.write(line + '\n')
            data_file.close()

    return

# %% POWER SWEEP ANALYSIS


def power_sweep_analysis(jac='2-point', mode='amp', units='linear',
                         ar_fit=False, xlim=None, unwrap=False,
                         residual=False, inv=False):

    paths, files, dirpath = pth.files()

    trace = {}
    for path in paths:
        trace[path] = Trace(model='parallel')
        trace[path].cab_load_trace(path=path)

    fit_results = {}
    j = 1
    for i in trace:
        print(('\033[1;36m Trace number %i / %i at %.2f dBm \033[0m'
               % (j, len(trace), trace[i].power)))

        resonance, resonance_position = _guess_resonance(trace[i])

        x0 = np.array([_guess_amplitude(trace[i].t),
                       _guess_phase_delay(
                           trace[i].f, trace[i].t, unwrap=False),
                       _guess_phase_zero(trace[i].f, trace[i].t, unwrap=False),
                       _guess_relative_amplitude(),
                       _guess_kappa_internal(trace[i]),
                       _guess_kappa_coupling(trace[i])[0],
                       1.0,
                       resonance])

        fit_results[i] = trace[i].do_fit(x0, unwrap=True, inv=False,
                                         units='linear', residual=False, ar_fit=ar_fit,
                                         save=True, jac=jac, mode=mode, xlim=xlim)

        j += 1

    a = {}
    ki = {}
    kc = {}
    fr = {}
    Qi = {}
    Qc = {}
    P = {}
    Pwat = {}
    numphot = {}
    for i in fit_results:
        a[i] = fit_results[i].xf[0]
        ki[i] = fit_results[i].xf[4]
        kc[i] = fit_results[i].xf[5]
        fr[i] = fit_results[i].xf[7]
        Qi[i] = fr[i]/(2*ki[i])
        Qc[i] = fr[i]/(2*kc[i])
        P[i] = trace[i].power
        Pwat[i] = conv.dBmtoP(P[i])
        numphot[i] = conv.Ptophotons(
            Pwat[i], Qc[i]*Qi[i]/(Qc[i]+Qi[i]), 2*np.pi*fr[i], Qc[i])

    fig2, ax2 = plt.subplots(figsize=(9, 7), dpi=uplt.get_dpi())
    for i in trace:
        ax2.scatter(numphot[i], Qi[i], marker='o', s=100, color='tomato')
        ax2.scatter(numphot[i], Qc[i], marker='s', s=100, color='royalblue')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(['Qi', 'Qc'], fontsize=20)
    ax2.set_xlabel("Number of photons", size=20)
    ax2.set_ylabel("Quality factors", size=20)
    ax2.tick_params(which='both', direction='in')
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='minor', labelsize=12)
    ax2.grid(which='both')
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(9, 7), dpi=uplt.get_dpi())
    for i in trace:
        ax3.plot(numphot[i], fr[i], 'sg', markersize=10)
    ax3.legend(['fres'])
    ax3.set_xlabel("Number of photons", size=18)
    ax3.set_ylabel("Resonance Frequency (GHz)", size=18)
    ax3.tick_params(direction='in')
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='minor', labelsize=12)
    ax3.grid(which='both')
    plt.show()
    fig3.tight_layout()

    fig4, ax4 = plt.subplots(figsize=(9, 7), dpi=uplt.get_dpi())
    for i in trace:
        ax4.plot(numphot[i], fr[i], 'sg', markersize=10)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend(['fres'])
    ax4.set_xlabel("Number of photons", size=18)
    ax4.set_ylabel("Resonance Frequency (GHz)", size=18)
    ax4.tick_params(direction='in')
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.tick_params(axis='both', which='minor', labelsize=12)
    ax4.grid(which='both')
    plt.show()
    fig4.tight_layout()

    return
