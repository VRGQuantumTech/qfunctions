# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:53:51 2022

@author: Victor
"""

import numpy as np
import tracelib as tl

#%%

""" These functions load the data from measurement (load_trace) or simulation
(sonnet_load_trace). They are includen in fit_functions library (ft) """
trace = tl.Trace(model = 'parallel')
trace.yebes_load_trace()
trace.plot(units='linear', guess_FHWM = True)

#%%

""" This code makes the fit and plot, you don't need to run anything else """

#ω,t,p = ft.load_trace()
model = 'parallel'
params = np.array([5.59e-1, -2.36E-07, 5.85, 1.0001E+00,
                   4.355e2, 5.2486e3, 0., 5.672e8])

#trace.fake_trace(params, 1e-4, model = model, flim = [params[7]*0.9994, params[7]*1.0005])
trace.yebes_load_trace()

resonance, resonance_position = tl._guess_resonance(trace.f, trace.t)

x0 = np.array([tl._guess_amplitude(trace.t),
               tl._guess_phase_delay(trace.f,trace.t, unwrap = True),
               tl._guess_phase_zero(trace.f,trace.t, unwrap = True),
               tl._guess_relative_amplitude(),
               tl._guess_kappa_internal(resonance_position, trace.f, trace.t),
               tl._guess_kappa_coupling(resonance_position, trace.f, trace.t)[0],
               1.0,
               resonance])

trace.do_fit(x0, unwrap = True, inv = False, units = 'linear', residual = False)

