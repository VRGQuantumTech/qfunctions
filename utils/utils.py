# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:31:25 2023

Utilities library.

@author: Victor Rollano
"""

import numpy as np
import math as mt


class plot():
    
    def guess_magnitude_order(magnitude, unit = None):
        
        """
        Guess order of a magnitude. For example, if magnitude is a numpy array of
        frequencies from 1.0e9 Hz to 2.0e9 Hz, the function guess the best convertion
        for the magnitude, in this case GHz and scales it. The function returns the
        scaled magnitude (an array from 1.0 GHz to 2.0 GHz), the unit (GHz), and 
        the order of the scaling (9)
        
        Parameters
        -------
        magnitude: numpy array, list or float. Number to guess the order.
        Unit: string. Unit of the provided magnitude 
        
        Returns
        -------
        magnitude: numpy array or float.
                   Provided magnitude scaled to the proper units.
        unit: string. Unit of the scaled magnitude
        order: int. Order of the scaling performed to the magnitude
        """
        
        mag_max = np.amax(magnitude)
        
        try:
            order =_aux_round(mag_max)     
        except: Exception('Provide list, numpy array, int or float')
        
        if type(magnitude) is list: magnitude = np.array(list)
            
        if unit is not None:
            if order >= 9:
                magnitude = magnitude*1e-9
                unit = f'G{unit}'
            elif order >= 6 and order < 9:
                magnitude = magnitude*1e-6
                unit = f'M{unit}'
            elif order >= 3 and order < 6:
                magnitude = magnitude*1e-3
                unit = f'k{unit}'
            elif order < 3:
                pass
        else:
            magnitude = magnitude*10**(- order)
            unit = ''
        
        return magnitude, unit, order
    
    def set_power_unit(units, t, HALF_position = None, LER_position = None):
        
        """
        Set power array units in linear or dB.
        
        Parameters
        -------
        units: string. 'linear' or 'dB'
        t: numpy array. Trace from a measurement.
        HALF_position: int. Position of the FHWM in t.
        LER_position: int. Position of the resonance frequency in t.
        
        Returns
        -------
        t_out: numpy array. Trace from measurement in the specified units.
        tunit: string. Unit of the returned trace
        half: float. Value of the transmission at the FWHM.
        resonance: float. Value of the transmission at resonance.
        """
        
        if type(units) != str:
            Warning('Invalid units format. Vertical units set in dB')
            t_out = 20*np.log10(abs(t))
            if HALF_position != None: half = 20*np.log10(abs(t[HALF_position]))
            else: half = None
            if LER_position != None: resonance = 20*np.log10(abs(t[LER_position]))
            else: resonance = None
            tunit = '$|S_{21}|$ (dB)'
        
        else:
            if units == 'linear':
                t_out = abs(t)
                if HALF_position != None: half = abs(t[HALF_position])
                else: half = None
                if LER_position != None:  resonance = abs(t[LER_position])
                else: resonance = None
                tunit = '$|S_{21}|$ (linear)'
                
            elif units == 'dB':
                if HALF_position != None: half = 20*np.log10(abs(t[HALF_position]))
                else: half = None 
                if LER_position != None: resonance = 20*np.log10(abs(t[LER_position]))
                else: resonance = None
                tunit = '$|S_{21}|$ (dB)'
            else:
                Warning('Non recognized units. Vertical units set in dB')
                t_out = 20*np.log10(abs(t))
                if HALF_position != None: half = 20*np.log10(abs(t[HALF_position]))
                else: half = None
                if LER_position != None: resonance = 20*np.log10(abs(t[LER_position]))
                else: resonance = None
                tunit = '$|S_{21}|$ (dB)'
                
        return t_out, tunit, half, resonance
    
#%% Internal functions
    
def _aux_round(mag):
    
    order = mt.log(mag, 10)
    if abs(mt.ceil(order) - order) <= 0.000000000000001: order = mt.ceil(order)
    else: order = mt.floor(order)
    
    return order
