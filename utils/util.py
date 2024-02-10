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
        
        if type(magnitude) == float: 
            if magnitude == 0.: return magnitude, unit, 0
        
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
    
    def set_power_unit(trace, units):
        
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
        
        if trace.mod == 'iq': t = abs(trace.t)
        else: t = trace.t
        
        if type(units) != str:
            Warning('Invalid units format. Vertical units set in dB')
            t_out = 20*np.log10(t)
            if trace.HALF_position != None: half = 20*np.log10(t[trace.HALF_position])
            else: half = None
            if trace.LER_position != None: resonance = 20*np.log10(t[trace.LER_position])
            else: resonance = None
            tunit = '$|S_{21}|$ (dB)'
        
        else:
            if units == 'linear':
                t_out = t
                if trace.HALF_position != None: half = t[trace.HALF_position]
                else: half = None
                if trace.LER_position != None:  resonance = t[trace.LER_position]
                else: resonance = None
                tunit = '$|S_{21}|$ (linear)'
                
            elif units == 'dB':
                if trace.HALF_position != None: half = 20*np.log10(t[trace.HALF_position])
                else: half = None 
                if trace.LER_position != None: resonance = 20*np.log10(t[trace.LER_position])
                else: resonance = None
                tunit = '$|S_{21}|$ (dB)'
            else:
                Warning('Non recognized units. Vertical units set in dB')
                t_out = 20*np.log10(t)
                if trace.HALF_position != None: half = 20*np.log10(t[trace.HALF_position])
                else: half = None
                if trace.LER_position != None: resonance = 20*np.log10(t[trace.LER_position])
                else: resonance = None
                tunit = '$|S_{21}|$ (dB)'
                
        return t_out, tunit, half, resonance
    
    
    def get_dpi():
        import sys
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        screen = app.screens()[0]
        dpi = screen.physicalDotsPerInch()
        app.quit()
        
        return dpi
    
    def plottoinline():
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'inline')
        
        return
    
    def plottoqt():
        from IPython import get_ipython
        get_ipython().run_line_magic('matplotlib', 'qt')
        
    
#%%

class Solver():
    
    def cubic(a, b, c, d):

        if (a == 0 and b == 0):                     # Case for handling Liner Equation
            return np.array([(-d * 1.0) / c])                 # Returning linear root as numpy array.

        elif (a == 0):                              # Case for handling Quadratic Equations

            D = c * c - 4.0 * b * d                       # Helper Temporary Variable
            if D >= 0:
                D = mt.sqrt(D)
                x1 = (-c + D) / (2.0 * b)
                x2 = (-c - D) / (2.0 * b)
            else:
                D = mt.sqrt(-D)
                x1 = (-c + D * 1j) / (2.0 * b)
                x2 = (-c - D * 1j) / (2.0 * b)
                
            return np.array([x1, x2])               # Returning Quadratic Roots as numpy array.

        f = _findF(a, b, c)                          # Helper Temporary Variable
        g = _findG(a, b, c, d)                       # Helper Temporary Variable
        h = _findH(g, f)                             # Helper Temporary Variable

        if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
            if (d / a) >= 0:
                x = (d / (1.0 * a)) ** (1 / 3.0) * -1
            else:
                x = (-d / (1.0 * a)) ** (1 / 3.0)
            return np.array([x, x, x])              # Returning Equal Roots as numpy array.

        elif h <= 0:                                # All 3 roots are Real

            i = mt.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
            j = i ** (1 / 3.0)                      # Helper Temporary Variable
            k = mt.acos(-(g / (2 * i)))           # Helper Temporary Variable
            L = j * -1                              # Helper Temporary Variable
            M = mt.cos(k / 3.0)                   # Helper Temporary Variable
            N = mt.sqrt(3) * mt.sin(k / 3.0)    # Helper Temporary Variable
            P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

            x1 = 2 * j * mt.cos(k / 3.0) - (b / (3.0 * a))
            x2 = L * (M + N) + P
            x3 = L * (M - N) + P

            return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.

        elif h > 0:                                 # One Real Root and two Complex Roots
            R = -(g / 2.0) + mt.sqrt(h)           # Helper Temporary Variable
            if R >= 0:
                S = R ** (1 / 3.0)                  # Helper Temporary Variable
            else:
                S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
            T = -(g / 2.0) - mt.sqrt(h)
            if T >= 0:
                U = (T ** (1 / 3.0))                # Helper Temporary Variable
            else:
                U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

            x1 = (S + U) - (b / (3.0 * a))
            x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * mt.sqrt(3) * 0.5j
            x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * mt.sqrt(3) * 0.5j

            return np.array([x1, x2, x3])           # Returning One Real Root and two Complex Roots as numpy array.
    
#%% Internal functions

# Solver helper function to return float value of f.
def _findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Solver helper function to return float value of g.
def _findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Solver helper function to return float value of h.
def _findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)

    
def _aux_round(mag):
    
    order = mt.log(mag, 10)
    if abs(mt.ceil(order) - order) <= 0.000000000000001: order = mt.ceil(order)
    else: order = mt.floor(order)
    
    return order