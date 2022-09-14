#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import qutip
import numpy as np
import scipy.io as sio
import math
import h5py
import pycuda as cuda
from pycuda import compiler, gpuarray
import pycuda.autoinit
import timeit
import scipy.optimize as sop
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from IPython.display import clear_output as cl
import os
import tkinter as tk # Import the module to load the paths of the files
import tkinter.filedialog # The function that opens the file dialog to select the file/files/directory
from natsort import natsorted

"""    ########################## SAMPLE CLASS ######################       """

class Sample():
    
    "Sample Class has all the information regarding to the Hamiltonian we are working with."
    
    def __init__(sample, S,I,NS,NI,NJ,Ap,Az,p,Ix,Iy,Iz,Sx,Sy,Sz,ge,gi,E,kp,D,kz,Ms,Mi,ist_ab,density,B_steven,J):
        
        sample.S = S # electronic spin
        sample.I = I # nuclear spin
        sample.NS = NS # Electronic Spin Dimension
        sample.NI = NI # Nuclear Spin Dimension
        sample.NJ = NJ # Total Spin Dimension
        sample.Ap = Ap # Perpendicular Hyperfine constant
        sample.Az = Az # Z - Hyperfine constant
        sample.p = p # Quadrupolar Interaction constant
        sample.Ix = Ix # Nuclear Spin X qutip matrix
        sample.Iy = Iy # Nuclear Spin Y qutip matrix
        sample.Iz = Iz # Nuclear Spin Z qutip matrix
        sample.Sx = Sx # Electronic Spin X qutip matrix
        sample.Sy = Sy # Electronic Spin Y qutip matrix
        sample.Sz = Sz # Electronic Spin Z qutip matrix
        sample.ge = ge # Electronic giromagnetic constant (Zeeman term)
        sample.gi = gi # Nuclear giromagnetic constant (Zeeman term)
        sample.E = E # Coupling induced by local strain 
        sample.kp = kp # Coupling induced by electric fields
        sample.D = D # Zero field splitting induced by local strain
        sample.kz = kz # Zero field splitting induced by electric fields
        sample.Ms = Ms # Electronic spin states vector
        sample.Mi = Mi # Nuclear spin states vector
        sample.ist_ab = ist_ab # Isotopical concentration
        sample.density = density # Spin density
        sample.B_steven = B_steven # Bij paramenters for high-order Oij operators
        sample.J = J # Exchange interaction value
        
    def hamiltonian(sample, B_ext = np.array([0,0,0]), E_ext = np.array([0,0,0])):
        
        # This function calculates a general Hamiltonian given a set of values and arrays.
        # It returns a single array H.
        
        "ELECTRONIC ZEEMAN PARAMETERS"
        mu_B = 9.27401e-24 / 6.6261e-34 #Hz/T
        
        "PLUS AND MINUS SPIN OPERATORS"
        Sp = sample.Sx + 1j*sample.Sy
        Sm = sample.Sx - 1j*sample.Sy
        
        "SQUARED OPERATORS"
        S2=sample.Sx*sample.Sx+sample.Sy*sample.Sy+sample.Sz*sample.Sz
        #I2=sample.Ix*sample.Ix+sample.Iy*sample.Iy+sample.Iz*sample.Iz
        
        "ELECTRONIC ZEEMAN INTERACTION"
        ZE11 = mu_B*sample.ge[0]*B_ext[0]*qutip.tensor(sample.Sx,qutip.qeye(sample.NI))
        ZE22 = mu_B*sample.ge[1]*B_ext[1]*qutip.tensor(sample.Sy,qutip.qeye(sample.NI))
        ZE33 = mu_B*sample.ge[2]*B_ext[2]*qutip.tensor(sample.Sz,qutip.qeye(sample.NI))
        
        
        "NUCLEAR ZEEMAN PARAMETERS"
      
        mu_N = 5.05078E-27 / 6.6261e-34 #Hz/T
        
        "NUCLEAR ZEEMAN INTERACTION"
        
        NZE11 = mu_N*sample.gi*B_ext[0]*qutip.tensor(qutip.qeye(sample.NS), sample.Ix)
        NZE22 = mu_N*sample.gi*B_ext[1]*qutip.tensor(qutip.qeye(sample.NS), sample.Iy)
        NZE33 = mu_N*sample.gi*B_ext[2]*qutip.tensor(qutip.qeye(sample.NS), sample.Iz)
            
        "HYPERFINE INTERACTION"
        HF11 = sample.Ap*qutip.tensor(sample.Sx,sample.Ix)
        HF22 = sample.Ap*qutip.tensor(sample.Sy,sample.Iy)
        HF33 = sample.Az*qutip.tensor(sample.Sz,sample.Iz)
        
        "QUADRUPOLAR INTERACTION NUCLEAR SPIN"
        Q33 = sample.p*qutip.tensor(qutip.qeye(sample.NS),sample.Iz)**2
            
        "QUDRUPOLAR INTERACTION ELECTRONIC SPIN (ZERO FIELD SPLITTING TERM)"
        QE33 = (sample.D + sample.kz*E_ext[2])*qutip.tensor(sample.Sz, qutip.qeye(sample.NI))**2
        
        
        "STRAIN INTERACTION"
        STI = (sample.E + sample.kp*E_ext[0])*qutip.tensor(sample.Sx, qutip.qeye(sample.NI))**2 - (sample.E + sample.kp*E_ext[1])*qutip.tensor(sample.Sy, qutip.qeye(sample.NI))**2
            
            
        "HIGH ORDER TERMS"
        O20=sample.B_steven['B20']*qutip.tensor(3*sample.Sz**2-S2,qutip.qeye(sample.NI))
        O40=sample.B_steven['B40']*qutip.tensor(35*sample.Sz**4-30*S2*sample.Sz**2+25*sample.Sz**2-6*S2+3*S2**2,qutip.qeye(sample.NI))
        O44=sample.B_steven['B44']*qutip.tensor(0.5*(Sp**4+Sm**4),qutip.qeye(sample.NI))
        O60=qutip.tensor(231*sample.Sz**6-315*S2*sample.Sz**4+735*sample.Sz**4+105*S2**2*sample.Sz**2-
                          525*S2*sample.Sz**2+294*sample.Sz**2-5*S2**3+40*S2**2-60*S2,qutip.qeye(sample.NI))
        O60=sample.B_steven['B60']*O60
            
        Oij = O20 + O40 +  O44 + O60 
        
        "COMPLETE HAMILTONIAN"
        H = ZE11 + ZE22 + ZE33 + NZE11 + NZE22 + NZE33 + HF11 + HF22 + HF33 + Q33 + QE33 + STI + Oij
    
        return H

    def load_sample(sample):
    
        if sample == '171Yb_Trensal':
            "SPIN VALUES (dimensionless)"
            S = 0.5; I = 0.5; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [2.935,2.935,4.29]; gi = -0.02592;
            "HYPERFINE VALUES (Hz)"
            Ap = 2.2221e9; Az = 3.3729e9 # Hz - from K. Pedersen et al. Inorg. Chem. 2015, 54, 15, 7600–7606
            #Ap = 0.0; Az = 3.957754e9 # Hz - From P.E. Atkinson et al. PRA 100 042505 (2019)
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 0.0 ; kz = 0.0
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 0.14
            "SPIN DENSITY"
            density = 1.6941e27 #spin/m3
            
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
            
        if sample == '172Yb_Trensal':
            "SPIN VALUES (dimensionless)"
            S = 0.5; I = 0.0; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [2.935,2.935,4.29]; gi = 0; # Mail Stergios (based on 172 Isotope)
            "HYPERFINE VALUES (Hz)"
            Ap = 0.0; Az = 0.0
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 0.0 ; kz = 0.0
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 0.70
            "SPIN DENSITY"
            density = 1.6941e27 #spin/m3
            
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
            
        if sample == '173Yb_Trensal':
            "SPIN VALUES (dimensionless)"
            S = 0.5; I = 2.5; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [2.935,2.935,4.29]; gi = -0.2592; # Mail Stergios (based on 172 Isotope)
            #ge = [2.935,2.935,4.35]
            "HYPERFINE VALUES (Hz)"
            Ap = - 6.15e8; Az = - 8.979e8 # Original 
            #Az = - 9.128e8
            #Ap = - 7.15e8
            #Ap = - 1.09436111e9; Az = - 0.82635179e9; # Hz - From P.E. Atkinson et al. PRA 100 042505 (2019)  
            "QUADRUPOLAR VALUE (Hz)"
            p = - 6.6e7 # Original
            #p = - 7.3e7 
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 0.0 ; kz = 0.0
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 0.16
            "SPIN DENSITY"
            density = 1.6941e27 #spin/m3
            
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
            
        if sample == 'DPPH':
            "SPIN VALUES (dimensionless)"
            S = 0.5; I = 0.5; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [2,2,2]; gi = 0.0;
            "HYPERFINE VALUES (Hz)"
            Ap = 0.0; Az = 0.0 
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 0.0 ; kz = 0.0
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 1.0
            "SPIN DENSITY"
            density = 1.6941e27 #spin/m3
            
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
            
        if sample == 'NV_centers':
            "SPIN VALUES (dimensionless)"
            S = 1; I = 1; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [2.002,2.002,2.002]; gi = 1.0;
            "HYPERFINE VALUES (Hz)"
            Ap = 0.0; Az = - 2.1e6 
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0
            "STRAIN TERM (Hz)"
            E = 2e6 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 2.878e9 ; kz = 0.0 
            
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 1.0
            "SPIN DENSITY"
            density = 1.6941e27 #spin/m3
            
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
        
        if sample == 'HoW10':
            "SPIN VALUES (dimensionless)"
            S = 8; I = 3.5; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [1.25,1.25,1.25]; gi = 0.0;
            "HYPERFINE VALUES (Hz)"
            Ap = 0.0; Az = 8.292972e8
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 0.0 ; kz = 0.0
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 1.0
            "SPIN DENSITY"
            density = 6.1335e26 #spin/m3
            
            kBoverh = 2.0836612351e10 # Hz/K 
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0.8653*kBoverh, 'B21': 0., 'B22': 0.,
                        'B40': 0.0100*kBoverh, 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.0045*kBoverh,
                        'B60': -0.000073*kBoverh, 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
        
        if sample == 'MnBr':
            "SPIN VALUES (dimensionless)"
            S = 2.5; I = 2.5; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI; # PRL 110, 027601 (2013)
            "ZEEMAN VALUES (dimensionless)"
            ge = [2,2,2.03]; gi = 1.3872;  # Talal Samples and TFM sebas. 3.468/2.5 ?
            "HYPERFINE VALUES (Hz)"
            #Ap = -225e6; Az = - 234e6   # Physics of the Solid State, Volume 52, Issue 3, pp.515-522
            #Ap = 0; Az = 186e6   # Mail Talal. and if it is Anisotropic
            Ap = 186e6; Az = 186e6   # Mail Talal. and if it is isotropic
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 4*4.67/30 #Hz/Vm^-1
            "ZERO FIELD SPLITTING (Hz) FOR STRAIN INTERACTION"
            D = 0.174*2.99728e10 ; kz = 4*0.7 #Hz / Vm^-1 // PRL 110, 027601 (2013)
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 1.0
            "SPIN DENSITY"
            density = 2.1e27 #spin/m^3
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
            
            
        if sample == 'VOporf':
            "SPIN VALUES (dimensionless)"
            S = 0.5; I = 3.5; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [1.9845,1.9845,1.963]; gi = 0.0;
            "HYPERFINE VALUES (Hz)"
            Ap = 151e6; Az = 475e6 # Original 
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0 
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 0.0 ; kz = 0.0
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 1.0
            "SPIN DENSITY"
            density = 1.6941e27 #spin/m3
            
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
       
        if sample == 'CNTporf':
            "SPIN VALUES (dimensionless)"
            S = 0.5; I = 3/2; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [2.055,2.055,2.19]; gi = 0.0; # Paper CuMint E.Burzurí)
            
            "HYPERFINE VALUES (Hz)"
            Ap = 40e6; Az = 620e6 # Original
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0 
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 0.0
            "ZERO FIELD SPLITTING (Hz)"
            D = 0.0 ; kz = 0.0
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 1.0
            "SPIN DENSITY"
            density = 1.6941e27 #spin/m3 - CHANGE IT
        
            "HIGH ORDER OPERATORS COEFICIENTS (4.2K)"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = 0. # Hz
            
        if sample == 'MnBr_Ising':
            "SPIN VALUES (dimensionless)"
            S = 2.5; I = 2.5; NS = int(2*S + 1); NI = int(2*I + 1); NJ = NS*NI;
            "ZEEMAN VALUES (dimensionless)"
            ge = [2.0,2.0,2.03]; gi = 1.3872; 
            "HYPERFINE VALUES (Hz)"
            #Ap = 0; Az = 0
            Ap = 186e6; Az = 186e6
            "QUADRUPOLAR VALUE (Hz)"
            p = 0.0 
            "STRAIN TERM (Hz)"
            E = 0.0 ; kp = 4*4.67/30 #Hz/Vm^-1
            
            "ZERO FIELD SPLITTING (Hz) FOR STRAIN INTERACTION"
            D = 0.174*2.99728e10 ; kz = 4*0.7 #Hz / Vm^-1 // PRL 110, 027601 (2013)
            
            "ISOTOPICAL CONCENTRATION (dimensionless)"
            ist_ab = 1.0
            "SPIN DENSITY"
            density = 2.1e27 #spin/m3 - CHANGE IT
        
            "HIGH ORDER OPERATORS COEFICIENTS"
            B_steven = {'B20': 0., 'B21': 0., 'B22': 0.,
                        'B40': 0., 'B41': 0., 'B42': 0., 'B43': 0., 'B44': 0.,
                        'B60': 0., 'B61': 0., 'B62': 0., 'B63': 0., 'B64': 0., 'B65': 0., 'B66': 0.}
            
            "EXCHANGE INTERACTION VALUE"
            J = -0.131*5.21652e9 # Hz
            
        "ELECTRONIC SPIN STATES VECTOR"
        Ms = np.arange(-1*S, S + 1, 1)
        "NUCLEAR SPIN STATES VECTOR"
        if I != 0:
            Mi = np.arange(-1*I, I + 1, 1)
        else:
            Mi = 0
            
        "SPIN MATRICES"
        Sx=qutip.jmat(S,'x'); Sy=qutip.jmat(S,'y'); Sz=qutip.jmat(S,'z')
        Ix=qutip.jmat(I,'x'); Iy=qutip.jmat(I,'y'); Iz=qutip.jmat(I,'z')
              
        return Sample(S,I,NS,NI,NJ,Ap,Az,p,Ix,Iy,Iz,Sx,Sy,Sz,ge,gi,E,kp,D,kz,Ms,Mi,ist_ab,density,B_steven,J)

""" ####################################################################### """

             
class qdata():
    
    "# qdata Class has all the information regarding to the Hamiltonian after diagonalization"

    def __init__(qdat, Havl, Have, E_NMR, Rabi, E_EPR, ME, f, Th_F):
        qdat.Havl = Havl
        qdat.Have = Have
        qdat.E_NMR = E_NMR
        qdat.Rabi = Rabi
        qdat.E_EPR = E_EPR
        qdat.ME = ME
        qdat.f = f
        qdat.Th_F = Th_F
        
        """#Havl are the eigenvalues of the hamiltonian.
        #Have are the eigenstates of the hamiltonian.
        #E_NMR are the frequencies of the NMR transitions.
        #E_EPR are the frequencies of the EPR transitions.
        #ME is the NJ x NJ matrix that contains the complex matrix elements for
        the operator T (user defined).
        #f is a float number or an array (depending on wether there is a given
        transition or not) that contains the frequency/ies of the given (or not)
        transition/s.
        #Th_F is the thermal factor for each pair of transitions"""
        
    def thermfactor(Havl,T):
        
        # Havl is the result of diagonalizing the Hamiltonian. It has NJ eigenvalues.
        # T is the temperature
        # G is the ground level for the calculation
        # E is the excited level for the calculation
    
        "PARTITION FUNCTION"
        
        # Havl has frequency units (Hz)
        K = 1.38064852e-23 # J/K
        E_T = K*T/6.6261e-34 # Hz
        
        # Partition vector Zi
        Zi = np.exp(-(Havl - np.amin(Havl))/E_T)
        
        # Sum of all partition elements
        Z = np.sum(Zi)
        
        "THERMAL FACTOR"
        P = Zi/Z # M - B distribution
        
        "BOSONIC OCCUPATION NUMBER"
        # Nbos = 1 + 1/((Zi(G)/Zi(E)-1)
                      
        # Th_f = P*Nbos
        Th_f = P
                      
        return Th_f
    
    def coupling(smpl, B_ext = np.array([0,0,0]), Temp = 0, T = 0, G = 0, E = 0, name = None):
    
        """# IF T,G AND E ARE GIVEN: This function calculates the matrix
        elements for a given operator T between the levels G (ground) and E
        (excited) for a molecule with electronic spinc S and nuclear spin I in
        a magnetic field B (Teslas). It gives back QData.ME, a 2 x 2 array
        that contains the four matrix elements for the given levels (G,E)
        in their complex form. It also gives back QData.GC, a 2 x 2 array that
        contains the modulus of the four matrix elements for the given levels
        (G,E). It also gives back QData.f, a float number that contains the
        frequency (in GHz) for the (G,E) transition. 
        
        # IF G AND E ARE CERO AND T IS GIVEN: This function calculates the
        matrix elements for all the transitions. It gives back QData.ME, a
        NJ x NJ array contains the matrix elements in their complex form for
        all of the possible transitions. It also gives back QData.GC, a
        NJ x NJ array contains the modulus of matrix elements for all of the
        possible transitions. It also gives back QData.f, a NJ x NJ array that
        contains the frequencies (in GHz) for all of the possible transitions.
        
        # IF T, G AND E ARE CERO: This function calculates E_NMR, a 8*I x 1
        array that contains the energy value for the all of the NMR transitions
        in the molecule. It also calculates E_EPR a NJ x 1 array that contains
        the energy value for all the EPR transitions in the molecule.
        
        # IN EVERY CASE, THE FUNCTION ALWAYS GIVES BACK QData.Havl, a vector
        of NJ x 1 dimensions the contains the eigenvalues of the 
        hamiltonian H."""
    
        # Here we are going to save the results
        E_NMR = np.zeros((smpl.NJ*smpl.NJ,3))
        #Rabi_NMR = np.zeros((Sample.NJ*Sample.NJ))
        E_EPR = np.zeros((smpl.NJ,3))
        c_nmr = 0 # NMR transitions counter
        
        """if Sample.J != 0:
            H = ising_hamiltonian(Sample, B)
        else:
            H = Sample.hamiltonian(Sample, B)"""
        
        H = Sample.hamiltonian(smpl, B_ext)
        
        # Eigenstates calculation
        HD = H.eigenstates()
        
        # HD has the eigenvalues in the first component (0) and the eigenvectors in the second component (1)
        Havl = HD[0] # Hz
        Have = HD[1] # No Units
        # No Units. Thermal factor P for each energy level
        
        if G == 0 and E == 0 and T == 0:
            
            Rabi = np.zeros((smpl.NJ,smpl.NJ) , dtype = float) # RABI FREQ/Brf FOR ALL TRANSITIONS
            ME = np.zeros((smpl.NJ,smpl.NJ) , dtype = complex)
            Th_F = 0
            f = 0
            
            if Temp != 0:
                Th_F = qdata.thermfactor(Havl,Temp)
            
            if name != 'HoW10':
                for i in range(smpl.NJ):              
                    for k in range(smpl.NJ):
                    
                        # NOT EPR TRANSITIONS
                        if (i != smpl.NJ - (k+1)):
                            if (k != smpl.NJ - (i+1)):
                                                                          
                                E_NMR[c_nmr,0] = Havl[i]-Havl[k] #E_i - E_k
                                E_NMR[c_nmr,1] = k # ground
                                E_NMR[c_nmr,2] = i # excited
                                c_nmr = c_nmr + 1
                    
                        # EPR TRANSITIONS
                        elif i == smpl.NJ - (k+1): 
                            E_EPR[k,0] = Havl[i]-Havl[k] #E_i - E_k
                            E_EPR[k,1] = k # ground
                            E_EPR[k,2] = i # excited
                    
                        # EPR TRANSITIONS
                        elif k == smpl.NJ - (i+1): 
                            E_EPR[k,0] = Havl[i]-Havl[k] #E_i - E_k
                            E_EPR[k,1] = k # ground
                            E_EPR[k,2] = i # excited
        
            # CLOCK TRANSITIONS IN PARTICULAR CASES
            if name == 'HoW10':
                E_EPR = np.zeros((1,3))
                E_EPR[0,0] = Havl[8]-Havl[7]
                E_EPR[0,1] = 7 # ground
                E_EPR[0,2] = 8 # excited
        
        # CALCULATES MATRIX ELEMENTS FOR OPERATOR T BETWEEN ALL LEVEL PAIRS
        # T in Hz/T
        elif G == 0 and E == 0 and T != 0:
                    
            ME = np.zeros((smpl.NJ,smpl.NJ) , dtype = complex) # MATRIX ELEMENTS IN COMPLEX FORM
            f = np.zeros((smpl.NJ,smpl.NJ) , dtype = float) # FREQUENCIES FOR ALL TRANSITIONS
            Th_F = np.zeros((smpl.NJ,smpl.NJ) , dtype = float) # THERMAL FACTOR FOR ALL TRANSITIONS    
            Rabi = np.zeros((smpl.NJ,smpl.NJ) , dtype = float) # RABI FREQ/Brf FOR ALL TRANSITIONS
            
            if Temp != 0:
                Th_f = qdata.thermfactor(Havl,Temp)
                
            for i in range(smpl.NJ): # Excited - rows
                for k in range(smpl.NJ): # Ground - columns
                    if i == k:
                        ME[i,k] = 0
                        f[i,k] = (Havl[i] - Havl[k]) # Hz
                        Th_F[i,k] = 0
                    else:
                        ME[i,k] = T.matrix_element((Have[i].unit()).dag() , Have[k].unit()) # Hz/T
                        f[i,k] = (Havl[i] - Havl[k]) # Hz
                        if Temp != 0:
                            Th_F[i,k] = abs(Th_f[i] - Th_f[k])
                        
                        # NMR TRANSITIONS
                        if i == k + 1 and k != int(smpl.NJ/2-1) and smpl.I != 0:
                            # i = 6 k = 5 is an EPR transition
                            # Only taking into account the allowed ones
                            E_NMR[c_nmr] = Havl[i]-Havl[k] # E_i - E_k
                            E_NMR[c_nmr,1] = k # ground
                            E_NMR[c_nmr,2] = i # excited
                            
                            c_nmr = c_nmr + 1
                    
                        # NMR TRANSITIONS
                        elif i == k - 1 and k != int(smpl.NJ/2) and smpl.I != 0 :
                            # i = 5 k = 6 is an EPR transition
                            # Only taking into account the allowed ones
                            E_NMR[c_nmr] = Havl[i]-Havl[k] # E_i - E_k
                            E_NMR[c_nmr,1] = k # ground
                            E_NMR[c_nmr,2] = i # excited
                            c_nmr = c_nmr + 1
                            
                        # EPR TRANSITIONS
                        elif i == smpl.NJ - (k+1): 
                            E_EPR[k,0] = Havl[i]-Havl[k] #E_i - E_k
                            E_EPR[k,1] = k # ground
                            E_EPR[k,2] = i # excited
                    
                        # EPR TRANSITIONS
                        elif k == smpl.NJ - (i+1): 
                            E_EPR[k,0] = Havl[i]-Havl[k] #E_i - E_k
                            E_EPR[k,1] = k # ground
                            E_EPR[k,2] = i # excited                        
                            
            Rabi = np.absolute(ME) # Hz/T
            
        else:
            
            ME = np.zeros((2,2) , dtype = complex)
            Th_F = np.zeros((2,2) , dtype = float)
            
            if Temp != 0:
                Th_f = qdata.thermfactor(Havl,Temp)
                
            f = 0
            
            ME[0,0] = T.matrix_element((Have[G].unit()).dag() , Have[G].unit())
            
            if Temp != 0:
                Th_F[0,0] = abs(Th_f[G] - Th_f[G])
                              
            ME[1,1] = T.matrix_element((Have[E].unit()).dag() , Have[E].unit())
            if Temp != 0:
                Th_F[1,1] = abs(Th_f[E] - Th_f[E])
            
            ME[0,1] = T.matrix_element((Have[E].unit()).dag() , Have[G].unit())
            if Temp != 0:
                Th_F[0,1] = abs(Th_f[E] - Th_f[G])
            
            ME[1,0] = np.conj(ME[0,1])
            Th_F[1,0] = Th_F[0,1]
            
            Rabi = np.absolute(ME)
            
            f = (Havl[E] - Havl[G])/(1e9) #GHz
                       
        return qdata(Havl,Have,E_NMR,Rabi,E_EPR,ME,f,Th_F)
                     
" ######################################################################### "

#%%
" #################### RESONATOR CURRENT DISTRIBUTION ###################### "

class Current():
    
    def __init__ (current, Ixrs, Iyrs, posx, posy, posz):
        
        # CURRENT INFORMATION
        
        current.Ixrs = Ixrs
        current.Iyrs = Iyrs
        current.posx = posx
        current.posy = posy
        current.posz = posz
        
    def mod_plot(current, cmap = 'RdBu_r', save = False, fmt = '.png', units = 'nA'):
    
        if save:
            dirpath = pth.folder()
            print('Saving images at %s' %dirpath)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            
        X = current.posx*1e6 # m to um
        Y = current.posy*1e6 # m to um
        
        if units == 'nA': scale = 1e9
        
        GR = np.sqrt(current.Ixrs**2+current.Iyrs**2)*scale
        
        plt.figure()        
        plt.pcolor(X, Y, GR,  cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ I  module \ XY \ plane' , fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Imap_XY_module_' + cmap + fmt, dpi = 1024, bbox_inches='tight')
        plt.show()
        
        return
    
    def vec_plot(current, cmap = 'RdBu_r', save = False, fmt = '.png', units = 'nA'):
    
        if save:
            dirpath = pth.folder()
            print('Saving images at %s' %dirpath)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            
        X = current.posx*1e6 # m to um
        Y = current.posy*1e6 # m to um
        
        if units == 'nA': scale = 1e9
        
        GR = current.Ixrs*scale
        
        plt.figure()        
        plt.pcolor(X, Y, GR, cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ I_{x} \ XY \ plane' , fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Imap_XY_Xcomponent_' + cmap + fmt, dpi = 1024, bbox_inches='tight')
        plt.show()
        
        GR = current.Iyrs*scale
        
        plt.figure()        
        plt.pcolor(X, Y, GR, cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ I_{y} \ XY \ plane' , fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Imap_XY_Ycomponent_' + cmap + fmt, dpi = 1024, bbox_inches='tight')
        plt.show()
        
        return

" ######################################################################### "

#%%
" #################### RESONATOR MAGNETIC FIELDS ############################ "

class Field():
    
    def __init__ (field, Brfx, Brfy, Brfz, Brf, posx, posy, posz, dx, dy, dz):
        
        # MAGNETIC FIELD INFORMATION
        
        field.Brfx = Brfx
        field.Brfy = Brfy
        field.Brfz = Brfz
        field.Brf = Brf
        field.posx = posx
        field.posy = posy
        field.posz = posz
        field.dx = dx
        field.dy = dy
        field.dz = dz
        
    """#Brfx is the Lx x Ly x Lz array that contains the x component of the Brf field. Loaded from a .mat file.
    #Brfy is the Lx x Ly x Lz array that contains the y component of the Brf field. Loaded from a .mat file.        
    #Brfz is the Lx x Ly x Lz array that contains the z component of the Brf field. Loaded from a .mat file.
    #posx is the array that contains spatial information along the x direction. Loaded from a .mat file.
    #posy is the array that contains spatial information along the y direction. Loaded from a .mat file.
    #posz is the array that contains spatial information along the z direction. Loaded from a .mat file.
    #dx is the discretization step in x direction. Calculated from posx.
    #dy is the discretization step in y direction. Calculated from posy.
    #dz is the discretization step in z direction. Calculated from posz.
    #xc is an int value that specifies the value for an YZ cut in the G_r(x,y,z) map. Loaded from a .mat file."""
    
    
    def mod_plot(field, zc=0, cmap = 'RdBu_r', save = False, fmt = '.png', units = 'nT'):
        
        if save:
            dirpath = pth.folder() + '/XY_cuts/'
            print('Saving images at %s' %dirpath)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            
        X = field.posx*1e6 # m to um
        Y = field.posy*1e6 # m to um
        
        if units == 'nT': scale = 1e9
        
        GR = field.Brf[:,:,zc]*scale
        
        plt.figure()        
        plt.pcolor(X, Y, GR, norm=colors.SymLogNorm(linthresh=10, vmin=np.min(GR), vmax=np.max(GR)) ,  cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ B \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_module_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        return
    
    def vec_plot(field, zc=0, cmap = 'RdBu_r', save = False, fmt = '.png', units = 'nT'):
        
        if save:
            dirpath = pth.folder() + '/XY_cuts/'
            print('Saving images at %s' %dirpath)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            
        X = field.posx*1e6 # m to um
        Y = field.posy*1e6 # m to um
        
        if units == 'nT': scale = 1e9        
        
        GRx = field.Brfx[:,:,zc]*scale
        GRy = field.Brfy[:,:,zc]*scale
        GRz = field.Brfz[:,:,zc]*scale
        
        plt.figure()        
        plt.pcolor(X, Y, GRx, norm=colors.SymLogNorm(linthresh=10, vmin=np.min(GRx), vmax=np.max(GRx)) ,  cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ B$_x$ \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_Xcomponent_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        plt.figure()        
        plt.pcolor(X, Y, GRy, norm=colors.SymLogNorm(linthresh=10, vmin=np.min(GRy), vmax=np.max(GRy)) ,  cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ B$_y$  \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_Ycomponent_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        plt.figure()        
        plt.pcolor(X, Y, GRz, norm=colors.SymLogNorm(linthresh=10, vmin=np.min(GRz), vmax=np.max(GRz)) ,  cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ B$_z$  \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_Zcomponent_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        
        return
    
    def xz_cut(field, yc, cmap = 'RdBu', save = True, fmt = '.png'):        
        
        "XZ CUTS"
        
        if save:
            dirpath = pth.folder() + '/XZ_cuts/'
            print('Saving images at %s' %dirpath)
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        
        Z = field.posz*1e6 # m to um
        X = field.posx*1e6 # m to um
        Y = field.posy*1e6 # m to um
        
        
        GR = field.Brf[:,:,0] 
        
        plt.figure()        
        plt.pcolor(X, Y, GR*1e9, norm=colors.SymLogNorm(linthresh=10, vmin=np.min(field.Brf[:,:,0]*1e9), vmax=np.max(field.Brf[:,:,0]*1e9)) ,  cmap=plt.get_cmap('RdBu_r'))
        plt.colorbar(label='B (nT)')
        plt.title('$ \ B \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[0]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        plt.axhline(field.posy[yc]*1e6, lw=2.0, color='white', linestyle='dashed')
        if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_yc%.2fum_' + cmap + fmt %(field.posy[yc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        GR = field.Brf[yc,:,:]
        
        plt.figure()
        plt.pcolor(X, Z, GR.transpose(), norm=colors.LogNorm(vmin=GR.min(), vmax=GR.max()) , cmap = cmap )
        plt.colorbar(label='T');
        plt.title('|B| XZ ycut = %.2e $\mu$m' %(field.posy[int(yc)]*1e6))
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('z ($\mu$m)' , fontsize = 14)
        plt.ylim((np.amin(Z),np.amax(Z)))
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Bmodule_map_XZslice_yc%.2fum_' + cmap + fmt %(field.posy[yc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        plt.figure()
        levels = np.geomspace(GR.min(),GR.max(),21)
        CS = plt.contour(X, Z, GR.transpose(), levels, norm=colors.LogNorm(vmin=GR.min(), vmax=GR.max()), cmap = cmap)
        plt.clabel(CS, inline = False, fontsize = 0)
        plt.colorbar(label='T');
        plt.title('|B| XZ ycut = %.2e $\mu$m' %(field.posy[int(yc)]*1e6))
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('z ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/B_contour_XZslice_yc%.2fum_' + cmap + fmt %(field.posy[yc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
    
        return
    

    def yz_cut(field, xc, cmap = 'RdBu', save = True,  fmt = '.png'):        
    
        "YZ CUTS"
        
        if save:
            dirpath = pth.folder() + '/YZ_cuts/'
            print('Saving images at %s' %dirpath)
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        Z = field.posz*1e6 # m to um
        X = field.posx*1e6 # m to um
        Y = field.posy*1e6 # m to um
        
        
        GR = field.Brf[:,:,0] 
        
        plt.figure()        
        plt.pcolor(X, Y, GR*1e9, norm=colors.SymLogNorm(linthresh=10, vmin=np.min(field.Brf[:,:,0]*1e9), vmax=np.max(field.Brf[:,:,0]*1e9)) ,  cmap=plt.get_cmap('RdBu_r'))
        plt.colorbar(label='B (nT)')
        plt.title('$ \ B \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[0]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        plt.axvline(field.posx[xc]*1e6, lw=2.0, color='white', linestyle='dashed')
        plt.savefig(dirpath + '/Bmodule_map_XYslice_xc%.2fum_' + cmap + fmt %(field.posx[xc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        GR = field.Brf[:,xc,:]
        
        plt.figure()
        plt.pcolor(Y, Z, GR.transpose(), norm=colors.LogNorm(vmin=GR.min(), vmax=GR.max()) , cmap = cmap )
        plt.colorbar(label='T');
        plt.title('|B| YZ xcut = %.2e $\mu$m' %(field.posx[int(xc)]*1e6))
        plt.xlabel('y ($\mu$m)',fontsize = 14); plt.ylabel('z ($\mu$m)' , fontsize = 14)
        plt.ylim((np.amin(Z),np.amax(Z)))
        plt.tick_params(direction = 'in')
        plt.savefig(dirpath + '/Bmodule_map_YZslice_xc%.2fum_' + cmap + fmt %(field.posx[xc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        plt.figure()
        levels = np.geomspace(GR.min(),GR.max(),21)
        #levels=[1.0e-9, 5.0e-9, 1.0e-8, 5.0e-8, 1.0e-7, 5.0e-6, 1.0e-5, 5.0e-5, 1.0e-4]
        CS = plt.contour(Y, Z, GR.transpose(), levels, norm=colors.LogNorm(vmin=GR.min(), vmax=GR.max()), cmap = cmap)
        plt.clabel(CS, inline = False, fontsize = 0)
        plt.colorbar(label='T');
        plt.title('|B| YZ xcut = %.2e $\mu$m' %(field.posx[int(xc)]*1e6))
        plt.xlabel('y ($\mu$m)',fontsize = 14); plt.ylabel('z ($\mu$m)' , fontsize = 14)
        #plt.ylim(0.,50.)
        plt.tick_params(direction = 'in')
        plt.savefig(dirpath + '/B_contour_YZslice_xc%.2fum_' + cmap + fmt %(field.posx[xc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
    
        return

"#########################################################################"
#%%

class File():
    def __init__ (file, H2DX, H2DY, H2DZ, H2DM, posx, posy, posz, dx, dy, dz, xc):
        file.H2DX = H2DX
        file.H2DY = H2DY
        file.H2DZ = H2DZ
        file.H2DM = H2DM
        file.posx = posx
        file.posy = posy
        file.posz = posz
        file.dx = dx
        file.dy = dy
        file.dz = dz
        file.xc = xc  

        
class GData():        
    def __init__ (gdata, G_r, g_0, g_N, g_bar, delta_E, Th_f):   
        
        # MAIN RESULTS OF gfactor FUNCTION
        
        gdata.G_r = G_r # SPACE DEPENDENT COUPLING FACTOR MATRIX
        gdata.g_0 = g_0 # COUPLING FACTOR PER SPIN (NON SPACE DEPENDENT)
        gdata.g_N = g_N # COUPLING FACTOR (NON SPACE DEPENDENT)
        gdata.g_bar = g_bar # PROMEDIUM COUPLING FACTOR PER SPIN (NO SPACE DEPENDENT)
        gdata.delta_E = delta_E # EIGENVALUES DERIVED FROM DIAGONALIZATION IN qf.gfactor FUNCION
        gdata.Th_f = Th_f # THERMAL POLARIZATION VECTOR
                           
#G_r contains the spatial dependent values of the coupling factor for a given transition (G,E) and a magnetic field
    #distribution.
#g_0 is the integrated value of the G_r(x,y,z) elements over the entire volume without taking account the spin number N.
#g_N is the integrated value of the G_r(x,y,z) over the entire given volume, taking into account the spin density (spin/m3)

class Data_exp():
    
# QData Class has all the information regarding to the Hamiltonian after diagonalization

    def __init__(data_exp, TR, omega, Bplot):
        data_exp.TR = TR
        data_exp.omega = omega
        data_exp.Bplot = Bplot
        
# TR stores de transmission data (normalized to desired line)
# omega stores frequency data
# Bplot stores field data (normalized to desired line)


class Data_adjust():
    
# QData Class has all the information regarding to the Hamiltonian after diagonalization

    def __init__(data_adjust, TR, TRaprox, params, B, F):
        data_adjust.TR = TR
        data_adjust.TRaprox = TRaprox
        data_adjust.params = params
        data_adjust.B = B
        data_adjust.F = F
        
# TR stores the transmission data (normalized to desired line)
# TRaprox stores the transmission data calculated using adjusted parameters
# paras stores the adjustes parameters
# omega stores frequency data
# Bplot stores field data (normalized to desired line)

class Data_clean():
    
# Data_clean class stores Freq vs Field maps after cleaning discontinuities (if necessary)

    def __init__(data_clean, TR, omega, Bplot):
        data_clean.TR = TR
        data_clean.omega = omega
        data_clean.Bplot = Bplot
        
# TR stores the transmission cleaned data 
# omega stores frequency cleaned data
# Bplot stores field data cleaned 

class Data_clean():
    
# Data_clean class stores Freq vs Field maps after cleaning discontinuities (if necessary)

    def __init__(data_clean, TR, omega, Bplot):
        data_clean.TR = TR
        data_clean.omega = omega
        data_clean.Bplot = Bplot
        
# TR stores the transmission cleaned data 
# omega stores frequency cleaned data
# Bplot stores field data cleaned

class FvF():
    
# Class for Resonance Frequency vs Field data

    def __init__(fvf, fres, bres):
        fvf.fres = fres
        fvf.bres = bres

# fres is a one-dimensional array which stores fres value
# bres is a one-dimensional array which stores fres value
        
#########################################################################################################################

""" ##################### LOAD FILES ##################################### """

class pth:
    "Class that contains the functions to import and manipulate paths"


    @classmethod
    def file(self):
        """
        Function that loads a single file path.
        
        Returns
        -------
        path: str
            Path of the file
        file: str
            Name of the file
        directory: str 
            Directory of the file
        """
 
        
        try:
            root=tk.Tk() # It helps to display the root window
            root.withdraw() # Hide a small window openned by tkinter
            root.attributes("-topmost", True)
            path = tk.filedialog.askopenfilename(parent=root) # Shows dialog box and return the path of the file
            root.destroy()
            file = os.path.basename(path) # Get the name of the file
            dirpath = os.path.dirname(path) # Get the directory path of the file
            
            return path, file, dirpath

        except:
            raise Exception('the path cannot be imported')
            return None    

    @classmethod
    def folder(self):
        """
        Function that loads a single folder path.
        
        Returns
        -------
        path: str
            Path of the folder
        """
 
        
        try:
            root=tk.Tk() # It helps to display the root window
            root.withdraw() # Hide a small window openned by tkinter
            root.attributes("-topmost", True)
            path = tk.filedialog.askdirectory(parent=root) # Shows dialog box and return the path of the file
            root.destroy()
            
            return path

        except:
            raise Exception('the path cannot be imported')
            return None
        

    @classmethod
    def files(self):
        """
        Function that loads several selected file paths.
        
        Returns
        -------
        path: string
            Path of the files
        file: string
            Name of the files
        directory: string
            Directory of the files
        """

        try:        
            root=tk.Tk() # It helps to display the root window
            root.withdraw() # Hide a small window openned by tkinter
            root.attributes("-topmost", True)
            path = tk.filedialog.askopenfilenames(title='Select multiple files') # Shows dialog box and return the path of the file
            files = [os.path.basename(f) for f in path if '.ini' not in f] # Get a list of the name of the files
            files = natsorted(files) # Sorted the files naturally (in case it contains numbers)
            dirpath = os.path.dirname(path[0]) # Get the directoy path of the files
            
            return list(path), list(files), dirpath

        except:
            raise Exception('the paths cannot be imported')
            return None    
    
    
    
    @classmethod  
    def dirfiles(self, name=None, ext = None):
        """Function that reads all the files in the selected folder, files can be filtered by name and extension.
        
        Parameters
        ----------
        
        name: string, default None
            Name of the file to be filtered
        ext: string, default None
            Extension of the file to be filtered
            
        Returns
        -------
        path: str
            Path of the file
        file: str
            Name of the file
        directory: str 
            Directory of the file
        """
        
        try:
            root=tk.Tk() # It helps to display the root window
            root.withdraw() # Hide a small window openned by tkinter
            dirpath = tk.filedialog.askdirectory(title='Select directory') # Shows dialog box and return the path of the file   
            
            if name == None and ext == None: # If name and extension not used, import every file
                files = [f for f in os.listdir(dirpath) if '.ini' not in f]
                files = natsorted(files)   
                
            elif name == None and ext != None: # Import files with the extension used
                files = [f for f in os.listdir(dirpath) if '.ini' not in f and ext in f]
                files = natsorted(files)
                
            elif name != None and ext == None:# Import files with the name used
                files = [f for f in os.listdir(dirpath) if name in f and '.ini' not in f]
                files = natsorted(files)
                
            else: # Import files with the extension and name
                files = [f for f in os.listdir(dirpath) if name in f and '.ini' not in f and ext in f]
                files = natsorted(files) 
                   
            path = []
            for i in files: # Append every file path
                path.append(os.path.join(dirpath, i))
        
    
            return path, files, dirpath 
    
        except:
            raise Exception('the files from the directory cannot be imported')
            return None



    @classmethod 
    def resultsfolder(self, dirpath, name = None): 
        """
        Function that checks if a the folder exists, if not, creates a folder called 'Output'.
        
        Parameters
        ----------
        
        dirpath: string
            Full path where the folder will be created
        name: string, default None
            Name of the created folder, if None, name = Output
            
        Returns
        -------
        fdir: string
            Full path of the folder
        """
        
        try:
            if name == None:
                fdir = os.path.join(dirpath,'Output') # Get the path with the name of the file without the extension
            
            else:
                fdir = os.path.join(dirpath, name) # Get the path with the name of the file without the extension        
            
            if os.path.exists(fdir): # Check if the folder already exists
                None
            else: # If not, then creates the folder
                os.mkdir(fdir) # Creates the new folder in the specified path     
            return fdir
    
        except:
            raise Exception('not possible to create the results folder')
            return None



    @classmethod 
    def renamefiles(self, original, replace): # Useful to replace a lot of file names
        """
        Function that renames the selected files.
        
        Parameters
        ----------
        
        original: string
            Part of the name to change
        replace: string
            New name part replacing the original
            
        Returns
        -------
        None
        """
        
        path, files, dirpath = self.files() # Uses the file function
        try:
            for index, file in enumerate(files): # Loop for rename each file with the new name
                os.rename(os.path.join(dirpath,file), os.path.join(dirpath, file.replace(original, replace)))
        except:
            raise Exception('file names cannot be replaced')
            
###############################################################################

################################################### FUNCTIONS ########################################################### 
    
  
def loadfield(path, angles = {'angle': 0., 'fi': 0., 'theta': 0.}):
    
    # This program reads the stored magnetic field matrices generated by the GPU
    "LOADING THE FILE"
    
    # LOAD MATRICES FROM .MAT FILE
    Aux = sio.loadmat(path)
    Aux = Aux['field'] #T and m
        
    H2DZ = Aux['H2DX'][(0)][(0)] #T
    H2DY = Aux['H2DY'][(0)][(0)] #T
    H2DX = -1*Aux['H2DZ'][(0)][(0)] #T
    
    """if os.path.isfile(path):
        
        # LOAD MATRICES FROM .MAT FILE
        Aux = sio.loadmat(path + '/field.mat')
        Aux = Aux['field1'] #T and m
        
        H2DZ = Aux['H2DX'][(0)][(0)] #T
        H2DY = Aux['H2DY'][(0)][(0)] #T
        H2DX = -1*Aux['H2DZ'][(0)][(0)] #T
        
    elif os.path.isfile(path + '/fieldx.mat') and os.path.isfile(path + '/fieldy.mat') and os.path.isfile(path + '/fieldz.mat'):
    
        # LOAD MATRICES FROM .MAT FILES (SEPARATE)
        Aux = sio.loadmat(path + '/fieldx.mat')
        Aux = Aux['fieldx']
        H2DZ = Aux['H2DX'][(0)][(0)] #T
        
        Aux = sio.loadmat(path + '/fieldy.mat')
        Aux = Aux['fieldy']
        H2DY = Aux['H2DY'][(0)][(0)] #T
        
        Aux = sio.loadmat(path + '/fieldz.mat')
        Aux = Aux['fieldz']
        H2DX = -1*Aux['H2DZ'][(0)][(0)] #T"""
        
          
    "ROTATING THE Brf FIELD"
    
    fi = angles['fi']*math.pi/180
    theta = angles['theta']*math.pi/180
    
    # Molecule axes in lab coordinates
    Z_mol = np.array([math.sin(theta), math.cos(theta)*math.sin(fi), math.cos(theta)*math.cos(fi)])
    X_mol = np.cross(np.array([0,1,0]),Z_mol)
    X_mol = X_mol/np.linalg.norm(X_mol)
    Y_mol = np.cross(X_mol,Z_mol)
    
    # Base change matrix:
    M = np.zeros((3,3))
    M[:,0] = X_mol
    M[:,1] = Y_mol
    M[:,2] = Z_mol
    M = np.linalg.inv(M)
    
    "NEW COORDINATE MATRICES"
    Brfx = M[0,0]*H2DX + M[0,1]*H2DY + M[0,2]*H2DZ
    Brfy = M[1,0]*H2DX + M[1,1]*H2DY + M[1,2]*H2DZ
    Brfz = M[2,0]*H2DX + M[2,1]*H2DY + M[2,2]*H2DZ
    
    "LOADING SPATIAL PARAMETERS"
    posx = Aux['posx'][(0)][()][0][0]; # m
    posy = Aux['posy'][(0)][()][0][0]; # m
    posz = Aux['posz'][(0)][()][0][0]; # m

    dx = np.absolute(posx[1] - posx[0])#m
    dy = np.absolute(posy[1] - posy[0])#m
    dz = np.absolute(posz[1] - posz[0])#m
    
    "Brf MODULE CALCULATION"
    Brf = np.sqrt(np.multiply(Brfx,Brfx) + np.multiply(Brfy,Brfy) + np.multiply(Brfz,Brfz))
        
        
    return Field(Brfx,Brfy,Brfz,Brf,posx,posy,posz,dx,dy,dz)

def base_change(B,angles):
     
    # Changing from deg to rad
    fi = angles['fi']*math.pi/180
    theta = (angles['theta'] + angles['angle'])*math.pi/180
    
    # Molecule axes in lab coordinates
    Z_mol = np.array([math.sin(theta), math.cos(theta)*math.sin(fi), math.cos(theta)*math.cos(fi)])
    X_mol = np.cross(np.array([0,1,0]),Z_mol)
    X_mol = X_mol/np.linalg.norm(X_mol)
    Y_mol = np.cross(X_mol,Z_mol)
        
    # Base change matrix:
    M = np.zeros((3,3))
    M[:,0] = X_mol
    M[:,1] = Y_mol
    M[:,2] = Z_mol
    M = np.linalg.inv(M)
    
    return M.dot(B)

def loadcurrent(path):
    
    # This program reads the electric current matrices generated by Sonnet

    Aux = h5py.File(path,'r')
    Aux = Aux['Ifrs']
    #Aux = Aux[name]
    
    "LOADING Irs CURRENT MATRICES"
    Ixrs = np.float32(np.transpose(Aux['Ixrs'][()]))
    Iyrs = np.float32(np.transpose(Aux['Iyrs'][()]))
    
    "LOADING SPATIAL PARAMETERS"
    posx = np.float32(Aux['posx'][()][0])
    posy = np.float32(Aux['posy'][()][0])
    #posz = Aux['posz'][()][0]
    #posz = Aux['posz'][()][:,0]
    posz = 0.0
    
    return Current(Ixrs,Iyrs,posx,posy,posz)    

def gfactor(sample,B_ext,Temp,G,E,Field=[],M=[],u=np.array([1,0,0]), electric = False):
    
    # This function calculates the colective coupling of an ensemble on in a given magnetic field map PREVIOUSLY CALCULATED 
    # created by a Lumped Element Resonator.
    # Sample is a struct/class that carries sample information
    # B is the external magnetic field applied to the system (zeeman)
    # G is the ground state for the matrix elements
    # E is the excited state for the matrix elements
    # Field is the rf magnetic field previously calculated
    # density is the spin density in the crystal (WORK TO DO, PUT IT WITHIN SAMPLE CLASS)
    # mode is the mode calculation, GPU or CPU (WORK TO DO, REMOVE GPU MODE)
     
    H = Sample.hamiltonian(sample, B_ext)
    HD = H.eigenstates()
    Havl = HD[0]
    Have = HD[1]
    Th_F = thermfactor(Havl, Temp)
   
   
    if M != [] and Field != []:
        
        H2DX = Field.Brfx; H2DY = Field.Brfy; H2DZ = Field.Brfz;
                
        Field.Brfx = M[0,0]*H2DX + M[0,1]*H2DY + M[0,2]*H2DZ
        Field.Brfy = M[1,0]*H2DX + M[1,1]*H2DY + M[1,2]*H2DZ
        Field.Brfz = M[2,0]*H2DX + M[2,1]*H2DY + M[2,2]*H2DZ
        
    elif M != [] and Field == []:
        
        # Field unitary direction: although we are calculating g in Hz/T we need to stablish a magnetic field direction.
        #This direction is unitary
        # RF unitary vector in laboratory coord sys
        u = M.dot(u) # Lab coord sys to mol coord sys
        u = u/np.linalg.norm(u)
        
    
    if electric == True:
        T0 = sample.kp*qutip.tensor(sample.Sx,qutip.qeye(int(2*sample.I+1)))**2
        T1 = -1*sample.kp*qutip.tensor(sample.Sy,qutip.qeye(int(2*sample.I+1)))**2
        T2 = sample.kz*qutip.tensor(sample.Sz,qutip.qeye(int(2*sample.I+1)))**2
        
    else:
        #T: OPERATOR FOR CALCULATING MATRIX ELEMENTS (COUPLING)
        T0 = sample.ge[0]*(9.27401e-24 / 6.6261e-34)*qutip.tensor(sample.Sx,qutip.qeye(int(2*sample.I+1))) + sample.gi*(5.05078E-27 / 6.6261e-34)*qutip.tensor(qutip.qeye(int(2*sample.S+1)),sample.Ix) #Hz/T
        T1 = sample.ge[1]*(9.27401e-24 / 6.6261e-34)*qutip.tensor(sample.Sy,qutip.qeye(int(2*sample.I+1))) + sample.gi*(5.05078E-27 / 6.6261e-34)*qutip.tensor(qutip.qeye(int(2*sample.S+1)),sample.Iy) #Hz/T
        T2 = sample.ge[2]*(9.27401e-24 / 6.6261e-34)*qutip.tensor(sample.Sz,qutip.qeye(int(2*sample.I+1))) + sample.gi*(5.05078E-27 / 6.6261e-34)*qutip.tensor(qutip.qeye(int(2*sample.S+1)),sample.Iz) #Hz/T
    
    me0 = T0.matrix_element((Have[E].unit()).dag() , Have[G].unit())
    me1 = T1.matrix_element((Have[E].unit()).dag() , Have[G].unit())
    me2 = T2.matrix_element((Have[E].unit()).dag() , Have[G].unit())   
    
    
    
    if Field == []:
        
        Gr_x = u[0]*np.absolute(me0)
        Gr_y = u[1]*np.absolute(me1)
        Gr_z = u[2]*np.absolute(me2)
        
        G_r = np.sqrt(np.multiply(Gr_x,Gr_x) + np.multiply(Gr_y,Gr_y) + np.multiply(Gr_z,Gr_z))
        
        # Thermal factor
        Th_f = Th_F[G] - Th_F[E] # Dimensionless
        
        g_N = np.sqrt(np.sum(np.multiply(G_r,G_r)))*np.sqrt(abs(Th_f))
        g_bar = 0
        g_0 = 0
        
        
    else:
        
        Gr_x = Field.Brfx*np.absolute(me0)
        Gr_y = Field.Brfy*np.absolute(me1)
        Gr_z = Field.Brfz*np.absolute(me2)
    
        G_r = np.sqrt(np.multiply(Gr_x,Gr_x) + np.multiply(Gr_y,Gr_y) + np.multiply(Gr_z,Gr_z))
    
        NX = G_r.shape[0]
        NY = G_r.shape[1]
        NZ = G_r.shape[2]
    
        # Thermal factor
        Th_f = Th_F[G] - Th_F[E] # Dimensionless 
        
        G_0 = G_r*(np.sqrt(abs(Th_f)*Sample.ist_ab*Sample.density*Field.dx*Field.dy*Field.dz))
        g_0 = 0
        g_N = np.sqrt(np.sum(np.multiply(G_0,G_0))) 
        g_bar = g_0/math.sqrt(NX*NY*NZ)
    
    # Transitiom frequencies
    delta_E = Havl[E] - Havl[G] # Hz  
                
    return GData(G_r,g_0,g_N,g_bar,delta_E,Th_f)


""" ####################### CUDA CODES ################################### """
kernel_code_template = """

    #include<math.h>
    
    __global__ void ArrayDotCtKernel(float *A, float b, float *C){
    
    //__global__ indicates function that runs on 
    //device (GPU) and is called from host (CPU) code

        unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
        unsigned idy = threadIdx.y + blockDim.y*blockIdx.y;
        unsigned idz = threadIdx.z + blockDim.z*blockIdx.z;
        
        int X = %(DimX)s;
        int Y = %(DimY)s;
        int Z = %(DimZ)s;
                       
        if ( ( idx < X) && (idy < Y) && ( idz < Z) ){
           C[Z*Y*idx + Z*idy + idz] = A[Z*Y*idx + Z*idy + idz]*b;
        }
        
        __syncthreads();
    }
"""  
    
def gpu_arraydotct(A,b):
    
    DimZ = A.shape[2]
    DimY = A.shape[1]
    DimX = A.shape[0]
    
    A_gpu = gpuarray.to_gpu(A) 
    #b_gpu = gpuarray.to_gpu(b)
    
    # create empty gpu array for the result (C = A * b)
    C_gpu = gpuarray.empty((DimX, DimY, DimZ), np.float32)

    # get the kernel code from the template 
    # by specifying the constants DimX, DimY, DimZ
    kernel_code = kernel_code_template %{
        'DimX': DimX, 'DimY': DimY, 'DimZ': DimZ 
    }

    # compile the kernel code 
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    ArrayDotCte = mod.get_function("ArrayDotCtKernel")
    
    "DEFINING BLOCK SIZE"
    bdim = (8,8,8) # 512 threads per block
    
    "DEFINING GRID SIZE"
    dx, mx = divmod(DimX, bdim[0])
    dy, my = divmod(DimY, bdim[1])
    dz, mz = divmod(DimZ, bdim[2])
    g_x = dx + (mx>0)
    g_y = dy + (my>0)
    g_z = dz + (mz>0)
    gdim = (g_x, g_y, g_z) #Grid size
    
    ArrayDotCte(A_gpu,b,C_gpu, block = bdim, grid=gdim)
    
    A_gpu.gpudata.free()
    
    C = C_gpu.get()
    C_gpu.gpudata.free()
    
    
    return C

def factorizehalf(x):
    factors = np.array([1])
    for i in range(2, int(x)):
        if (x % i) == 0:
            factors = np.append(factors, np.array([i]), axis = 0)
    
    factor = factors[int(np.size(factors)/2)]
    
    return factor

def getDeviceSMs():
    
    var_name = cuda.device_attribute.MULTIPROCESSOR_COUNT
    SMs = cuda.Device(0).get_attribute(var_name)
            
    return(SMs)

def getDeviceGridSize():
    
    var_name = cuda.device_attribute.MAX_GRID_DIM_X
    GridSize_x = cuda.Device(0).get_attribute(var_name)
            
    return(GridSize_x)

mod = compiler.SourceModule("""

    #include<math.h>
    #include <stdio.h>
    
    __global__ void Current2Field(float *Ixrs, float *Iyrs, float *H2D, float *posx, float *posy, float *posz, int X, int Y, int Z){
    
    //REMEMBER: this function is executed at each threat(thread?) at the same time!!
    //__global__ indicates function that runs on device (GPU) and is called from host (CPU) code

        int idx = threadIdx.x + blockDim.x*blockIdx.x;
        int idy = threadIdx.y + blockDim.y*blockIdx.y;
        int idz = threadIdx.z + blockDim.z*blockIdx.z;
        
        // If H1 H2 H3 have no pointer they are not accesible by the other threads.  
    
        float H1 = 0;
        float H2 = 0;
        float H3 = 0;
        
        float dx;
        float dy;
        float dz;
        if ((idx < X) && (idy < Y) && (idz < Z)){
            dz = - posz[idx + X*idy + X*Y*idz];

        }
        float modr;
        float imodr;
        float power2 = 2;
        float power3 = 3;
               
        int i = 0;
        
        // This loop is executed in all threads at the same time.
        
        for (i = 0 ; i < X*Y ; ++i){
            
            dx = posx[idx + X*idy] - posx[i];
            dy = posy[idx + X*idy] - posy[i]; 
            
            modr = sqrt(pow(dx,power2) + pow(dy,power2) + pow(dz,power2));
            imodr = 1/modr;
            
            H1 += Iyrs[i]*dz*pow(imodr, power3);
            H2 += - Ixrs[i]*dz*pow(imodr, power3);
            H3 += (Ixrs[i]*(dy) - Iyrs[i]*(dx))*pow(imodr, power3);
            
        }
        
        //Sync to ensure that every aux value is fully calculated at each threat before continue
        __syncthreads();
        
        if ( ( idx < X) && (idy < Y) && ( idz < Z) ){
            H2D[idx + X*idy + X*Y*idz] = H1;
        } else if ( (idx < X) && (idy < Y) && (idz >= Z) && (idz < 2*Z) ){
            H2D[idx + X*idy + X*Y*idz] = H2;
        } else if ( (idx < X) && (idy < Y) && (idz >= 2*Z) && (idz < 3*Z) ){
            H2D[idx + X*idy + X*Y*idz] = H3;
        }
        __syncthreads();
    }
    
    __global__ void SegCurrent2Field(float *Ixrs, int DX, float *Iyrs, int DY, float *H2D, float *xi, float *yi, float *posx, float *posy, float *posz, int *loop, int X, int Y, int Z){                                             
    
    //REMEMBER: this function is executed at each GPU thread at the same time!!
    //__global__ indicates function that runs on device (GPU) and is called from host (CPU) code

        // GPU thread identifiers 
        int idx = threadIdx.x + blockDim.x*blockIdx.x;
        int idy = threadIdx.y + blockDim.y*blockIdx.y;
        int idz = threadIdx.z + blockDim.z*blockIdx.z;
        
        // If E1 E2 E3 have no memory pointer (*) they are not accesible by the other threads.
        // E1 is for Erfx calculation, E2 is for Erfy calculation and E3 is for Erfz calculation
        float H1 = 0;
        float H2 = 0;
        float H3 = 0;
        
        // If dx dy dz have no memory pointer (*) they are not accesible by the other threads.
        // dx, dy, dz are de distance (dx,dy,dz) from the pixel in which the program is making
        // the calculation to the pixel that is taking for the calculation at each iteration.
        float dx;
        float dy;
        float dz;
        
        // Variables with no memory pointer (*) do not depend on the thread
        float modr; // (dx,dy,dz) modulus
        float imodr; // (dx,dy,dz) modulus^-1
        float power2 = 2; // cte
        float power3 = 3; // cte
        // TO DO LATER: replace power2 and power3 by constants. Reduces memory and time.  
        
        // This loop is executed in all threads at the same time.
        int i = 0; 
                     
        if ((idx < X) && (idy < Y) && (idz < Z)){
            // dz = - posz[idx + X*idy + X*Y*idz]; (Wrong?)          
            dz = posz[idx + X*idy + X*Y*idz]; // Since zi[i] = 0 for all sources (all sources on z = 0)
            
            for (i = 0 ; i < DX*DY ; ++i){
                
                dx = posx[idx + X*idy + X*Y*idz] - xi[i];
                dy = posy[idx + X*idy + X*Y*idz] - yi[i]; 
            
                modr = sqrt(pow(dx,power2) + pow(dy,power2) + pow(dz,power2));
                imodr = 1/modr;
            
                H1 += Iyrs[i]*dz*pow(imodr, power3);             
                H2 += - Ixrs[i]*dz*pow(imodr, power3);
                H3 += (Ixrs[i]*(dy) - Iyrs[i]*(dx))*pow(imodr, power3);
                // In python, multiply constants!
            }
        }
        
        //Sync to ensure that every aux value (E1 E2 E3) is fully calculated at each threat before continue
        __syncthreads();
        
        if ( loop[idx + X*idy + X*Y*idz] == 0 ){
            H2D[idx + X*idy + X*Y*idz] = H1;
        } else if ( loop[idx + X*idy + X*Y*idz] == 1  ){
            H2D[idx + X*idy + X*Y*idz] = H2;
        } else if ( loop[idx + X*idy + X*Y*idz] == 2  ){
            H2D[idx + X*idy + X*Y*idz] = H3;
        }
        __syncthreads();
    }
    """)

######################################################################################

def gpu_fieldv2(Ixrs,Iyrs,posx,posy,posz):
    
    DimX = (posx.shape[0])
    DimY = (posy.shape[0])
    DimZ = (posz.shape[0])
    Dim = np.int64(DimX*DimY*3*DimZ)
    
    step = np.float32(np.absolute(posx[1] - posx[0])) # m
    
    X , Y = np.meshgrid(posx, posy)
    
        
    "FLATTENING CURRENT AND GEOMETRIC ARRAYS"
    
    print('Flattening matrices . . .')
    
    # First we need to creat flat arrays to store the 2D geometric (X,Y) and current (Ixrs, Iyrs) arrays:
    Ixrs_flat = np.zeros((DimX*DimY), dtype = 'float32')
    Iyrs_flat = np.zeros((DimX*DimY), dtype = 'float32')
    X_flat = np.zeros((DimX*DimY), dtype = 'float32')
    Y_flat = np.zeros((DimX*DimY), dtype = 'float32')  
    
    # Now we flat the 2D geometric and current arrays:
    # remember!! i: rows, j: columns
    for j in range(DimX):
        for i in range(DimY):
            Ixrs_flat[i*DimX + j] = Ixrs[i,j]
            Iyrs_flat[i*DimX + j] = Iyrs[i,j]
            X_flat[i*DimX + j] = X[i,j]
            Y_flat[i*DimX + j] = Y[i,j]
    
    # Finally we transfer the flat geometric and current arrays to the GPU:
    print('Transferring geometric and current data to gpu . . .')
    Ixrs_gpu = gpuarray.to_gpu(Ixrs_flat)
    Iyrs_gpu = gpuarray.to_gpu(Iyrs_flat)
    X_gpu = gpuarray.to_gpu(X_flat)
    Y_gpu = gpuarray.to_gpu(Y_flat)        
            
    "SOLUTION ARRAY AND AUX ARRAYS CREATION"
    
    print('Creating void 3D array to store solution . . .')
    
    H2D = np.zeros(Dim, dtype = 'float32')
    # H2D is a flat array that will store the results of the calculation. 
    # In order to store the three magnetic field arrays (Brfx, Brfy, Brfz) it has a size that is three times DimX*DimY*DimZ.
    # The first DimX*DimY*DimZ elements correspond to Brfx, the elements between DimX*DimY*DimZ and 2*DimX*DimY*DimZ correspond
    # to Brfy and the last DimX*DimY*DimZ elements correspond to Brfz
   
        
    # posz3d stores the z coordinate corresponding to each element in H2D, in order to simplify the logic in the CUDA program
    posz3d = np.zeros(Dim, dtype = 'float32')
    posx3d = np.zeros(Dim, dtype = 'float32')
    posy3d = np.zeros(Dim, dtype = 'float32')
        
    for i in range(DimZ):
        posz3d[i*DimX*DimY :(i+1)*DimX*DimY] = posz[i]
        posz3d[DimZ*DimX*DimY + i*DimX*DimY : DimZ*DimX*DimY + (i+1)*DimX*DimY] = posz[i]
        posz3d[2*DimZ*DimX*DimY + i*DimX*DimY : 2*DimZ*DimX*DimY + (i+1)*DimX*DimY] = posz[i]
        posx3d[i*DimX*DimY : (i+1)*DimX*DimY] = X_flat
        posx3d[DimZ*DimX*DimY + i*DimX*DimY : DimZ*DimX*DimY + (i+1)*DimX*DimY] = X_flat
        posx3d[2*DimZ*DimX*DimY + i*DimX*DimY : 2*DimZ*DimX*DimY + (i+1)*DimX*DimY] = X_flat
        posy3d[i*DimX*DimY : (i+1)*DimX*DimY] = Y_flat
        posy3d[DimZ*DimX*DimY + i*DimX*DimY : DimZ*DimX*DimY + (i+1)*DimX*DimY] = Y_flat
        posy3d[2*DimZ*DimX*DimY + i*DimX*DimY : 2*DimZ*DimX*DimY + (i+1)*DimX*DimY] = Y_flat
        
    
    # Each element of this array tells to the CUDA program which component is calculating (Brfx = 0, Brfy = 1, Brfz = 2).
    loop = np.zeros(Dim, dtype = 'int32')
    loop[DimZ*DimX*DimY:2*DimZ*DimX*DimY] = 1
    loop[2*DimZ*DimX*DimY:3*DimZ*DimX*DimY] = 2
    
    "PROBLEM DIVISON"
    
    # If the H2D flat array is too big (3*DimX*DimY*DimZ) the program will divide it in smaller flat arrays with dimension
    # DimS = 512*512*32 = 8388608.
    
    if ( Dim  <= 8388608 ):
        DimC = 1
    else:
        # Divide the problem in 512x512x32 flat arrays.
        # This is only in computing terms, is not an actual rearrangement.
        # Dimension of the smaller arrays:
        DimS = 8388608
        # Subarray counter:
        DimC = int(math.ceil(Dim/(8388608)))
        
    if DimC == 1: # No division
        
        print('Initializing normal kernel')
        
        # Get the kernel function from the precompiled module.
        # Important! the compilation of the CUDA kernel will be performed during the import step, not during execution.
        func = mod.get_function("Current2Field")
        
        print('The 3D problem has ' + str(DimX*DimY*3*DimZ) + ' elements . . .')
        
        H2D_gpu = gpuarray.to_gpu(H2D)
        posz_gpu = gpuarray.to_gpu(posz3d)
    
        "DEFINING BLOCK SIZE"
        bdim = (8,8,8) # 512 threads per block (computing 1x)
        
        "DEFINING GRID SIZE"
        dx, mx = divmod(DimX, bdim[0])
        dy, my = divmod(DimY, bdim[1])
        dz, mz = divmod(3*DimZ/DimC, bdim[2])
        g_x = dx + (mx>0)
        g_y = dy + (my>0)
        g_z = dz + (mz>0)
        gdim = (g_x, g_y, int(g_z)) #Grid size
        
        print('Block size: ' + str(bdim))
        print('Grid size: ' + str(gdim))
        
        print('Performing the calculation . . .')
        func(Ixrs_gpu, Iyrs_gpu, H2D_gpu, X_gpu, Y_gpu, posz_gpu, np.int32(DimX), np.int32(DimY), np.int32(DimZ), block = bdim, grid=gdim)
        
        H2D = H2D_gpu.get()
        H2D = H2D*step*1e-7
        posz_gpu.gpudata.free()
        
    else:
        
        "DEFINING INDIVIDUAL MATRICES DIMENSION"
        DimXi = 512
        DimYi = 512
        DimZi = 32
        
        print('Initializing segmented kernel')
        print(mod)
        
        # Get the segmented kernel function from the precompiled module.
        # Important! the compilation of the CUDA kernel will be performed during the import step, not during execution.
        func = mod.get_function("SegCurrent2Field")
        
        print('Problem divided in ' + str(DimC) + ' slices . . .')
        print('Each slice has ' + str(DimS) + ' elements . . .')
        
        
        "DEFINING BLOCK SIZE"
        bdim = (8,8,8) # 8*8*8 = 512 threads per block (computing 1x)
                       # 1024 threads per block (computing 2x)
           
        "DEFINING GRID SIZE"
        dx, mx = divmod(DimXi, bdim[0])
        dy, my = divmod(DimYi, bdim[1])
        dz, mz = divmod(DimZi, bdim[2])
        g_x = dx + (mx>0)
        g_y = dy + (my>0)
        g_z = dz + (mz>0)
        gdim = (g_x, g_y, int(g_z)) # Grid size
        
        print('Block size: ' + str(bdim))
        print('Grid size: ' + str(gdim))
        
        i = 0
        Loop_time = 0
        Total_time = DimC*Loop_time
        
        for i in range(DimC):            
            
            if i == DimC - 1:
                
                print('Starting the last loop')
                
                                          
                Remaining_time = (DimC - i)*Loop_time

                Minutes, Hours = math.modf(Remaining_time/3600)
                Seconds, Minutes = math.modf(Minutes*60)
                
                print('Remaining time: ' + str(Hours) + ' h, ' + str(Minutes) + ' min, ' + str(round(Seconds,2)) + ' sec.')
                
                # Creating auxiliar arrays to store the last slice
                H2Di = np.zeros(DimS, np.float32)
                poszi = np.zeros(DimS, np.float32)
                poszi[0 : Dim - i*DimS] = posz3d[i*DimS : Dim]
                posxi = np.zeros(DimS, np.float32)
                posxi[0 : Dim - i*DimS] = posx3d[i*DimS : Dim]
                posyi = np.zeros(DimS, np.float32)
                posyi[0 : Dim - i*DimS] = posy3d[i*DimS : Dim]
                loopi = np.zeros(DimS, np.int32)
                loopi[0 : Dim - i*DimS] = loop[i*DimS : Dim]
                print(loopi[0])
                print(loopi[Dim - i*DimS-1])
                
                DimXii = DimXi
                DimYii = DimYi
                DimZii = DimZi
                #DimR = int(Dim - i*DimS)
                #DimYii = int(factorizehalf(DimR))
                #DimZii = int(factorizehalf(DimR/DimYii))
                #DimXii = int(DimR/(DimYii*DimZii))
                
                msize = (DimXii, DimYii, DimZii)
                
                "DEFINING BLOCK SIZE"
                bdim = (8,8,8) # 512 threads per block (computing 1x)
                                  
                "DEFINING GRID SIZE"
                dx, mx = divmod(DimXii, bdim[0])
                dy, my = divmod(DimYii, bdim[1])
                dz, mz = divmod(DimZii, bdim[2])
                g_x = dx + (mx>0)
                g_y = dy + (my>0)
                g_z = dz + (mz>0)
                gdim = (g_x, g_y, int(g_z)) # Grid size
                
                print('Submatrix size: ' + str(msize))
                print('Block size: ' + str(bdim))
                print('Grid size: ' + str(gdim))
                
                # Transferring auxiliar arrays to gpu to do the math
                H2D_gpu = gpuarray.to_gpu(H2Di)
                posz_gpu = gpuarray.to_gpu(poszi)
                posx_gpu = gpuarray.to_gpu(posxi)
                posy_gpu = gpuarray.to_gpu(posyi)
                loop_gpu = gpuarray.to_gpu(loopi)
                
                print('Performing the last calculation . . .')
                                          
                func(Ixrs_gpu, np.int32(DimX), Iyrs_gpu, np.int32(DimY), H2D_gpu, X_gpu, Y_gpu, posx_gpu, posy_gpu, posz_gpu, loop_gpu, np.int32(DimXii), np.int32(DimYii), np.int32(DimZii), block = bdim, grid=gdim)
                
                H2Di = H2D_gpu.get()
                H2D_gpu.gpudata.free()
                posz_gpu.gpudata.free()
                posx_gpu.gpudata.free()
                posy_gpu.gpudata.free()
                loop_gpu.gpudata.free()
            
                print('Transferring data to CPU . . .')
                H2D[i*DimS : Dim] = H2Di[0 : Dim - i*DimS]*step*1e-7
            
                print('Finished the last loop')
            
            else:
                
                print('Starting loop number ' + str(i+1))
                
                if i != 0:
                    Remaining_time = (DimC - i)*Loop_time

                    Minutes, Hours = math.modf(Remaining_time/3600)
                    Seconds, Minutes = math.modf(Minutes*60)
                                    
                    print('Remaining time: ' + str(Hours) + ' h, ' + str(Minutes) + ' min, ' + str(round(Seconds,2)) + ' sec.')
                    
                tic = timeit.default_timer();
                
                # Creating auxiliar arrays to store slices in the loops
                H2Di = np.zeros(DimS, np.float32)
                poszi = posz3d[i*DimS : (i+1)*DimS]
                posxi = posx3d[i*DimS : (i+1)*DimS]
                posyi = posy3d[i*DimS : (i+1)*DimS]
                loopi = loop[i*DimS : (i+1)*DimS]
                
                if loopi[0] == 0:
                    print('Bx component')
                elif loopi[0] == 1:
                    print('By component')
                else:
                    print('Bz component')
                
                # Transferring auxiliar arrays to gpu to do the math
                H2D_gpu = gpuarray.to_gpu(H2Di)
                posz_gpu = gpuarray.to_gpu(poszi)
                posx_gpu = gpuarray.to_gpu(posxi)
                posy_gpu = gpuarray.to_gpu(posyi)
                loop_gpu = gpuarray.to_gpu(loopi)
                            
                print('Performing the calculation . . .')
                
                func(Ixrs_gpu, np.int32(DimX), Iyrs_gpu, np.int32(DimY), H2D_gpu, X_gpu, Y_gpu, posx_gpu, posy_gpu, posz_gpu, loop_gpu, np.int32(DimXi), np.int32(DimYi), np.int32(DimZi), block = bdim, grid=gdim)
            
                # Freeing memory between loops
                H2Di = H2D_gpu.get()
                H2D_gpu.gpudata.free()
                posz_gpu.gpudata.free()
                posx_gpu.gpudata.free()
                posy_gpu.gpudata.free()
                loop_gpu.gpudata.free()
                
                # Saving loop result in the big array
                print('Transferring data to CPU . . .')
                H2D[i*DimS:(i+1)*DimS] = H2Di*step*1e-7
                
                print('Finished loop number ' + str(i+1))
                toc = timeit.default_timer();
                
                Loop_time = (Loop_time + (toc - tic))/(i+1) # s            
        
    Ixrs_gpu.gpudata.free()
    Iyrs_gpu.gpudata.free()
    
    print('Deflattening matrices . . .')
    
    Brfx = np.zeros((DimY,DimX,DimZ) , np.float32) 
    Brfy = np.zeros((DimY,DimX,DimZ) , np.float32)
    Brfz = np.zeros((DimY,DimX,DimZ) , np.float32)
    Brf = np.zeros((DimY,DimX,DimZ) , np.float32)
    
    i = 0
    j = 0
    k = 0
    
    "DEFLATTENING CURRENT MATRICES"
    for k in range(DimZ):
        for i in range(DimY):
            for j in range(DimX):
                Brfx[i,j,k] = H2D[DimX*i + j + DimX*DimY*k]
                Brfy[i,j,k] = H2D[DimX*DimY*DimZ + DimX*i + j + DimX*DimY*k]
                Brfz[i,j,k] = H2D[2*DimX*DimY*DimZ + DimX*i + j + DimX*DimY*k]
                Brf[i,j,k] = np.sqrt(Brfx[i,j,k]*Brfx[i,j,k] + Brfy[i,j,k]*Brfy[i,j,k] + Brfz[i,j,k]*Brfz[i,j,k])
                
    
    "STABLISHING DISCRETIZATION IN X,Y,Z"
    dx = np.absolute(posx[1] - posx[0])
    dy = np.absolute(posy[1] - posy[0])
    dz = np.absolute(posz[1] - posz[0])
               
    return Field(Brfx,Brfy,Brfz,Brf,posx,posy,posz,dx,dy,dz)
 
"""            LOAD EXPERIMENTAL DATA FUNCTION               """

def load_data_exp(path,norm,norm_mode,drift,substract,cut,old = False):

    # This function loads experimental transmission vs field vs frequency maps measured in the BlueFors dilution cryostat.
    # Path is the relative path of the .dat, including its name and extension (.dat)
    # norm is number of line you want to normalize the data (int number).
    # IMPORTANT! If norm == 0, the function will load the data without normalize it
    # norm mode is the normalization mode: 1. 'lin' to normalize in linear mode 2. 'log' to normalize in logarithmic mode and
    # 3. 'last' to normalize to the last line (in which case norm value is ignored).
    # drift is the value of the magnetic field drift (in T)
    
    print('loading file')
    FILE = open(path, 'r')
    FILE.readline()

    # Void lists to store experimental data
    field = []
    frequency = []
    TR = []
    
    # New cryostat or old cryostat?
    if old == False:
        n = 0
    elif old == True:
        n = 2
        
    # Reading line by line and storing data in the lists
    stop = 0
    while stop == 0:
        line = FILE.readline().split(',')
        if line == ['']:
            stop = 1
            break;
        else:
            field.append(float(line[1+n]))
            frequency.append(float(line[4+n]))            
            TR.append(float(line[6+n]))
    
    print('finished loading file')
    # Converting lists into arrays:
    field = np.array(field)
    frequency = np.array(frequency)
    TR = np.array(TR)
        
    # Building omega vector:
    stop = 0
    fn = 1 # number of frequency points
    while stop == 0:
        if frequency[fn] == frequency.min():
            break;
        else:
            fn = fn + 1
    
    # Creating frequency vector
    omega = np.linspace(frequency.min(),frequency.max(),fn)*1e-9 # GHz
   
    # Number of fields measured in experiment:
    bn = int(np.size(field)/fn)

    # Zeeman magnetic field (T)
    drift = 0.0
    n = bn # number of applied magnetic fields. Number of elements in the magnetic field vector 
    BP = (field[np.size(omega)] - field[0])# magnetic field step (T)
    reorder = False
    print(BP)
    if BP >= 0:
        B =[0.0, 0.0, field.min() + drift] # IN MOLECULE COORDINATE SYSTEM (Z parallel C3)
    elif BP < 0:
        print('negative step!')
        reorder = True
        #TR_aux = np.zeros(np.size(TR))
        #for i in range(n):
        #    TR_aux[(n-i-1)*fn:(n-i)*fn] = TR[i*fn:(i+1)*fn]
            
        #TR = TR_aux
        BP = -1*BP
        B =[0.0, 0.0, field.min() + drift] # IN MOLECULE COORDINATE SYSTEM (Z parallel C3)
        
           
    Bplot = np.zeros(bn - norm)
    
    #### ONLY FOR NORMALIZATION TO LAST MODE ######################
    
    stop = False
    if norm_mode == 'last':
        print('You have selected normalization to last line')
        norm = 1
        while stop == False:
            select = input('Do you want log or lin? (log/lin)')
            if select == 'log':                        
                stop = True
            elif select == 'lin':
                stop = True
            else: 
                print('Wrong mode . . . ')
                stop = False
    ###############################################################
    
    if substract == False:
        TRplot_exp = np.zeros((fn, bn-norm))
        for i in range(bn - norm):
            Bplot[i] = B[2]*1e3 # mT
            print(Bplot[i])
              
            if norm == 0 and norm_mode == 'log':
                TRplot_exp[:,i] = (TR[i*fn:(i+1)*fn]) #dB
            elif norm == 0 and norm_mode == 'lin':
                TRplot_exp[:,i] = np.power(10,(TR[i*fn:(i+1)*fn]/20))
            elif norm_mode == 'lin' and norm != 0:
                Si = np.power(10,(TR[i*fn:(i+1)*fn]/20))
                Sii = np.power(10,(TR[(i+norm)*fn:(i+norm+1)*fn]/20))
                TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
            elif norm_mode == 'log' and norm != 0:
                Si = TR[i*fn:(i+1)*fn]
                Sii = TR[(i+norm)*fn:(i+norm+1)*fn]
                TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
            elif norm_mode == 'last' and norm == 1:
                
                if select == 'log':
                    Si = TR[i*fn:(i+1)*fn]
                    Sii = TR[(bn - 1)*fn:(bn)*fn]
                    
                elif select == 'lin':
                    Si = np.power(10,(TR[i*fn:(i+1)*fn]/20))
                    Sii = np.power(10,(TR[(bn - 1)*fn:(bn)*fn]/20))

                TRplot_exp[:,i] = np.divide((Si - Sii), Sii)         
        
            B[2] = B[2] + BP
            
    if reorder == True:
        TRaux = np.zeros(np.shape(TRplot_exp))
        for i in range(bn-norm):
            TRaux[:,bn-norm-i-1] = TRplot_exp[:,i]
        TRplot_exp = TRaux
        del(TRaux)
        
    ##### SUBSTRACTING BACKGROUND #####
    if substract == True:
        plt.plot(omega,TR[0:np.size(omega)],'b-')
        plt.show() 
    
        lower_freq = float(input('Introduce frequency BELOW resonance (GHz): '))
        upper_freq = float(input('Introduce frequency ABOVE resonance (GHz): '))
        
        ask_cut = input('Do you want to cut (y/n)?')
        cut = False
        if ask_cut == 'y':
            cut = True
        
        ##### Finding FREQ positions in EXPERIMENTAL DATA #####
        down = 0
        stop = False
        while stop == False:
            if omega[down] >= lower_freq:
                stop = True
            else:
                down = down + 1
    
        up = np.size(omega)-1
        stop = False
        while stop == False:
            if omega[up] <= upper_freq:
                stop = True
            else:
                up = up - 1
            
        #######################################################
        
        if cut == True:
            omega_cut = omega[down:up]
            TR_cut = np.zeros((bn - norm)*np.size(omega_cut))
            print(up)
            print(down)
            print(up-down)
            print(np.size(omega_cut))
            
            for i in range(bn - norm):
                TR_cut[i*np.size(omega_cut):(i+1)*np.size(omega_cut)] = TR[i*fn + down :(i+1)*fn - (fn-up)]
                plt.plot(omega_cut, TR_cut[i*np.size(omega_cut):(i+1)*np.size(omega_cut)])
                
            plt.title('SELECTED FREQUENCY RANGE')
            plt.xlabel('Field (mT)' ,fontsize = 12, fontweight='bold')
            plt.ylabel('Frequency (GHz)' ,fontsize = 12, fontweight='bold')
            plt.tick_params(direction = 'in')    
            plt.show()
        
        iterations = int(input('Introduce number of iterations to remove background'))
        grade = int(input('Polynomial order do you want to use:'))
        per = float(input('Introduce fraction of points to fit (between 0 and 1): '))
        
        for it in range(iterations): 
            
            B =[0.0, 0.0, field.min() + drift]
            
            if cut == True:
                fn = np.size(omega_cut)
                TRplot_exp = np.zeros((fn, bn-norm))
                TR_background = background(TR_cut,omega_cut,int(fn*per),int(fn-fn*per),grade)
                
                # TR_background is in dB
                TR_level = TR_background[0]

                print('Level is: ' + str(round(TR_level,2)) + ' dB')
                
                for i in range(bn - norm):
                    Bplot[i] = B[2]*1e3 # mT  
                    TR_cut[i*fn:(i+1)*fn] = TR_cut[i*fn:(i+1)*fn] - TR_background + TR_level #dB
                    #TR_cut[i*fn:(i+1)*fn] = TR_cut[i*fn:(i+1)*fn] - background_line(TR_cut[i*fn:(i+1)*fn], omega_cut) 
                    if norm == 0 and norm_mode == 'log':
                        TRplot_exp[:,i] = TR_cut[i*fn:(i+1)*fn]
                    elif norm == 0 and norm_mode == 'lin':
                        TRplot_exp[:,i] = np.power(10,(TR_cut[i*fn:(i+1)*fn]/20))
                        #TRplot_exp[:,i] = TRplot_exp[:,i] - background_line(TRplot_exp[:,i], omega_cut)
                    
                    B[2] = B[2] + BP
                
            else:
                
                TRplot_exp = np.zeros((fn, bn-norm))
                
                TR_background = background(TR,omega,down,up,grade)
                # TR_background is in dB
                
                for i in range(bn - norm):
                    Bplot[i] = B[2]*1e3 # mT  
                    TR[i*fn:(i+1)*fn] = (TR[i*fn:(i+1)*fn]) - TR_background #dB
                    if norm == 0 and norm_mode == 'log':
                        TRplot_exp[:,i] = TR[i*fn:(i+1)*fn]
                    elif norm == 0 and norm_mode == 'lin':
                        TRplot_exp[:,i] = np.power(10,(TR[i*fn:(i+1)*fn]/20))
                    
                    B[2] = B[2] + BP
        
               
        for i in range(bn - norm):
            plt.plot(omega_cut, TRplot_exp[:,i])
    
        plt.title('BACKGROUND SUBSTRACTED')
        plt.xlabel('Frequency (GHz)' ,fontsize = 12, fontweight='bold')
        if norm_mode == 'log':
            plt.ylabel('S$_{21}$ (dB)' , fontsize = 12, fontweight='bold')
        elif norm_mode == 'lin':
            plt.ylabel('|t| (dimensionless)' , fontsize = 12, fontweight='bold')
        plt.tick_params(direction = 'in')
        plt.show()
        omega = omega_cut
    #######################################################
    
    ####### CLEANING DATA ##########
    if cut == True:
        clean = input('Do you want to clean (y/n)?')
    else:
        clean = 'n'
    
    stop = False
    
    if clean == 'y':
        
        clean_counter = 0
        
        while stop == False:
            
            if clean_counter == 0:        
                Data_clean = clean_data(Bplot, omega, TRplot_exp)
                TRplot_exp = Data_clean.TR
                omega = Data_clean.omega
                Bplot = Data_clean.Bplot
                clean_counter = clean_counter + 1
                
            else:    
                clean = input('Do you want to clean other segment(y/n)?')
                if clean == 'n': 
                    stop = True 
                else:    
                    Data_clean = clean_data(Bplot, omega, TRplot_exp)
                    TRplot_exp = Data_clean.TR
                    omega = Data_clean.omega
                    Bplot = Data_clean.Bplot
                    clean_counter = clean_counter + 1
        
    ###############################
    
    TR = TRplot_exp
    
    # TR is a matrix that stores de transmission values (normalized or not). Easier to handle than the raw files
    # omega is the frequency vector #GHz
    # Bplot is the magnetic field vector #mT
    
    return Data_exp(TR,omega,Bplot)

def load_powersweep(path,norm,norm_mode,old = False):

    # This function loads experimental transmission vs field vs frequency maps measured in the BlueFors dilution cryostat.
    # Path is the relative path of the .dat, including its name and extension (.dat)
    # norm is number of line you want to normalize the data (int number).
    # IMPORTANT! If norm == 0, the function will load the data without normalize it
    # norm mode is the normalization mode: 1. 'lin' to normalize in linear mode 2. 'log' to normalize in logarithmic mode and
    # 3. 'last' to normalize to the last line (in which case norm value is ignored).
    # drift is the value of the magnetic field drift (in T)
    
    FILE = open(path, 'r')
    FILE.readline()

    # Void lists to store experimental data
    power = []
    frequency = []
    TR = []
    
    # New cryostat or old cryostat?
    if old == False:
        n = 0
    elif old == True:
        n = 2
        
    # Reading line by line and storing data in the lists
    stop = False
    while stop == False:
        line = FILE.readline().split(',')
        if line == ['']:
            stop = True
            break;
        else:
            power.append(float(line[3+n]))
            frequency.append(float(line[4+n]))            
            TR.append(float(line[6+n]))

    # Converting lists into arrays:
    power = np.array(power)
    frequency = np.array(frequency)
    TR = np.array(TR)
        
    # Building omega vector:
    stop = 0
    fn = 1 # number of frequency points
    while stop == 0:
        if frequency[fn] == frequency.min():
            break;
        else:
            fn = fn + 1
    
    # Creating frequency vector
    omega = np.linspace(frequency.min(),frequency.max(),fn)*1e-9 # GHz
   
    # Number of power in measured in experiment:
    pn = int(np.size(power)/fn) # number of applied power in values. Number of elements in the power vector 

    # Power in vector
    ΔP = (power[np.size(omega)] - power[0]) # Power in step
    
    Pplot = np.zeros(pn - norm)
    
    #### ONLY FOR NORMALIZATION TO LAST MODE ######################
    
    stop = False
    if norm_mode == 'last':
        print('You have selected normalization to last line')
        norm = 1
        while stop == False:
            select = input('Do you want log or lin? (log/lin)')
            if select == 'log':                        
                stop = True
            elif select == 'lin':
                stop = True
            else: 
                print('Wrong mode . . . ')
                stop = False
    ###############################################################
    
    TRplot_exp = np.zeros((fn, pn-norm))
    
    for i in range(pn - norm):
                  
        if norm == 0 and norm_mode == 'log':
            TRplot_exp[:,i] = (TR[i*fn:(i+1)*fn]) #dB
        elif norm == 0 and norm_mode == 'lin':
            TRplot_exp[:,i] = np.power(10,(TR[i*fn:(i+1)*fn]/20))
        elif norm_mode == 'lin' and norm != 0:
            Si = np.power(10,(TR[i*fn:(i+1)*fn]/20))
            Sii = np.power(10,(TR[(i+norm)*fn:(i+norm+1)*fn]/20))
            TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
        elif norm_mode == 'log' and norm != 0:
            Si = TR[i*fn:(i+1)*fn]
            Sii = TR[(i+norm)*fn:(i+norm+1)*fn]
            TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
        elif norm_mode == 'last' and norm == 1:
            
            if select == 'log':
                Si = TR[i*fn:(i+1)*fn]
                Sii = TR[(pn - 1)*fn:(pn)*fn]
                
            elif select == 'lin':
                Si = np.power(10,(TR[i*fn:(i+1)*fn]/20))
                Sii = np.power(10,(TR[(pn - 1)*fn:(pn)*fn]/20))

            TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
            
        Pplot[i] = power[0] + i*ΔP #dBm
            
    TR = TRplot_exp
    
    # TR is a matrix that stores de transmission values (normalized or not). Easier to handle than the raw files
    # omega is the frequency vector #GHz
    # Pplot is the applied power vector #dBm
    
    return Data_exp(TR,omega,Pplot)


def load_tempsweep(path,name, norm = 1, norm_mode = 'log'):

    # This function loads experimental transmission vs field vs frequency maps measured in the BlueFors dilution cryostat.
    # Path is the relative path of the .dat, including its name and extension (.dat)
    # norm is number of line you want to normalize the data (int number).
    # IMPORTANT! If norm == 0, the function will load the data without normalize it
    # norm mode is the normalization mode: 1. 'lin' to normalize in linear mode 2. 'log' to normalize in logarithmic mode and
    # 3. 'last' to normalize to the last line (in which case norm value is ignored).
    # drift is the value of the magnetic field drift (in T)
    
    #print([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    
    print(len(name))
    print(name)
    
    files = [file for file in os.listdir(path) if file[:len(name)] == name]
    print(len(files))
    
    # Void lists to store experimental data
    temp = []
    frequency = []
    
    i = 0
    
    for f in files:
        
        TR_aux = []
        
        FILE = open(path + f, 'r')
        FILE.readline()
        
        line = FILE.readline().split(',')
        temp.append(float(line[2]))
        
         # Reading line by line and storing data in the lists
        stop = False
        while stop == False:
            line = FILE.readline().split(',')
            if line == ['']:
                stop = True
                break;
            else:
                
                if i == 0: frequency.append(float(line[4])) 
                TR_aux.append(float(line[6]))
        
        # Building transmission 2D matrix (S21 vs temp and freq)
                
                
        if i == 0:
            TR = np.zeros((len(TR_aux),len(files)))
            TR[:,0] = TR_aux
            
        else: TR[:,i] = TR_aux
        
        i = i + 1
        
    temp = np.array(temp)
    frequency = np.array(frequency)
    
    print(frequency)
    
        
    # Ordering temps and transmission matrix
    stop = False
    i = 1
   
    AUX2 = 0
    AUX1 = 0
    
    while stop == False:
        if temp[i-1] < temp[i]:  
            
            j = i-1
            aux = temp[i]            
            AUX2 = TR[:,j].copy()
            AUX1 = TR[:,i].copy()                                            
            temp[i] = temp[i-1]                    
            temp[i-1] = aux         
            TR[:,i] = AUX2
            TR[:,j] = AUX1
            i = 1
            
        else:
            i = i + 1
            if i == len(temp):
                stop = True
    
    ###############################################################            
    # NORMALIZATION
                
    tn = np.size(temp)
    fn = np.size(frequency)
    TRplot_exp = np.zeros((fn , tn - norm))
    
    for i in range(tn - norm):
                  
        if norm == 0 and norm_mode == 'log':
            TRplot_exp[:,i] = TR[:,i] #dB
        elif norm == 0 and norm_mode == 'lin':
            TRplot_exp[:,i] = np.power(10,(TR[:,i]/20))
        elif norm_mode == 'lin' and norm != 0:
            Si = np.power(10,(TR[:,i]/20)) 
            Sii = np.power(10,(TR[:,i+1]/20))
            TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
            temp = temp[0:tn-norm]
        elif norm_mode == 'log' and norm != 0:
            Si = TR[:,i] #dB
            Sii = TR[:,i+1] #dB
            TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
            temp = temp[0:tn-norm]
        elif norm_mode == 'last' and norm == 1:
            temp = temp[0:tn-1]
            if norm_mode == 'log':
                Si = TR[:,i] #dB
                Sii = TR[:,np.size(tn-norm)-1] #dB
                
            elif norm_mode == 'lin':
                Si = np.power(10,(TR[:,i]/20))
                Sii = np.power(10,(TR[:,np.size(tn-norm)-1]/20))

            TRplot_exp[:,i] = np.divide((Si - Sii), Sii)
            
            
    TR = TRplot_exp
    
    
    return Data_exp(TR,frequency,temp)

""" LORENTZIAN/FANO ONE TRANSMISSION vs FREQUENCY ADJUST """

def fit_measurement(path, x0, substract,
                       field_limits = [],
                       bounds = (np.array([0.,0.,0.,0.]), np.array([1.,1.,1.,1.]))):    
           
    ##### EXPERIMENTAL DATA LOADING #####
    
    # Loading parameters
    norm_mode = 'lin'
    norm = 0
    drift = 0.0
   
    # Executing qf function to load the file
    Data_exp = load_data_exp(path,norm,norm_mode,drift,substract,False)
    
    # Data_exp.TR: Color tranmission map (if norm = 0, and norm_mode = 'lin' it is linear without normalization)
    # Data_exp.Bplot: X axis (Fields) ¡(mT)!
    # Data_exp.omega: Y axis (Frequencies) ¡(GHz)!
    ######################################
    
    limits = np.array([0,0])
    
    if field_limits != []:
        limits[0] = find_value(field_limits[0],Data_exp.Bplot)
        limits[1] = find_value(field_limits[1],Data_exp.Bplot)
    else:
        limits[0] = 0; limits[1] = np.size(Data_exp.Bplot)
    
    
    n = np.size(Data_exp.Bplot[limits[0]:limits[1]])
    l = np.size(Data_exp.omega) 
    
    
    ωr = x0[4] # NOT FIXED!! (omega2 in model)
    
    
    x0_fix = x0[0:4] # FROM FIT ONE CURVE NO COUPLING - FIXED PARAMETERS

    x0 = x0[4:9] # GAMMA AND COUPLING - NOT FIXED PARAMETERS

    
    B , F = np.meshgrid(Data_exp.Bplot[limits[0]:limits[1]]*1e-3, Data_exp.omega*1e9) # create meshgrid for 2D data (T and Hz)

    xdata = np.vstack((B.ravel(), F.ravel()/ωr)) # Stack flattened (ravel) grids (T and Hz)
    
    TRaprox = np.zeros((l,n))
    
    print('Initial parameters: BAD-GOOD LER COUPLING (Hz), BAD LER LOSSES (Hz), GOOD LER LOSSES (HZ), BAD LER FREQUENCY (Hz)')
    print(x0_fix)
    
    mu_B = 9.27401e-24 / 6.6261e-34 #Hz/T
    x0[3] = x0[3]*mu_B # Multiplying slope
    x0[4] = x0[4]
    
    print('Initial parameters: GOOD LER FREQ (Hz), SPIN LOSSES (Hz), SPIN COUPLING (HZ), SPIN SLOPE (Hz/T)')
    print(x0)
    
            
    constat_arguments = {'fixed': np.array(x0_fix)}
        
    result = sop.least_squares(least_respin, x0, jac = '3-point', args =[xdata,
                                                                      Data_exp.TR[:,limits[0]:limits[1]].ravel()],
                                                                      kwargs = constat_arguments,  
                                                                      bounds = bounds)     
        
    params_aux = np.array([result.x[0], result.x[1], result.x[2], result.x[3], result.x[4]])

    # PARAMS_AUX: [ωr, γ, Ω , G] 
    # X0_FIX:  [a, κi, κe, φ] 
                             
    params_aux = np.append(x0_fix, params_aux)
    
    print('Adjusted parameters: GOOD LER FREQ (Hz), SPIN LOSSES (Hz), SPIN COUPLING (HZ), SPIN SLOPE')
    print(params_aux)   
        
    TRaprox = func_respin(B, F, params_aux)
                             
    params = params_aux
                                           
        
    return Data_adjust(Data_exp.TR[:,limits[0]:limits[1]],TRaprox,params,B,F)


""" REMOVE BACKGROUND """

def background(TR,omega,down,up,grade):
    
    # THIS FUNCTION CALCULATES THE BACKGROUND OF THE EXPERIMENTAL TRANSMISSION DATA
    TR_nopeak = np.zeros(np.size(omega)-(up-down))
    omega_nopeak = np.zeros(np.size(omega)-(up-down))
                            
    TR_nopeak[0:down] = TR[0:down]
    TR_nopeak[down:np.size(omega)-(up-down)] = TR[up:np.size(omega)]
    omega_nopeak[0:down] = omega[0:down]
    omega_nopeak[down:np.size(omega)-(up-down)] = omega[up:np.size(omega)]
    
    p = np.poly1d(np.polyfit(omega_nopeak, TR_nopeak, grade))
    print(p)
    
    TR_background = p(omega)
        
    plt.plot(omega,TR[0:np.size(omega)],'b-')
    plt.plot(omega,TR_background,'--', color = 'orange')
    plt.show()             
    
    # TR_background is in dB
    
    return TR_background

def background_line(TR,omega):
    
    y = np.array([TR[0],TR[np.size(omega)-1]])
    x = np.array([omega[0],omega[np.size(omega)-1]])
    
    p = np.poly1d(np.polyfit(x,y,1))
    
    line_background = p(omega) - 1.0           
    
    return line_background

""" NEAREST ISOTOPE """
def nearest(B,LER_frequency):
    
    samples = ['171Yb_Trensal','172Yb_Trensal','173Yb_Trensal']
    energies = np.array([])
    ground = np.array([])
    excited = np.array([])
    isotope = np.array([])
    
    for i in range(3):
        Sample = load_sample(samples[i])
        QData = coupling(Sample,B,0.15,0,0,0)
        energies = np.append(energies,QData.E_EPR[0:int(Sample.NJ/2),0])
        ground = np.append(ground,QData.E_EPR[0:int(Sample.NJ/2),1])
        excited = np.append(excited,QData.E_EPR[0:int(Sample.NJ/2),2])
        for k in range(int(Sample.NJ/2)):
            isotope = np.append(isotope,i)      
            
    index = 0
    for i in range(np.size(energies)):
        if abs(energies[i] - LER_frequency) < abs(energies[index] - LER_frequency):
            index = i
    
    nearest_level = np.array([energies[index],ground[index],excited[index],isotope[index]])
    
    return nearest_level

""" CLEAN RESONATOR DATA """

def clean_data(B, omega, TR):
    
    bn = np.size(B)
    fn = np.size(omega)
    
    ### PREVIEW DATA ###
    Bplot , Fplot = np.meshgrid(B, omega*1e9)
    plt.pcolor(Bplot,Fplot,TR, cmap=plt.get_cmap('RdBu'))
    plt.show()
    
    Field_pre = float(input('Introduce magnetic field to start cleaning (mT)'))
    stop = False
    Field_position_pre = 0
    while stop == False:
        if B[Field_position_pre] >= Field_pre:
            stop = True
        else:
            Field_position_pre = Field_position_pre + 1
                      
    Field_post = float(input('Introduce magnetic field to end cleaning (mT)'))
    stop = False
    Field_position_post = 0
    while stop == False:
        if B[Field_position_post] >= Field_post:
            stop = True
        else:
            Field_position_post = Field_position_post + 1
    
    Frequency_tolerance = float(input('Introduce frequency tolerance: (Hz) '))
    
    for i in range(Field_position_pre,Field_position_post):
        LER_frequency_pre = TR[:,i-1].min()
        LER_frequency_post = TR[:,i].min()
        
        ### FIDING POSITIONS ###
        LER_position_pre = 0
        LER_position_post = 0             
        stop = False        
        while stop == False:
            if TR[LER_position_pre,i-1] == LER_frequency_pre:
                stop = True
            else:
                LER_position_pre = LER_position_pre + 1
        
        stop = False
        while stop == False:
            if TR[LER_position_post,i] == LER_frequency_post:
                stop = True
            else:
                LER_position_post = LER_position_post + 1
        
        LER_frequency_pre = omega[LER_position_pre]
        LER_frequency_post = omega[LER_position_post]
        Delta_Frequency = (LER_frequency_post - LER_frequency_pre)*1.0e9        
        Delta_LER_position = LER_position_post-LER_position_pre
        
        #plt.plot(TR[:,Field_position_pre])
        #plt.plot(TR[:,Field_position_post])
        #plt.show()
        
        #########################
        
        if abs(Delta_Frequency) >= Frequency_tolerance:
            
            print('Magnetic field = ' + str(B[i]) + ' mT')
            print(Delta_Frequency)
            print(LER_position_pre)
            print(LER_position_post)
            if i <= bn - 50 and i>=50:
                Bplot , Fplot = np.meshgrid(B[i-50:i+50], omega[LER_position_post-50:LER_position_post+50]*1e9)
                TRplot = TR[LER_position_post-50:LER_position_post+50,i-50:i+50]
            elif i < 50:
                Bplot , Fplot = np.meshgrid(B[0:i+50], omega[LER_position_post-50:LER_position_post+50]*1e9)
                TRplot = TR[LER_position_post-50:LER_position_post+50,0:i+50]
            elif i > bn - 50:
                Bplot , Fplot = np.meshgrid(B[i-50:bn], omega[LER_position_post-50:LER_position_post+50]*1e9)
                TRplot = TR[LER_position_post-50:LER_position_post+50,i-50:bn]
                
            plt.pcolor(Bplot,Fplot,TRplot, cmap=plt.get_cmap('RdBu'))
            plt.axvline(B[i], color='w', linestyle='--')
            plt.show()
                             
            fix = input('Do you want to fix this discontinuity? (y/n)')
            
            ##### FIXING DISCONTINUITY #####
            if fix == 'y':
                if (LER_position_post < LER_position_pre):                    
                    for k in range(fn-abs(Delta_LER_position)-1,-1,-1):                        
                        TR[k+abs(Delta_LER_position),i:bn] = TR[k,i:bn]
                    
                    print('fixing')
                        
                    TR = TR[abs(Delta_LER_position):fn,:]
                    omega = omega[abs(Delta_LER_position):fn]
                    fn = np.size(omega)
                    
                elif (LER_position_post > LER_position_pre):
                    for k in range(fn-abs(Delta_LER_position)):
                        TR[k,i:bn] = TR[k+abs(Delta_LER_position),i:bn]
                    
                    print('fixing')
                    TR = TR[0:fn-abs(Delta_LER_position),:]
                    omega = omega[0:fn-abs(Delta_LER_position)]
                    fn = np.size(omega)
                    
            elif fix == 'n':
                print('Jumping to next discontinuity')

    return Data_clean(TR, omega, B)

##### FIND INDEX IN A 1D ARRAY ######

def find_value(value, vector):
    
    position = 0
    stop = False
    while stop == False:
        if vector[position] >= value:
            stop = True
        elif position == np.size(vector)-1:
            stop = True
            position = False
        else:
            position = position + 1
        
    return position

##### RESONANCE FREQUENCY VS FIELD #####

def freqvfield(path, limits = None):
    
   
    ##### EXPERIMENTAL DATA LOADING #####
    norm_mode = 'lin'
    norm = 0
    drift = 0.0
    substract = True
    Data_exp = load_data_exp(path,norm,norm_mode,drift,substract,False)
    #####################################
   
    if limits.any() == None:
        limits = np.array([0, np.size(Data_exp.Bplot)])
    else:
        limits[0] = find_value(limits[0],Data_exp.Bplot)
        limits[1] = find_value(limits[1],Data_exp.Bplot)

        
    n = np.size(Data_exp.Bplot[limits[0]:limits[1]])
    l = np.size(Data_exp.omega)
    Fres = np.zeros(n)
    Bres = np.zeros(n)
    
    for i in range(n):
        
        res = np.where(Data_exp.TR[:,i+limits[0]] == Data_exp.TR[:,i+limits[0]].min())
        Fres[i] = Data_exp.omega[res[0]]
        Bres[i] = Data_exp.Bplot[i+limits[0]]
        
    return FvF(Fres,Bres)

#####################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value),)
    idxy = np.unravel_index(idx, array.shape)
    return idxy




"""            NO COUPLING GAUSSIAN FUNCTION               """

def least_lorentz(params, x, y, LER_freq, Energy_level):
    # Function to fit the data with least squares.
    
    kappa = params[0]
    G = params[1]
    gamma = params[2]
   
    K = G/(1j*(Energy_level - x) + gamma) 
    
    residual = y - np.absolute(1.0 - kappa/(1j*(LER_freq - x) + kappa + K))
    return residual

def func_lorentz(x, params, LER_freq, Energy_level):
    # Function to plot the fitting.
    
    kappa = params[0]
    G = params[1]
    gamma = params[2]
   
    K = G/(1j*(Energy_level - x) + gamma) 
        
    return np.absolute(1.0 - kappa/(1j*(LER_freq - x) + kappa + K))

"""def least_fano(params, x, y):
    # Function to fit the data with least squares.
    
    LER_freq = params[0]
    kappa = params[1]
    q_fano = params[2]
    
    epsilon = 2*(x - LER_freq)/kappa
        
    residual = y - (np.power((q_fano + epsilon),2)/(1 + np.power(epsilon,2)))
    return residual"""

"""def func_fano(x, params):
    # Function to fit the data with least squares.
    
    LER_freq = params[0]
    kappa = params[1]
    q_fano = params[2]
    
    epsilon = 2*(x - LER_freq)/kappa
        
    return (np.power((q_fano + epsilon),2)/(1 + np.power(epsilon,2)))"""
                         
def least_background(params,x,y):
    # Function to fit data background with least squares
    
    K = 0
    
    for i in range(np.size(params)-2):
        K = K + params[i]*x**i
    
    K = K + params[-2]*np.cos(x) + params[-1]*np.sin(x)
                         
    return y - K

def func_background(x, params):
    # Function to fit data background with least squares                
    
    K = 0
    
    for i in range(np.size(params)-2):
        K = K + params[i]*x**i
    
    K = K + params[-2]*np.cos(x) + params[-1]*np.sin(x)

    return K

##### FANO FUNCTIONS FOR ADJUSTMENT WITH NO SPINS ##########

"""def M_2f (λ, γ1, γ2, ω1, ω2, ω):
    ω = ω
    return np.array([[ω1 - ω - 1j* γ1, λ ], 
                     [λ, ω2 - ω - 1j* γ2 ]], dtype=complex)

def X1_2f(λ, γ1, γ2, ω1, ω2, ω):
    ω=ω
       
    Mat = M_2f (λ, γ1, γ2, ω1, ω2, ω)
    inverse = np.linalg.inv(Mat)
    f1 = np.array([1.,0.],  dtype=complex)
    xarray = np.matmul(inverse,f1)
    
    return (np.absolute(xarray[0]))"""

def M_2f (κi, κe, ωr, γ , G, Ω, ω): # no spins
    ω=ω
    κ = κi + κe
    return np.array([[1j*(ωr - ω) + κ , -1j*G], 
                     [-1j*G, 1j*(Ω - ω) + γ]], dtype=complex)


def X1_2f(a, κi, κe, ωr, φ, γ, G, Ω, ω):
    ω=ω
    Mat= M_2f (κi, κe, ωr, γ , G, Ω, ω)
    inverse = np.linalg.inv(Mat)
    f1=np.array([np.sqrt(κe*np.exp(1j*φ)),0.],  dtype=complex)
    xarray = np.matmul(inverse,f1)
    
    return a*np.absolute(1-np.sqrt(κe*np.exp(1j*φ))*xarray[0])

def X1_2f_cpw(κi, κe, ωr, φ, γ, G, Ω, ω):
    ω=ω
    Mat= M_2f (κi, κe, ωr, γ , G, Ω, ω)
    inverse = np.linalg.inv(Mat)
    f1=np.array([np.sqrt(κe*np.exp(1j*φ)),0.],  dtype=complex)
    xarray = np.matmul(inverse,f1)
    
    return np.absolute(np.sqrt(κe*np.exp(1j*φ))*xarray[0])

"""def least_fano(params, ω, t, ω2):
    ω=ω

    λ = params[0] # fix
    γ1 = params[1] # fix
    γ2 = params[2] # fix
    ω1 = params[3] # fix
    
    myfunction = np.vectorize(X1_2f)
    
    return t - myfunction(λ, γ1, γ2, ω1, ω2, ω)
    
def func_fano(ω , params):
    ω=ω

    λ = params[0] # fix
    γ1 = params[1] # fix
    γ2 = params[2] # fix
    ω1 = params[3] # fix
    ω2 = params[4] # fix
    
    myfunction = np.vectorize(X1_2f)
    
    return myfunction(λ, γ1, γ2, ω1, ω2, ω)"""

def least_res(params, ω, t, fixed):
    ω=ω

    a = params[0]
    κi = params[1] 
    κe = params[2]
    φ = params[3]
    
    ωr = fixed[0]
    γ = fixed[1] 
    G = fixed[2]
    Ω = fixed[3] 
    
    myfunction = np.vectorize(X1_2f)
    
    return t - myfunction(a, κi, κe, ωr, φ, γ, G, Ω, ω)


def least_res_cpw(params, ω, t, fixed):
    ω=ω

    κi = params[0] 
    κe = params[1]
    φ = params[2]
    
    ωr = fixed[0]
    γ = fixed[1] 
    G = fixed[2]
    Ω = fixed[3] 
    
    myfunction = np.vectorize(X1_2f_cpw)
    
    return t - myfunction(κi, κe, ωr, φ, γ, G, Ω, ω)

def min_res_cpw(params,ω,t):
    ω=ω

    κi = params[0] 
    κe = params[1]
    φ = params[2]
    ωr = params[3]
    
    γ = 0. 
    G = 0.
    Ω = 0.
    
    myfunction = np.vectorize(X1_2f_cpw)
    
    return np.sum(np.abs((t - myfunction(κi, κe, ωr, φ, γ, G, Ω, ω))**2))



def least_respin(params, M, T, fixed):
    B , ω = M 
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.

    a = fixed[0]
    κi = fixed[1] 
    κe = fixed[2]
    φ = fixed[3]
    
    ωr = params[0]
    γ = params[1]
    G = params[2]    
    Ω = params[3]*B + params[4] 
        
    
    myfunction = np.vectorize(X1_2f)
    T_apx = myfunction(a, κi, κe, ωr, φ, γ, G, Ω, ω) #1D Array
    
    return T - T_apx


def func_respin(X, Y, params):
    
    n = X.shape[1]
    l = X.shape[0]
    
    M = np.vstack((X.ravel(),Y.ravel()))
    
    B , ω = M
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.

    a = params[0]
    κi = params[1] 
    κe = params[2]
    φ = params[3]
    ωr = params[4]
    γ = params[5] 
    G = params[6]
    Ω = params[7]*B + params[8] 
    
    
    T = np.zeros((l,n))
    
    myfunction = np.vectorize(X1_2f)
    T_apx = myfunction(a, κi, κe, ωr, φ, γ, G, Ω, ω) #1D Array
    
    for j in range(l):
        for i in range(n):
            T[j,i] = T_apx[i + j*n]
            
    return T

def func_respin_cpw(X, Y, params):
    
    n = X.shape[1]
    l = X.shape[0]
    
    M = np.vstack((X.ravel(),Y.ravel()))
    
    B , ω = M
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.

    κi = params[0] 
    κe = params[1]
    φ = params[2]
    ωr = params[3]
    γ = params[4] 
    G = params[5]
    Ω = params[6]*B + params[7] 
    
    
    T = np.zeros((l,n), dtype = complex)
    
    myfunction = np.vectorize(X1_2f_cpw)
    T_apx = myfunction(κi, κe, ωr, φ, γ, G, Ω, ω) #1D Array
    
    for j in range(l):
        for i in range(n):
            T[j,i] = T_apx[i + j*n]
            
    return np.absolute(T)

##########################################################
                                         
"""def M_3f (λ, γ1, γ2, ω1, ω2, ω, g, γq, Ω):
    ω=ω
    return np.array([[ω1 - ω - 1j* γ1, λ, 0.], 
                     [λ,ω2 - ω - 1j* γ2, g ],
                     [0., g, Ω - ω - 1j* γq] ], dtype=complex)


def X1_3f(ω, B, λ, γ1, γ2, ω1, ω2, g, γq, Ω, A):
    
    Mat= M_3f(λ, γ1, γ2, ω1, ω2, ω, g, γq, Ω*B+A)
                                             
    inverse = np.linalg.inv(Mat)
    f1=np.array([1.,0.,0.],  dtype=complex)
    xarray = np.matmul(inverse,f1)
    
    return (np.absolute(xarray[0]))       

def least_fano_spins(params, M, T, λ, γ1, γ2, ω1):
                                             
    B , ω = M 
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.
    
    ω2 = params[0] # Good resonator frequency
    γq = params[1] # Spin losses
    G = params[2] # Spin coupling
    Ω = params[3] # Energy level slope (Hz/T)
    A = params[4] # Energy level y axis cut (Hz)
                                                              
    myfunction = np.vectorize(X1_3f)
                                             
    T_apx = myfunction(ω, B, λ, γ1, γ2, ω1, ω2, G, γq, Ω, A) #1D Array
        
    return T - T_apx
                            
def func_fano_spins(X, Y, params):
    
    n = X.shape[1]
    l = X.shape[0]
    
    M = np.vstack((X.ravel(),Y.ravel()))
    
    B , ω = M
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.
                                             
    λ = params[0]
    γ1 = params[1] 
    γ2 = params[2] 
    ω1 = params[3] 
    ω2 = params[4] 
    γq = params[5]
    G = params[6]
    Ω = params[7]
    A = params[8]
    
    T = np.zeros((l,n))
                  
    myfunction = np.vectorize(X1_3f)
    T_apx = myfunction(ω, B, λ, γ1, γ2, ω1, ω2, G, γq, Ω, A) #1D Array
                  
    for j in range(l):
        for i in range(n):
            T[j,i] = T_apx[i + j*n]
                  
    return T"""

##########################################################
    
def M_5f (κi, κe, ωr, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, ω): # no spins
    ω=ω
    κ = κi + κe
    return np.array([[1j*(ωr - ω) + κ , -1j*G1, -1j*G2, -1j*G3, -1j*G4], 
                     [-1j*G1, 1j*(np.abs(Ω1) - ω) + γ1, 0, 0, 0],
                     [-1j*G2, 0, 1j*(np.abs(Ω2) - ω) + γ2, 0, 0],
                     [-1j*G3, 0, 0, 1j*(np.abs(Ω3) - ω) + γ3, 0],
                     [-1j*G4, 0, 0, 0, 1j*(np.abs(Ω4) - ω) + γ4]], dtype=complex)


def X1_5f(a, κi, κe, ωr, φ, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, ω):
    ω=ω
    Mat= M_5f (κi, κe, ωr, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, ω)
    inverse = np.linalg.inv(Mat)
    f1=np.array([np.sqrt(κe*np.exp(1j*φ)),0.,0.,0.,0.],  dtype=complex)
    xarray = np.matmul(inverse,f1)
    
    return a*np.absolute(1-np.sqrt(κe*np.exp(1j*φ))*xarray[0])

def M_7f (κi, κe, ωr, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, γ5, G5, Ω5, γ6, G6, Ω6, ω): # no spins
    ω=ω
    κ = κi + κe
    return np.array([[1j*(ωr - ω) + κ , -1j*G1, -1j*G2, -1j*G3, -1j*G4, -1j*G5, -1j*G6], 
                     [-1j*G1, 1j*(np.abs(Ω1) - ω) + γ1, 0, 0, 0, 0, 0],
                     [-1j*G2, 0, 1j*(np.abs(Ω2) - ω) + γ2, 0, 0, 0, 0],
                     [-1j*G3, 0, 0, 1j*(np.abs(Ω3) - ω) + γ3, 0, 0, 0],
                     [-1j*G4, 0, 0, 0, 1j*(np.abs(Ω4) - ω) + γ4, 0, 0],
                     [-1j*G5, 0, 0, 0, 0, 1j*(np.abs(Ω5) - ω) + γ5, 0],
                     [-1j*G6, 0, 0, 0, 0, 0, 1j*(np.abs(Ω6) - ω) + γ6]], dtype=complex)


def X1_7f(a, κi, κe, ωr, φ, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, γ5, G5, Ω5, γ6, G6, Ω6, ω):
    ω=ω
    Mat= M_7f (κi, κe, ωr, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, γ5, G5, Ω5, γ6, G6, Ω6, ω)
    inverse = np.linalg.inv(Mat)
    f1=np.array([np.sqrt(κe*np.exp(1j*φ)),0.,0.,0.,0.,0.,0.],  dtype=complex)
    xarray = np.matmul(inverse,f1)
    
    return a*np.absolute(1-np.sqrt(κe*np.exp(1j*φ))*xarray[0])

def least_tworespin(params, M, T, fixed):
    B , ω = M 
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.

    a = fixed[0]
    κi = fixed[1] 
    κe = fixed[2]
    φ = fixed[3]
    
    ωr = params[0]
    
    γ1 = params[1]
    G1 = params[2]    
    Ω1 = params[3]*B + params[4]
    
    γ2 = params[5]
    G2 = params[6]    
    Ω2 = params[7]*B + params[8] 
        
    
    myfunction = np.vectorize(X1_5f)
    T_apx = myfunction(a, κi, κe, ωr, φ, γ1, G1, Ω1, γ2, G2, Ω2, ω) #1D Array
    
    return T - T_apx

def func_fourrespin(X, Y, params):
    
    n = X.shape[1]
    l = X.shape[0]
    
    M = np.vstack((X.ravel(),Y.ravel()))
    
    B , ω = M
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.

    a = params[0]
    κi = params[1] 
    κe = params[2]
    φ = params[3]
    ωr = params[4]
    
    γ1 = params[5] 
    G1 = params[6]
    Ω1 = params[7]*B + params[8]
    
    γ2 = params[9] 
    G2 = params[10]
    Ω2 = params[11]*B + params[12]
    
    γ3 = params[13] 
    G3 = params[14]
    Ω3 = params[15]*B + params[16]
    
    γ4 = params[17] 
    G4 = params[18]
    Ω4 = params[19]*B + params[20]
 
    
    T = np.zeros((l,n))
    
    myfunction = np.vectorize(X1_5f)
    T_apx = myfunction(a, κi, κe, ωr, φ, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, ω) #1D Array
    
    for j in range(l):
        for i in range(n):
            T[j,i] = T_apx[i + j*n]
            
    return T

def func_sixrespin(X, Y, params):
    
    n = X.shape[1]
    l = X.shape[0]
    
    M = np.vstack((X.ravel(),Y.ravel()))
    
    B , ω = M
    # M is a (2, N) array. First row is B flattened. Second row is ω flattened
    # B is in T, ω is in Hz.

    a = params[0]
    κi = params[1] 
    κe = params[2]
    φ = params[3]
    ωr = params[4]
    
    γ1 = params[5] 
    G1 = params[6]
    Ω1 = params[7]*B + params[8]
    
    γ2 = params[9] 
    G2 = params[10]
    Ω2 = params[11]*B + params[12]
    
    γ3 = params[13] 
    G3 = params[14]
    Ω3 = params[15]*B + params[16]
    
    γ4 = params[17] 
    G4 = params[18]
    Ω4 = params[19]*B + params[20]
    
    γ5 = params[21] 
    G5 = params[22]
    Ω5 = params[23]*B + params[24]
    
    γ6 = params[25] 
    G6 = params[26]
    Ω6 = params[27]*B + params[28]
 
    
    T = np.zeros((l,n))
    
    myfunction = np.vectorize(X1_7f)
    T_apx = myfunction(a, κi, κe, ωr, φ, γ1, G1, Ω1, γ2, G2, Ω2, γ3, G3, Ω3, γ4, G4, Ω4, γ5, G5, Ω5, γ6, G6, Ω6, ω) #1D Array
    
    for j in range(l):
        for i in range(n):
            T[j,i] = T_apx[i + j*n]
            
    return T
########## AUTO FIT BY LINE ########################

def auto_res(params, ω, t, fixed, mode):
    ω=ω
    if mode == 'kappa_int':
        
        κi = params[0]
        
        a = fixed[0]
        κe = fixed[1]
        φ = fixed[2]
        ωr = fixed[3]
        
    if mode == 'kappa_tot':
         
        κi = params[0]
        κe = params[1]
        
        a = fixed[0]
        φ = fixed[1]
        ωr = fixed[2]
        
    if mode == 'total':
        
        a = params[0]
        κi = params[1]
        κe = params[2]
        φ = params[3]
        ωr = params[4]
        
    myfunction = np.vectorize(X1_2f)
    
    return t - myfunction(a, κi, κe, ωr, φ, 0, 0, 0, ω)

def auto_fit(F,TR,params, mode = 'kappa_int', jac ='3-point', gtol = 1e-10, verbose=0):
    
    if mode == 'kappa_int':
        
        xf = np.array([params[0], params[2], params[3], params[4], 0, 0, 0, 0])
        
        constant_arguments = {'fixed': xf, 'mode': mode}
        x0 = params[1]
        
        result = sop.least_squares(auto_res, x0, jac=jac,
                                   args = [F,TR], kwargs = constant_arguments, gtol = gtol, verbose=verbose)
        
        param = np.append(result.x, 0.0)
        
    if mode == 'kappa_tot':
        
        xf = np.array([params[0], params[3], params[4], 0, 0, 0, 0])
        
        constant_arguments = {'fixed': xf, 'mode': mode}
        
        x0 = np.array([params[1], params[2]])
        
        result = sop.least_squares(auto_res, x0, jac=jac,
                                   args = [F,TR], kwargs = constant_arguments, gtol = gtol, verbose=verbose)
        
        param = result.x
    
    return param, result

def auto_fit_error(result, N, k):
    
    rms = np.sqrt(np.sum(result.fun**2)/(N - k))

    J = result.jac # Fit Jacobian matrix (not squared (m,n))
    H = np.matmul(J.transpose(),J) # Fit Hessian matrix (squared (m,m))
    cov = np.sqrt(np.linalg.inv(H)*rms/k) # Covariance matrix (squared(m,m))

    error = np.diagonal(cov)
    error = error[0]
    
    return error, rms