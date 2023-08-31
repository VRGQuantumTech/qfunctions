# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:48:14 2023

@author: Victor

Sample Quantum Data (sample_qdata).
This library consists in two classes, Sample and QData.
Sample is the class with all the Hamiltonian parameters for different samples.
QData is the class with all methods to diagonalize the Hamiltonian after building
it with class Sample.
      
# v0.1 - Work in progress.

"""

import qutip
import numpy as np

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
