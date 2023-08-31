# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:48:14 2023

@author: Victor

Sample Quantum Data (sample_qdata).
This library consists in two classes, Sample and QData.
Sample is the class with all the Hamiltonian parameters for different samples.
QData is the class with all methods to diagonalize it, calculate and plot the energy levels and energy
transitions of the Hamiltonian after building it with class Sample.
      
# v0.2 - Minor changes in Sample class:
         Try - Exception paradigm added to give an error if user does not use string as a name
         Added an else condition to raise Exception if the name given by user does not
               correspond to a valid sample               
         Reworked QData class with new methods.
         NEW METHODS IN QDATA:
         qdata.eigenvalues() diagonalizes and calculates de eigenvalues and eigenstates of a Hamiltonian.
         qdata.elevels() calculates and plots the energy levels (using qdata.eigenvalues) as a function
               of external magnetic field from an initial magnetic field value Bi to a final magnetic
               field value Bf.
         qdata.transitions() caculates the energy transitions for a given Hamiltonian.

"""

import qutip
import numpy as np
                     
class sample():
    
    "Sample Class has all the information regarding to the Hamiltonian we are working with."
    
    def __init__(self, S,I,NS,NI,NJ,Ap,Az,p,Ix,Iy,Iz,Sx,Sy,Sz,ge,gi,E,kp,D,kz,Ms,Mi,ist_ab,density,B_steven,J,name):
        
        self.S = S # electronic spin
        self.I = I # nuclear spin
        self.NS = NS # Electronic Spin Dimension
        self.NI = NI # Nuclear Spin Dimension
        self.NJ = NJ # Total Spin Dimension
        self.Ap = Ap # Perpendicular Hyperfine constant
        self.Az = Az # Z - Hyperfine constant
        self.p = p # Quadrupolar Interaction constant
        self.Ix = Ix # Nuclear Spin X qutip matrix
        self.Iy = Iy # Nuclear Spin Y qutip matrix
        self.Iz = Iz # Nuclear Spin Z qutip matrix
        self.Sx = Sx # Electronic Spin X qutip matrix
        self.Sy = Sy # Electronic Spin Y qutip matrix
        self.Sz = Sz # Electronic Spin Z qutip matrix
        self.ge = ge # Electronic giromagnetic constant (Zeeman term)
        self.gi = gi # Nuclear giromagnetic constant (Zeeman term)
        self.E = E # Coupling induced by local strain 
        self.kp = kp # Coupling induced by electric fields
        self.D = D # Zero field splitting induced by local strain
        self.kz = kz # Zero field splitting induced by electric fields
        self.Ms = Ms # Electronic spin states vector
        self.Mi = Mi # Nuclear spin states vector
        self.ist_ab = ist_ab # Isotopical concentration
        self.density = density # Spin density
        self.B_steven = B_steven # Bij paramenters for high-order Oij operators
        self.J = J # Exchange interaction value
        self.name = name # crystal identifier
    
    @classmethod
    def load_sample(self, name):
        
        """
        Function to set sample parameters
        
        Parameters
        -------
        self: non-initialised class
        name: crystal identifier
        
        Returns
        -------
        self: initialised class
        """
        crystals = ['171Yb_Trensal','172Yb_Trensal','173Yb_Trensal',
                    'DPPH','NV_centers','HoW10','VOporf','CNTporf',
                    'MnBr_Ising']
        
        if type(name) != str:
            raise Exception('input must be a string')
            return None
        
        elif name == '171Yb_Trensal':
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
            
        elif name == '172Yb_Trensal':
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
            
        elif name == '173Yb_Trensal':
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
            
        elif name == 'DPPH':
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
            
        elif name == 'NV_centers':
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
        
        elif name == 'HoW10':
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
        
        elif name == 'MnBr':
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
            
            
        elif name == 'VOporf':
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
       
        elif name == 'CNTporf':
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
            
        elif name == 'MnBr_Ising':
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
            
        else:            
            warning = 'The input crystal type it is not in the database\n'
            print('\nPlease provide one of these crystal types as a string format:')
            for i in range(len(crystals)):
                print('----->' + crystals[i])
            print('\n')
            raise Exception(warning)
            return None
            
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
              
        return self(S,I,NS,NI,NJ,Ap,Az,p,Ix,Iy,Iz,Sx,Sy,Sz,ge,gi,E,kp,D,kz,Ms,Mi,ist_ab,density,B_steven,J,name)
    
    def hamiltonian(self, B_ext = np.array([0,0,0]), E_ext = np.array([0,0,0])):
        
        # This function calculates a general Hamiltonian given a set of values and arrays.
        # It returns a single array H.
        
        "ELECTRONIC ZEEMAN PARAMETERS"
        mu_B = 9.27401e-24 / 6.6261e-34 #Hz/T
        
        "PLUS AND MINUS SPIN OPERATORS"
        Sp = self.Sx + 1j*self.Sy
        Sm = self.Sx - 1j*self.Sy
        
        "SQUARED OPERATORS"
        S2=self.Sx*self.Sx+self.Sy*self.Sy+self.Sz*self.Sz
        #I2=sample.Ix*sample.Ix+sample.Iy*sample.Iy+sample.Iz*sample.Iz
        
        "ELECTRONIC ZEEMAN INTERACTION"
        ZE11 = mu_B*self.ge[0]*B_ext[0]*qutip.tensor(self.Sx,qutip.qeye(self.NI))
        ZE22 = mu_B*self.ge[1]*B_ext[1]*qutip.tensor(self.Sy,qutip.qeye(self.NI))
        ZE33 = mu_B*self.ge[2]*B_ext[2]*qutip.tensor(self.Sz,qutip.qeye(self.NI))
        
        
        "NUCLEAR ZEEMAN PARAMETERS"
      
        mu_N = 5.05078E-27 / 6.6261e-34 #Hz/T
        
        "NUCLEAR ZEEMAN INTERACTION"
        
        NZE11 = mu_N*self.gi*B_ext[0]*qutip.tensor(qutip.qeye(self.NS), self.Ix)
        NZE22 = mu_N*self.gi*B_ext[1]*qutip.tensor(qutip.qeye(self.NS), self.Iy)
        NZE33 = mu_N*self.gi*B_ext[2]*qutip.tensor(qutip.qeye(self.NS), self.Iz)
            
        "HYPERFINE INTERACTION"
        HF11 = self.Ap*qutip.tensor(self.Sx,self.Ix)
        HF22 = self.Ap*qutip.tensor(self.Sy,self.Iy)
        HF33 = self.Az*qutip.tensor(self.Sz,self.Iz)
        
        "QUADRUPOLAR INTERACTION NUCLEAR SPIN"
        Q33 = self.p*qutip.tensor(qutip.qeye(self.NS),self.Iz)**2
            
        "QUDRUPOLAR INTERACTION ELECTRONIC SPIN (ZERO FIELD SPLITTING TERM)"
        QE33 = (self.D + self.kz*E_ext[2])*qutip.tensor(self.Sz, qutip.qeye(self.NI))**2
        
        
        "STRAIN INTERACTION"
        STI = (self.E + self.kp*E_ext[0])*qutip.tensor(self.Sx, qutip.qeye(self.NI))**2 - (self.E + self.kp*E_ext[1])*qutip.tensor(self.Sy, qutip.qeye(self.NI))**2
            
            
        "HIGH ORDER TERMS"
        O20=self.B_steven['B20']*qutip.tensor(3*self.Sz**2-S2,qutip.qeye(self.NI))
        O40=self.B_steven['B40']*qutip.tensor(35*self.Sz**4-30*S2*self.Sz**2+25*self.Sz**2-6*S2+3*S2**2,qutip.qeye(self.NI))
        O44=self.B_steven['B44']*qutip.tensor(0.5*(Sp**4+Sm**4),qutip.qeye(self.NI))
        O60=qutip.tensor(231*self.Sz**6-315*S2*self.Sz**4+735*self.Sz**4+105*S2**2*self.Sz**2-
                          525*S2*self.Sz**2+294*self.Sz**2-5*S2**3+40*S2**2-60*S2,qutip.qeye(self.NI))
        O60=self.B_steven['B60']*O60
            
        Oij = O20 + O40 +  O44 + O60 
        
        "COMPLETE HAMILTONIAN"
        H = ZE11 + ZE22 + ZE33 + NZE11 + NZE22 + NZE33 + HF11 + HF22 + HF33 + Q33 + QE33 + STI + Oij
    
        return H

""" ####################################################################### """

#%%

""" ############################ QDATA ############################# """

class qdata():
    
    """
    class qdata @ qfunction stores and manages the information obtained after
    diagonalising the hamiltonian of a sample (the sample information is
    managed using class sample @ qfunctions)
    """
      
    def __init__(self, havl, have, freq, thf, thp):
        self.havl = havl # hamiltonian eigenvalues
        self.have = have # hamiltonian eigenvectors
        self.freq = freq # hamiltonian energy transitions
        self.thf = thf # hamiltonian thermal factors
        self.thp = thp # hamiltonian thermal polarizations
        
        
    @classmethod    
    def eigenvalues(self, smpl, B_ext):
        
        """
        Function to build and diagonalize the crystal Hamiltonian.
        
        Parameters
        -------
        smpl: a qfunction sample object carrying the information of the crystal Hamiltonian.
        B_ext: external (DC) magnetic field in molecule coordinate system.
        
        Returns
        -------
        self.havl: qdata value with the eigenstates of the Hamiltonian at a particular
                   external DC magnetic field.
        self.have: qdata value with the eigenvectors of the Hamiltonian at a particular
                   external DC magnetic field.
        """
        
        if type(smpl) != sample:
            raise Exception('1st input must be a qfunctions sample class variable')
            return None
            
        # Hamiltonian calculation
        H = smpl.hamiltonian(B_ext)
        
        # Eigenstates calculation
        HD = H.eigenstates()
        
        # HD has the eigenvalues in the first component (0) and the
        # eigenvectors in the second component (1)
        self.havl = HD[0] # Hz
        self.have = HD[1] # No units
        
        return self.havl, self.have
    
    @classmethod
    def elevels(self, smpl, Bi=0.0, Bf=0.1, n=101, angles = {'angle': 0., 'fi': 0., 'theta': 0.},
                plot = False, store = False, xmax = 0, xmin = 0, ymax = 0, ymin = 0, fmt = 'png',
                verbosity = False):
        
        B = np.array([0.,0.,0.])
        Bplot = np.zeros(n)
        BP = abs(Bf-Bi)/n
        Bmod = Bi
        
        M = rotationmatrix(angles['fi']*np.pi/180, angles['theta']*np.pi/180, verbosity = verbosity)
        
        Havl = np.zeros((n,smpl.NJ))
        
        for l in range(n):   
            
            # Updating B coordinates
            B[0] = Bmod*np.sin(angles['angle']) # T
            B[1] = 0
            B[2] = Bmod*np.cos(angles['angle'])  # T
            
            B_mol = M.dot(B) #updating B in molecular axes
            
            if verbosity == True:
                print('Lab coord sys:')
                print(B)
                print('Mol coord sys:')
                print(B_mol)
                            
            Havl[l,:],_ = qdata.eigenvalues(smpl,B_mol)
            Bplot[l] = Bmod
            Bmod = Bmod + BP # Actual magnetic field module
            
        if plot == True:
            
            if verbosity == True:
                print('Plotting energy levels')
                
            plt.figure('Energy levels ' + smpl.name)            
            for i in range(smpl.NJ):
                plt.plot(Bplot*1e3, Havl[:,i]*1e-9)
 
            plt.xlabel(r'$\mu_0$' + 'H (mT)' , fontsize = 14, fontweight='bold')
            plt.xticks(fontsize = 14)
            plt.ylabel('Frequency (GHz)' , fontsize = 14, fontweight='bold')
            plt.yticks(fontsize = 14)
            if xmax == xmin: plt.xlim(Bi*1e3,Bf*1e3)
            else: plt.xlim(xmin,xmax)
            if ymax != ymin: plt.ylim(ymin,ymax)
            plt.locator_params(axis='y', nbins=5)
            plt.tick_params(direction = 'in')
            plt.ticklabel_format(useOffset=False)
            plt.title('Energy levels ' + smpl.name, fontsize = 14, fontweight='bold')
            text = r"$\theta$" + ' = %.2f deg \n'%(angles['theta']) + r"$\varphi$" + ' = %.2f deg'%(angles['fi'])
            plt.figtext(1.575, 0.225, text, fontsize = 12, bbox = dict(facecolor='white', edgecolor='black', boxstyle='round', alpha = 0.15))
            
            if store == True:
                
                dirpath = pth.folder() + '/' + smpl.name
                print('Saving plot at %s' %dirpath)
                dirpath = dirpath + '/Energies/'
                file_name = 'Elevels_Theta%.2fdeg_Fi%.2fdeg.' %(angles['theta'],angles['fi'])
                Arr = np.zeros((n,smpl.NJ+1))
                Arr[:,0] = Bplot
                Arr[:,1:] = Havl
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                    
                plt.savefig(dirpath + file_name + fmt, dpi = 2048, bbox_inches='tight') 
                np.savetxt(dirpath + file_name +'txt' , Arr, fmt='%s') 
                    
            plt.show()
                    
        return Havl, Bplot


    @classmethod
    def transitions(self, smpl, Bi=0.0, Bf=0.1, n=101, angles = {'angle': 0., 'fi': 0., 'theta': 0.},
                plot = False, store = False, xmax = 0, xmin = 0, ymax = 0, ymin = 0, fmt = 'png',
                verbosity = False, index = [], experimental = {}, lw=2.5, legend=True):
        
        Havl, Bplot = qdata.elevels(smpl, Bi, Bf, n, angles = angles,
                    xmax = xmax, xmin = xmin, ymax = ymax, ymin = ymin,
                    fmt = 'png', verbosity = verbosity)
        
        freqs = np.zeros((n, smpl.NJ*smpl.NJ))
        
        for i in range(smpl.NJ):
            for j in range(smpl.NJ):
                    freqs[:,i*smpl.NJ+j] = Havl[:,j] - Havl[:,i]
        
        if plot == True:
            
            if verbosity == True:
                print('Plotting energy transitions')
                
            plt.figure('Energy transitions ' + smpl.name)
            
            if index == []:
                for i in range(smpl.NJ*smpl.NJ):
                    plt.plot(Bplot*1e3, freqs[:,i]*1e-9)
            else:
                if type(index)!=list:
                    raise Exception('please provide the transitions in a list format \n')
                else:
                    for i in range(len(index)):
                        label='%i -> %i'%(index[i][0],index[i][1])
                        plt.plot(Bplot*1e3, freqs[:,index[i][0]*smpl.NJ+index[i][1]]*1e-9,
                                 lw=lw,label=label)
                        
            if experimental.keys() != []:
                n = 0
                for i in experimental.keys():
                    if i == 'frequencies':
                        for j in experimental['frequencies']:
                            plt.hlines(j*1e-9, -1000, 1000, color = 'gray', linestyles = 'dashed') #
                    else:
                        f = np.zeros(len(experimental[i]))
                        for j in range(len(f)): f[j] = experimental['frequencies'][n]       
                        plt.scatter(np.array(experimental[i])*1e3, f*1e-9, s = 80, facecolors='gray', edgecolors='k', zorder=3)
                        n=n+1           
 
            plt.xlabel(r'$\mu_0$' + 'H (mT)' , fontsize = 14, fontweight='bold')
            plt.xticks(fontsize = 14)
            plt.ylabel('Frequency (GHz)' , fontsize = 14, fontweight='bold')
            plt.yticks(fontsize = 14)
            if xmax == xmin: plt.xlim(Bi*1e3,Bf*1e3)
            else: plt.xlim(xmin,xmax)
            if ymax != ymin: plt.ylim(ymin,ymax)
            plt.locator_params(axis='y', nbins=5)
            plt.tick_params(direction = 'in')
            plt.ticklabel_format(useOffset=False)
            if legend == True: plt.legend()
            plt.title('Energy transitions ' + smpl.name, fontsize = 14, fontweight='bold')
            text = r"$\theta$" + ' = %.2f deg \n'%(angles['theta']) + r"$\varphi$" + ' = %.2f deg'%(angles['fi'])
            plt.figtext(1.0, 0.8, text, fontsize = 12, bbox = dict(facecolor='white', edgecolor='black', boxstyle='round', alpha = 0.15))
            
            if store == True:
                
                dirpath = pth.folder() + '/' + smpl.name
                print('Saving plot at %s' %dirpath)
                dirpath = dirpath + '/Energies/'
                file_name = 'Etransitions_Theta%.2fdeg_Fi%.2fdeg.' %(angles['theta'],angles['fi'])
                Arr = np.zeros((n,smpl.NJ*smpl.NJ+1))
                Arr[:,0] = Bplot
                Arr[:,1:] = freqs
                if not os.path.exists(dirpath):
                    os.makedirs(dirpath)
                    
                plt.savefig(dirpath + file_name + fmt, dpi = 2048, bbox_inches='tight') 
                np.savetxt(dirpath + file_name +'txt' , Arr, fmt='%s') 
                    
            plt.show()
                    
        return freqs, Bplot
