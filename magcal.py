# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:01:16 2023

@author: VRG@HQDlab

# Description: library for magnetic field calculation using CUDA.

# v1.2   - Huge performance boost in SegCurrent2Field() CUDA function by 
           implementing loop unrolling and avoiding definition of variables
           by silent threads.
           Fixed time prediction in gpu_field().
           Fixed bdim bug (it was set at (8,8,8)) in the last calculation step
           for segmented calculation.
           Added _total_time() function.
           Improved information messages during execution.
           Removed gpu_field() function and renamed gpu_fieldv2() as gpu_field().
           Moved pth class (by DR@SuperTechCAB) to a loadpath.py library.
           Documented some functions and classes.
           
ToDo: Fix Current2Field() function for small size problems.
      Invert color scale in xz_cut() and yz_cut() methods.
      Finish _total_time() function to implement correct time prediction.
      Optimize deflattening loop at the end in gpu_field()
                       
"""

import numpy as np
import scipy.io as sio
import math
import h5py
import pycuda as cuda
from pycuda import compiler, gpuarray
import pycuda.autoinit
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import time
import getpass
import json
from loadpath import pth
from datetime import datetime

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
    
    
    def mod_plot(field, zc=0, cmap = 'RdBu_r', save = False, fmt = '.png', units = 'nT', linthresh = 1.0, title = False):
        
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
        plt.pcolor(X, Y, GR, norm=colors.SymLogNorm(linthresh=linthresh, vmin=np.min(GR), vmax=np.max(GR)) ,  cmap=plt.get_cmap(cmap))
        plt.colorbar(label='B (%s)' %(units))
        plt.title('$ \ B \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_module_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        return
    
    def vec_plot(field, zc=0, cmap = 'RdBu_r', save = False, fmt = '.png', units = 'nT', linthresh = 1.0, title = False, subplot = False,
                 figsize = (8.0, 4.8), clabel_pos=[0.15, -0.1, 0.7, 0.05]):
        
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
        
        vmax = np.max(np.array([GRx, GRy, GRz]))
        vmin = np.min(np.array([GRx, GRy, GRz]))
        
        
        if not subplot:
            
            plt.figure()        
            plt.pcolor(X, Y, GRx, norm=colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax) ,  cmap=plt.get_cmap(cmap))
            plt.colorbar(label='B (%s)' %(units))
            if title: plt.title('$ \ B$_x$ \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
            plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
            plt.tick_params(direction = 'in')
            if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_Xcomponent_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
            plt.show()
            
            plt.figure()        
            plt.pcolor(X, Y, GRy, norm=colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax) ,  cmap=plt.get_cmap(cmap))
            plt.colorbar(label='B (%s)' %(units))
            if title: plt.title('$ \ B$_y$  \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
            plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
            plt.tick_params(direction = 'in')
            if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_Ycomponent_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
            plt.show()
            
            plt.figure()        
            plt.pcolor(X, Y, GRz, norm=colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax) ,  cmap=plt.get_cmap(cmap))
            cb=plt.colorbar(orientation="vertical")
            cb.set_label(label='B (%s)' %(units), size='large', weight='bold')
            if title: plt.title('$ \ B$_z$  \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[zc]*1e6), fontsize = 14)
            plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
            plt.tick_params(direction = 'in')
            if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_Zcomponent_zc%.2fum_' + cmap + fmt %(field.posz[zc]*1e6), dpi = 1024, bbox_inches='tight')
            plt.show()
            
        else:
                       
            plt.figure(figsize=figsize)
            fig, (ax1,ax2,ax3) =  plt.subplots(1, 3, sharey=False, figsize=figsize)
            ax1.pcolor(X, Y, GRx, norm=colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax) ,  cmap=plt.get_cmap(cmap))
            ax1.set_xlabel('x ($\mu$m)', fontsize = 14.0, weight = 'bold'); ax1.set_ylabel('y ($\mu$m)', fontsize = 14.0, weight = 'bold')
            z2 = ax2.pcolor(X, Y, GRy, norm=colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax) ,  cmap=plt.get_cmap(cmap))
            ax2.set_xlabel('x ($\mu$m)', fontsize = 14.0, weight = 'bold'); ax2.set_ylabel('y ($\mu$m)', fontsize = 14.0, weight = 'bold')
            ax3.pcolor(X, Y, GRz, norm=colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax) ,  cmap=plt.get_cmap(cmap))
            ax3.set_xlabel('x ($\mu$m)', fontsize = 14.0, weight = 'bold'); ax3.set_ylabel('y ($\mu$m)', fontsize = 14.0, weight = 'bold')
            
            cbar_ax = fig.add_axes(clabel_pos) 
            # add_axes([xmin,ymin,dx,dy])
            # xmin, ymin = base point
            # dx, dy = size params
            cb = fig.colorbar(z2, cax=cbar_ax, orientation='horizontal')
            cb.set_label(label='B (%s)' %(units), size='large', weight='bold')
            ax1.tick_params(direction = 'in')
            ax2.tick_params(direction = 'in')
            ax3.tick_params(direction = 'in')
            
            plt.show()
                
        return
    
    def xz_cut(field, yc, cmap = 'RdBu', save = False, fmt = '.png', linthresh = 1.0, title = False):        
        
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
        plt.pcolor(X, Y, GR*1e9, norm=colors.SymLogNorm(linthresh=linthresh, vmin=np.min(field.Brf[:,:,0]*1e9), vmax=np.max(field.Brf[:,:,0]*1e9)) ,  cmap=plt.get_cmap('RdBu_r'))
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
    

    def yz_cut(field, xc, cmap = 'RdBu', save = False,  fmt = '.png', linthresh = 1.0, title = False):        
    
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
        plt.pcolor(X, Y, GR*1e9, norm=colors.SymLogNorm(linthresh=linthresh, vmin=np.min(field.Brf[:,:,0]*1e9), vmax=np.max(field.Brf[:,:,0]*1e9)) ,  cmap=plt.get_cmap('RdBu_r'))
        plt.colorbar(label='B (nT)')
        plt.title('$ \ B \ XY \ plane \ z_{cut} \ = \ %i \ \mu m$' %(field.posz[0]*1e6), fontsize = 14)
        plt.xlabel('x ($\mu$m)',fontsize = 14); plt.ylabel('y ($\mu$m)' , fontsize = 14)
        plt.tick_params(direction = 'in')
        plt.axvline(field.posx[xc]*1e6, lw=2.0, color='white', linestyle='dashed')
        if save: plt.savefig(dirpath + '/Bmodule_map_XYslice_xc%.2fum_' + cmap + fmt %(field.posx[xc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
        
        GR = field.Brf[:,xc,:]
        
        plt.figure()
        plt.pcolor(Y, Z, GR.transpose(), norm=colors.LogNorm(vmin=GR.min(), vmax=GR.max()) , cmap = cmap )
        plt.colorbar(label='T');
        plt.title('|B| YZ xcut = %.2e $\mu$m' %(field.posx[int(xc)]*1e6))
        plt.xlabel('y ($\mu$m)',fontsize = 14); plt.ylabel('z ($\mu$m)' , fontsize = 14)
        plt.ylim((np.amin(Z),np.amax(Z)))
        plt.tick_params(direction = 'in')
        if save: plt.savefig(dirpath + '/Bmodule_map_YZslice_xc%.2fum_' + cmap + fmt %(field.posx[xc]*1e6), dpi = 1024, bbox_inches='tight')
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
        if save: plt.savefig(dirpath + '/B_contour_YZslice_xc%.2fum_' + cmap + fmt %(field.posx[xc]*1e6), dpi = 1024, bbox_inches='tight')
        plt.show()
    
        return

    def crystal_base(field, axis=np.array([0,0,1])):
        
        zangle = angle(axis,np.array([0,0,1]))
        xangle = angle(axis,np.array([1,0,0]))
        
        M = rotationmatrix(xangle, zangle)
        
        Brfx = M[0,0]*field.Brfx + M[0,1]*field.Brfy + M[0,2]*field.Brfz
        Brfy = M[1,0]*field.Brfx + M[1,1]*field.Brfy + M[1,2]*field.Brfz
        Brfz = M[2,0]*field.Brfx + M[2,1]*field.Brfy + M[2,2]*field.Brfz
        
        field.Brfx = Brfx
        field.Brfy = Brfy
        field.Brfz = Brfz
        
        return field 
    
def loadfield(path, angles = {'angle': 0., 'fi': 0., 'theta': 0.}):
    
    # This program reads the stored magnetic field matrices generated by the GPU
    "LOADING THE FILE"
    
    # LOAD MATRICES FROM .MAT FILE
    Aux = sio.loadmat(path)
    Aux = Aux['field'] #T and m
        
    H2DZ = Aux['H2DX'][(0)][(0)] #T
    H2DY = Aux['H2DY'][(0)][(0)] #T
    H2DX = -1*Aux['H2DZ'][(0)][(0)] #T
            
          
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

def _getGPUparams():
    cuda.driver.init()
    print("%d device(s) found." % cuda.driver.Device.count())

    for ordinal in range(cuda.driver.Device.count()):
        dev = cuda.driver.Device(ordinal)
        print('Device #%d: %s' % (ordinal, dev.name()))
        mtpb = dev.get_attribute(cuda.driver.device_attribute.MAX_THREADS_PER_BLOCK)
        mbdx = dev.get_attribute(cuda.driver.device_attribute.MAX_BLOCK_DIM_X)
        mbdy = dev.get_attribute(cuda.driver.device_attribute.MAX_BLOCK_DIM_Y)
        mbdz = dev.get_attribute(cuda.driver.device_attribute.MAX_BLOCK_DIM_Z)
        mgdx = dev.get_attribute(cuda.driver.device_attribute.MAX_GRID_DIM_X)
        mgdy = dev.get_attribute(cuda.driver.device_attribute.MAX_GRID_DIM_X)
        mgdz = dev.get_attribute(cuda.driver.device_attribute.MAX_GRID_DIM_X)
        print('Max threads per block: %s' % mtpb)
        print('Max dimensions of a thread block (x,y,z): (%s, %s, %s)' % (mbdx, mbdy, mbdz))
        print('Max dimensions of a grid (x,y,z): (%s, %s, %s)' % (mgdx, mgdy, mgdz))
        print()

    return [mbdx, mbdy, mbdz], dev.name()

def _total_time(tic, toc, device_id, slides, elements):
    
    total_time = toc - tic
    Minutes, Hours = math.modf(total_time/3600)
    Seconds, Minutes = math.modf(Minutes*60)
    print('Total time: %i h, %i min, %.2f sec.' %(Hours, Minutes, Seconds))
    
    time_data = {}
    time_data['date'] = datetime.today().strftime('%Y-%m-%d')
    time_data['device_id'] = device_id
    time_data['slides'] = float(slides)
    time_data['elements'] = float(elements)
    time_data['lapse'] = [Hours, Minutes, Seconds]
    
    username = getpass.getuser()
    dirpath = 'C:/Users/' + username + '/.magcal/'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
    with open(dirpath + device_id, 'w') as f:
        json.dump(time_data, f, indent=4)
        
    return

#%% MAGNETIC FIELD CUDA AND PYCUDA FUNCTIONS

"""################### PRECOMPILATION BLOCK SIZE SETTINGS ##################"""

max_block_dim, device_id = _getGPUparams()

# Define block size for dynamic block size setting
block_size_map = {
    1024: (16, 8, 8),  # 1024 threads per block (computing 1x)
    512: (8, 8, 8),    # 512 threads per block (computing 1x)
}

bdim = block_size_map.get(max_block_dim[0])
if bdim is None:
    print('Unexpected block size')
    exit()  # Terminate the program since block size is unexpected

BLOCK_SIZE = bdim[0] * bdim[1] * bdim[2]

"""##################### CUDA CODE FOR GPU_FIELD FUNCTION ##################"""

mod = compiler.SourceModule("""

    #include<math.h>
    #include <stdio.h>
    
    #define BLOCK_SIZE %d
    
    __global__ void Current2Field(float *Ixrs, float *Iyrs, float *H2D,
                                  float *posx, float *posy, float *posz,
                                  int X, int Y, int Z)
    {
       int idx = threadIdx.x + blockDim.x * blockIdx.x;
       int idy = threadIdx.y + blockDim.y * blockIdx.y;
       int idz = threadIdx.z + blockDim.z * blockIdx.z;
       
       if ((idx < X) && (idy < Y) && (idz < Z))
       {
            float dx = posx[idx + X * idy + X * Y * idz];
            float dy = posy[idx + X * idy + X * Y * idz];
            float dz = posz[idx + X * idy + X * Y * idz];
    
            float H1 = 0;
            float H2 = 0;
            float H3 = 0;
    
            for (int i = 0; i < X * Y; i += 4)
            {
                
                for (int j = 0; j < 4; j++)
                {
                    float ijdx =  dx - posx[i + j];
                    float ijdy =  dy - posy[i + j];
        
                    float modr = sqrtf(ijdx * ijdx + ijdy * ijdy + dz * dz);
                    float imodr = 1 / modr;
        
                    H1 += Iyrs[i + j] * dz * (imodr * imodr * imodr);
                    H2 += -Ixrs[i + j] * dz * (imodr * imodr * imodr);
                    H3 += (Ixrs[i + j] * ijdy - Iyrs[i + j] * ijdx) * (imodr * imodr * imodr);
    
                }
            }
             
            H2D[idx + X * idy + X * Y * idz] = H1;
            H2D[idx + X * idy + X * Y * idz + X * Y * Z] = H2;
            H2D[idx + X * idy + X * Y * idz + 2 * X * Y * Z] = H3;
           
           
        }
        
     } 
            
    __global__ void SegCurrent2Field(float *Ixrs, int DX, float *Iyrs, int DY,
                                         float *H2D, float *xi, float *yi, float *posx,
                                         float *posy, float *posz, int *loop, int X, int Y, int Z)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        int idy = threadIdx.y + blockDim.y * blockIdx.y;
        int idz = threadIdx.z + blockDim.z * blockIdx.z;
        
        if ((idx < X) && (idy < Y) && (idz < Z))
        {
            float dx = posx[idx + X * idy + X * Y * idz];
            float dy = posy[idx + X * idy + X * Y * idz];
            float dz = posz[idx + X * idy + X * Y * idz];
    
            float H1 = 0;
            float H2 = 0;
            float H3 = 0;
    
            for (int i = 0; i < DX * DY; i += 4)
            {
                
                for (int j = 0; j < 4; j++)
                {
                    float ijdx =  dx - xi[i + j];
                    float ijdy =  dy - yi[i + j];
        
                    float modr = sqrtf(ijdx * ijdx + ijdy * ijdy + dz * dz);
                    float imodr = 1 / modr;
        
                    H1 += Iyrs[i + j] * dz * (imodr * imodr * imodr);
                    H2 += -Ixrs[i + j] * dz * (imodr * imodr * imodr);
                    H3 += (Ixrs[i + j] * ijdy - Iyrs[i + j] * ijdx) * (imodr * imodr * imodr);
    
                }
            }
        
        
            if (loop[idx + X * idy + X * Y * idz] == 0)
            {
                H2D[idx + X * idy + X * Y * idz] = H1;
            }
            else if (loop[idx + X * idy + X * Y * idz] == 1)
            {
                H2D[idx + X * idy + X * Y * idz] = H2;
            }
            else if (loop[idx + X * idy + X * Y * idz] == 2)
            {
                H2D[idx + X * idy + X * Y * idz] = H3;
            }    
            
        }
    }
                                                          
""" % BLOCK_SIZE)

"""#########################################################################"""


"""############################### PYCUDA CODE #############################"""

def gpu_field(Ixrs,Iyrs,posx,posy,posz):
    
    """
    Function to calculate the 3D magnetic field from a 2D density current
    distribution generated by sonnet. The z component of the current is assumed
    as zero so the current is in a plane and given by a 2D matrix. The magnetic
    field is calculated in the volume over that current distribution, so 
    the magnetic field data is returned in a 3D matrix.
    
    Parameters
    -------
    Ixrs: numpy 2D array. x component of the density current.
    Iyrs: numpy 2D array. y component of the density current.
    posx: numpy 1D array. x component of the spatial information of the problem.
    posy: numpy 1D array. y component of the spatial information of the probkem.
    posz: numpy 1D array. z component of the spatial information of the problem.
    
    Returns
    -------
    Field: field class structure. it contains the three components of the
           calculated magnetic field (Brfx, Brfy, Brfz), the module of the
           magnetic field (Brf), the spatial information of the problem (posx,
           posy, posz, dx, dy, dz).
    """
    tic_0 = time.time()
    
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
    
    print('Creating void 3D array to store solution . . . \n')
    
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
    
    max_block_dim, device_id = _getGPUparams()
    DimS = max_block_dim[0] * max_block_dim[1] * max_block_dim[2]
    
    if ( Dim  <= DimS ):
        DimC = 1
    else:
        # Divide the problem in 512x512x32 flat arrays.
        # This is only in computing terms, is not an actual rearrangement.
        # Dimension of the smaller arrays:
        # Subarray counter:
        DimC = int(math.ceil(Dim/(DimS)))
        
    if DimC == 1: # No division
        
        print('Initializing normal kernel')
        
        # Get the kernel function from the precompiled module.
        # Important! the compilation of the CUDA kernel will be performed during the import step, not during execution.
        func = mod.get_function("Current2Field")
        
        print('The 3D problem has ' + str(DimX*DimY*3*DimZ) + ' elements . . .')
        
        H2D_gpu = gpuarray.to_gpu(H2D)
        posz_gpu = gpuarray.to_gpu(posz3d)
    
        "DEFINING BLOCK SIZE"
        if max_block_dim[0] == 1024:
            bdim = (16,8,8) # 1024 threads per block (computing 1x)
        elif max_block_dim[0] == 512:
            bdim = (8,8,8) # 1024 threads per block (computing 1x)
        else:
            print('unexpected block size')
                    
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
        func(Ixrs_gpu, Iyrs_gpu, H2D_gpu, X_gpu, Y_gpu, posz_gpu, np.int32(DimX),
             np.int32(DimY), np.int32(DimZ), block = bdim, grid=gdim)
        
        H2D = H2D_gpu.get()
        H2D = H2D*step*1e-7
        posz_gpu.gpudata.free()
        
    else:
        
        "DEFINING INDIVIDUAL MATRICES DIMENSION"
        DimXi = max_block_dim[0]
        DimYi = max_block_dim[1]
        DimZi = max_block_dim[2]
        
        print('Initializing segmented kernel')
        
        # Get the segmented kernel function from the precompiled module.
        # Important! the compilation of the CUDA kernel will be performed
        # during the import step, not during execution.
        func = mod.get_function("SegCurrent2Field")
        
        print('Problem divided in ' + str(DimC) + ' slices . . .')
        print('Each slice has ' + str(DimS) + ' elements . . .')
        
        
        "DEFINING BLOCK SIZE"
        if max_block_dim[0] == 1024:
            bdim = (16,8,8) # 1024 threads per block (computing 1x)
        elif max_block_dim[0] == 512:
            bdim = (8,8,8) # 1024 threads per block (computing 1x)
        else:
            print('unexpected block size')
                       
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
        
        for i in range(DimC):            
            
            if i == DimC - 1:
                
                print('Starting the last loop')
                
                Remaining_time = (DimC - i)*Loop_time

                Hours, Seconds = divmod(Remaining_time, 3600)
                Minutes, Seconds = divmod(Seconds, 60)
                
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

                DimXii = DimXi
                DimYii = DimYi
                DimZii = DimZi

                msize = (DimXii, DimYii, DimZii)
                
                "DEFINING BLOCK SIZE"
                if max_block_dim[0] == 1024:
                    bdim = (16,8,8) # 1024 threads per block (computing 1x)
                elif max_block_dim[0] == 512:
                    bdim = (8,8,8) # 1024 threads per block (computing 1x)
                else:
                    print('unexpected block size')
                                
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
                                          
                func(Ixrs_gpu, np.int32(DimX), Iyrs_gpu, np.int32(DimY), H2D_gpu,
                     X_gpu, Y_gpu, posx_gpu, posy_gpu, posz_gpu, loop_gpu,
                     np.int32(DimXii), np.int32(DimYii), np.int32(DimZii),
                     block = bdim, grid=gdim)
                
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

                    Hours, Seconds = divmod(Remaining_time, 3600)
                    Minutes, Seconds = divmod(Seconds, 60)
                                    
                    print('Remaining time: ' + str(Hours) + ' h, ' + str(Minutes) + ' min, ' + str(round(Seconds,2)) + ' sec.')
                    
                tic = time.time();
                
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
                
                func(Ixrs_gpu, np.int32(DimX), Iyrs_gpu, np.int32(DimY), H2D_gpu,
                     X_gpu, Y_gpu, posx_gpu, posy_gpu, posz_gpu, loop_gpu,
                     np.int32(DimXi), np.int32(DimYi), np.int32(DimZi),
                     block = bdim, grid=gdim)
            
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
                toc = time.time();
                
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
    
    toc_1 = time.time()
    
    _total_time(tic_0, toc_1, device_id, DimC, Dim)
    
    return Field(Brfx,Brfy,Brfz,Brf,posx,posy,posz,dx,dy,dz)

def angle(vector1, vector2):
    
    """
    Function to know angle between two vectors
    
    Parameters
    -------
    vector1: numpy array 1st vector
    vector2: numpy array 2nd vector
    
    Returns
    -------
    angle: float angle between the two vectors
    """
    
    vector1 = vector1/np.linalg.norm(vector1) # normalisation
    vector2 = vector2/np.linalg.norm(vector2) # normalisation
    angle = np.arccos(np.dot(vector1,vector2)) # rad
    
    return angle

def rotationmatrix(fi=0., theta=0., verbosity = False):
    
    """
    Function to rotate a vector in 3D
    
    Parameters
    -------
    fi: angle with the x axis (lab)
    theta: angle with the z axis (lab)
    x and z are in the sample plane
    
    (spherical coordinate system)
    
    verbosity: if you want to print calculation information
    
    Returns
    -------
    M: numpy array
        Rotation matrix // Base change matrix
    """
    
    # Molecule axes in lab coordinates
    Z_mol = np.array([np.cos(fi)*np.sin(theta), np.sin(fi)*np.sin(theta), np.cos(theta)])
    
    X_mol = np.cross([0, 1, 0], Z_mol)
    X_mol = X_mol/np.linalg.norm(X_mol)

    Y_mol = np.cross(Z_mol,X_mol)
    Y_mol = Y_mol/np.linalg.norm(Y_mol)
    
    # Base change matrix:
    M = np.zeros((3,3))
    M[:,0] = X_mol
    M[:,1] = Y_mol
    M[:,2] = Z_mol
    M = np.linalg.inv(M)
    
    if verbosity == True:
        print('Z_mol norm:')
        print(np.linalg.norm(Z_mol))
        print(Z_mol)
        print('X_mol norm:')
        print(np.linalg.norm(X_mol))
        print(X_mol)
        print('Y_mol norm:')
        print(np.linalg.norm(Y_mol))
        print(Y_mol)
        print(np.arccos(np.dot(np.array([0,0,1]), Z_mol))*180/np.pi)
        print(M)
        
    return M
