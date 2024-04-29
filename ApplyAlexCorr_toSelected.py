# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 00:42:42 2024

@author: glham
#For quick application of correction parameters to non-ALEX traces
#applies Ia=Ia-alpha*Id
#Id=Id*gamma
beta not applicable to fret efficiency
delta only used if DirAccAtEnd
Approximates delta correction (-delta*Iaa)
Does so by averaging direct acceptor frames
"""

from scipy.io import loadmat, savemat
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch as fn


directory = r'C:\Users\Sudipta Lahiri\Desktop\ALEX corrected Data\RADX_FBH1 Controls\BRCA2-RAD51 with FBH1\Exp1\BRCA2-RAD51 with FBH1\DATA\selected_1' #folder containing the selected traces
save_subfolders = ['Corrected traces', 'DATA', 'selected_1'] #subfolders to navigate into, in order, to save resulsts

alpha = 0.034
beta = 0.581
gamma = 0.497
delta = 0.136

smooth_output=True #smooths using a running mean, shortens the trace by window-1 at beginning and end
smoothing_window = 5 #must be odd

DirAccAtEnd=False #if there is a window of direct acceptor excitation at the end of each trace
AccFrames=100 #where acceptor excitation begins. uses only first half this range to mitigate acceptor bleaching influence

pattern = 'spool*mol_*_tr_*.dat' #file pattern for opening traces. For now assumes files have been converted to .dat format used in ebFRET processing
app='corrected_' #what to append to saved corrected files at beginning

os.chdir(directory)

for i in save_subfolders:
    try:
        os.mkdir(i)
    except:
        pass
    os.chdir(i)
os.chdir(directory)
full_save_dir=''
for i in save_subfolders:
    full_save_dir = full_save_dir + i + '\\'

for file in os.listdir('.'):
    if not fn.fnmatch(file, pattern):
        continue
    data=np.loadtxt(file, skiprows=1)
    if smooth_output: #do smoothing
        data_smooth = data[:-(smoothing_window-1)]
        data_smooth[:,0] = data[:-(smoothing_window-1),0]
        data_smooth[:,1] = np.convolve(data[:,1], np.ones(smoothing_window)/smoothing_window, mode = 'valid')
        data_smooth[:,2] = np.convolve(data[:,2], np.ones(smoothing_window)/smoothing_window, mode = 'valid')
        data_smooth[:,3] = np.convolve(data[:,3], np.ones(smoothing_window)/smoothing_window, mode = 'valid')
        data = data_smooth
        AccFrames = AccFrames - int((smoothing_window-1)/2)

    if DirAccAtEnd:
        AO=np.mean(data[-AccFrames:,2]) #if direct acceptor is present, use this for delta correction
    else:
        AO=0.
    data[:,2]=data[:,2]-alpha*data[:,1]-delta*AO #apply alpha, delta
    data[:,1]=gamma*data[:,1] #apply gamma correction

    data[:,3]=data[:,2]/(data[:,1]+data[:,2]) #recalculate alpha, gamma, delta corrected FRET efficiency



    np.savetxt(full_save_dir + app + file, data, fmt='%.6f', header='time(s) Donor Acceptor FRET', delimiter=' ') #save output







