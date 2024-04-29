# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:56:12 2024

@author: glham

#works with trajectories from MainBatch_FRET
#requires a donor only and an acceptor only dataset. 
#output is alpha and delta parameters
"""

import numpy as np
#import pickle 
#import pickle5 as pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import GridSpec
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from scipy.io import loadmat
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max


DOfolder=r'D:\Data\01302024\Try_Again\Donor only'
AOfolder=r'D:\Data\01302024\Try_Again\Acceptor only'
Corrections_Folder = r'D:\Scripting\FRET\SL_GH_ALEX_Standards'

DOimages=7   #number of images expected in the folder
AOimages=7   #number of images expected in the folder

smoothing = False  #if trajectories are to be smoothed for calculations
window = 5  #sliding window size for average smoothing
smooth = lambda array: np.convolve(array, np.ones(window)/window, mode='valid')


alldata={'DO': {}, 'AO': {}} #data dictionary
#%%load DO data, compute basic parameters
os.chdir(DOfolder)
alldata['DO']['DD']=[]
alldata['DO']['AA']=[]
alldata['DO']['DA']=[]
alldata['DO']['E']=[]
alldata['DO']['S']=[]
for i in range(DOimages):
    if i==0:

        datatemp=loadmat(r'spool_sorted_FRET_Left.mat', simplify_cells=True)['FRET_pairs']   #works with mat files exported by MatLab trace extraction script
    else:
        datatemp=loadmat(r'spool_{0}_sorted_FRET_Left.mat'.format(i), simplify_cells=True)['FRET_pairs']
    alldata['DO']['DD'].extend([i['LeftID'] for i in datatemp])
    alldata['DO']['AA'].extend([i['RightIA'] for i in datatemp])
    alldata['DO']['DA'].extend([i['RightID'] for i in datatemp])
alldata['DO']['DD']=np.array(alldata['DO']['DD']).T
alldata['DO']['AA']=np.array(alldata['DO']['AA']).T
alldata['DO']['DA']=np.array(alldata['DO']['DA']).T    

if smoothing:
    for i in ['DD', 'AA', 'DA']:
        alldata['DO'][i] = np.apply_along_axis(smooth, 0, alldata['DO'][i])
        
        
alldata['DO']['E']=(alldata['DO']['DA']/(alldata['DO']['DD']+alldata['DO']['DA'])).T
alldata['DO']['S']=((alldata['DO']['DA']+alldata['DO']['DD'])/(alldata['DO']['DD']+alldata['DO']['DA']+alldata['DO']['AA'])).T
alldata['DO']['E'][np.isnan(alldata['DO']['E'])]=0
alldata['DO']['S'][np.isnan(alldata['DO']['S'])]=0
alldata['DO']['E'][np.isinf(alldata['DO']['E'])]=0
alldata['DO']['S'][np.isinf(alldata['DO']['S'])]=0
alldata['DO']['E'][alldata['DO']['E']>10**3]=0
alldata['DO']['S'][alldata['DO']['S']>10**3]=0
alldata['DO']['E'][alldata['DO']['E']<-10**3]=0
alldata['DO']['S'][alldata['DO']['S']<-10**3]=0
#%% load AO data
os.chdir(AOfolder)
alldata['AO']['DD']=[]
alldata['AO']['AA']=[]
alldata['AO']['DA']=[]
alldata['AO']['E']=[]
alldata['AO']['S']=[]
for i in range(AOimages):
    if i==0:
        datatemp=loadmat(r'spool_sorted_FRET_Right.mat', simplify_cells=True)['FRET_pairs']
    else:
        datatemp=loadmat(r'spool_{0}_sorted_FRET_Right.mat'.format(i), simplify_cells=True)['FRET_pairs']
    alldata['AO']['DD'].extend([i['LeftID'] for i in datatemp])
    alldata['AO']['AA'].extend([i['RightIA'] for i in datatemp])
    alldata['AO']['DA'].extend([i['RightID'] for i in datatemp])
alldata['AO']['DD']=np.array(alldata['AO']['DD']).T
alldata['AO']['AA']=np.array(alldata['AO']['AA']).T
alldata['AO']['DA']=np.array(alldata['AO']['DA']).T 
if smoothing:
    for i in ['DD', 'AA', 'DA']:
        alldata['AO'][i] = np.apply_along_axis(smooth, 0, alldata['AO'][i])

   
alldata['AO']['E']=(alldata['AO']['DA']/(alldata['AO']['DD']+alldata['AO']['DA'])).T
alldata['AO']['S']=((alldata['AO']['DA']+alldata['AO']['DD'])/(alldata['AO']['DD']+alldata['AO']['DA']+alldata['AO']['AA'])).T
alldata['AO']['E'][np.isnan(alldata['AO']['E'])]=0
alldata['AO']['S'][np.isnan(alldata['AO']['S'])]=0
alldata['AO']['E'][np.isinf(alldata['AO']['E'])]=0
alldata['AO']['S'][np.isinf(alldata['AO']['S'])]=0
alldata['AO']['E'][alldata['AO']['E']>10**3]=0
alldata['AO']['S'][alldata['AO']['S']>10**3]=0
alldata['AO']['E'][alldata['AO']['E']<-10**3]=0
alldata['AO']['S'][alldata['AO']['S']<-10**3]=0


#%% apply filters to clean up DO data
filterDO_int=alldata['DO']['DD'].flatten()>200 #bright molecules only
filterDO_int*=(alldata['DO']['E'].flatten()<.2)*(alldata['DO']['E'].flatten()>-.2)
plt.hist2d(alldata['DO']['E'].flatten()[filterDO_int],alldata['DO']['S'].flatten()[filterDO_int], bins=(np.linspace(-.50,1.5,100),np.linspace(-.50,1.5,100)), norm=LogNorm(), cmap='hsv')
#
plt.show()
#%%

DOhist=plt.hist2d(alldata['AO']['DD'].flatten(),alldata['AO']['AA'].flatten(),bins=(100,100), norm=LogNorm(), cmap='hsv')
plt.xlim(0,4000)
plt.ylim(0,2000)
plt.show()

#%%
filterAO_int=(alldata['AO']['AA'].flatten()>100) #bright molecules only
filterAO_int*=(alldata['AO']['S'].flatten()>.0)*(alldata['AO']['S'].flatten()<.4)

AOhist=plt.hist2d(alldata['AO']['E'].flatten()[filterAO_int],alldata['AO']['S'].flatten()[filterAO_int], bins=(np.linspace(0,1,100),np.linspace(0,1,100)), norm=LogNorm(), cmap='hsv')
#
plt.show()
#%% summary figure uncorr
fig = plt.figure()
gs = GridSpec(5,5,figure=fig)

ax1 = fig.add_subplot(gs[1:,:-1])
plt.xlabel('Apparent FRET Efficiency')
plt.ylabel('Apparent Stoichiometry')
ax2 = fig.add_subplot(gs[0, :-1])
ax2.set_xticklabels([])
ax3 = fig.add_subplot(gs[1:, -1])
ax3.set_yticklabels([])
ax1.hexbin(np.concatenate((alldata['AO']['E'].flatten(), alldata['DO']['E'].flatten())),np.concatenate((alldata['AO']['S'].flatten(), alldata['DO']['S'].flatten())), extent=(-.25,1.5,-.25,1.5), gridsize=(100,100),mincnt=80, bins='log', cmap='hsv')
ax2.hist(np.concatenate((alldata['AO']['E'].flatten(), alldata['DO']['E'].flatten())), bins = np.linspace(-.25,1.25,101), density = True, color = 'grey')
ax3.hist(np.concatenate((alldata['AO']['S'].flatten(), alldata['DO']['S'].flatten())), bins = np.linspace(-.25,1.25,101), density = True, color = 'grey', orientation = 'horizontal')


ax1.vlines(0, -.25,1.25, color='black')
ax1.hlines(0, -.25,1.25, color='black')
ax1.set_xlim(-.25,1.25)
ax1.set_ylim(-.25,1.25)
ax2.set_xlim(-.25,1.25)
ax3.set_ylim(-.25,1.25)

plt.savefig(Corrections_Folder + r'\uncorrected_DO_AO.png')
plt.savefig(Corrections_Folder + r'\uncorrected_DO_AO.pdf')
plt.show()

#%% calculate a, d
DO_meanE=np.mean(alldata['DO']['E'].flatten()[filterDO_int])
AO_meanS=np.mean(alldata['AO']['S'].flatten()[filterAO_int])

alpha=DO_meanE/(1-DO_meanE)
delta=AO_meanS/(1-AO_meanS)

alldata['AO']['DA_corrAD']=alldata['AO']['DA']-alpha*alldata['AO']['DD']-delta*alldata['AO']['AA']
alldata['AO']['S_corrAD']=(alldata['AO']['DA_corrAD']+alldata['AO']['DD'])/(alldata['AO']['DA_corrAD']+alldata['AO']['DD']+alldata['AO']['AA'])
alldata['AO']['E_corrAD']=(alldata['AO']['DA_corrAD'])/(alldata['AO']['DA_corrAD']+alldata['AO']['DD'])

alldata['DO']['DA_corrAD']=alldata['DO']['DA']-alpha*alldata['DO']['DD']-delta*alldata['DO']['AA']
alldata['DO']['S_corrAD']=(alldata['DO']['DA_corrAD']+alldata['DO']['DD'])/(alldata['DO']['DA_corrAD']+alldata['DO']['DD']+alldata['DO']['AA'])
alldata['DO']['E_corrAD']=(alldata['DO']['DA_corrAD'])/(alldata['DO']['DA_corrAD']+alldata['DO']['DD'])

#%% plot corrected S and E
fig = plt.figure()
gs = GridSpec(5,5,figure=fig)

ax1 = fig.add_subplot(gs[1:,:-1])
plt.xlabel('Apparent FRET Efficiency')
plt.ylabel('Apparent Stoichiometry')
ax2 = fig.add_subplot(gs[0, :-1])
ax2.set_xticklabels([])
ax3 = fig.add_subplot(gs[1:, -1])
ax3.set_yticklabels([])
ax1.hexbin(np.concatenate((alldata['AO']['E_corrAD'].flatten(), alldata['DO']['E_corrAD'].flatten())),np.concatenate((alldata['AO']['S_corrAD'].flatten(), alldata['DO']['S_corrAD'].flatten())), extent=(-.25,1.5,-.25,1.5), gridsize=(100,100),mincnt=80, bins='log', cmap='hsv')
ax2.hist(np.concatenate((alldata['AO']['E_corrAD'].flatten(), alldata['DO']['E_corrAD'].flatten())), bins = np.linspace(-.25,1.25,101), density = True, color = 'grey')
ax3.hist(np.concatenate((alldata['AO']['S_corrAD'].flatten(), alldata['DO']['S_corrAD'].flatten())), bins = np.linspace(-.25,1.25,101), density = True, color = 'grey', orientation = 'horizontal')

DO_meanE=np.mean(alldata['DO']['E'].flatten()[filterDO_int])
AO_meanS=np.mean(alldata['AO']['S'].flatten()[filterAO_int])

alpha=DO_meanE/(1-DO_meanE)
delta=AO_meanS/(1-AO_meanS)
ax1.vlines(0, -.25,1.25, color='black')
ax1.hlines(0, -.25,1.25, color='black')
ax1.set_xlim(-.25,1.25)
ax1.set_ylim(-.25,1.25)
ax2.set_xlim(-.25,1.25)
ax3.set_ylim(-.25,1.25)

plt.savefig(Corrections_Folder + r'\corrected_DO_AO.png')
plt.savefig(Corrections_Folder + r'\corrected_DO_AO.pdf')
plt.show()


#%% save corrections
with open(Corrections_Folder + r'\alpha_delta.txt','w+') as f:
    f.write('#alpha and delta calculated with {0} images from {1} and {2} from {3}\n'.format(DOimages, DOfolder, AOimages, AOfolder))
    f.write('{0}\n{1}'.format(alpha, delta))
    





