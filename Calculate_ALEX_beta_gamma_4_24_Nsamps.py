# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:56:12 2024

@author: glham

#works with trajectories from MainBatch_FRET
#calculates beta and gamma functions. requires either a previous run of alpha_delta script
#or setting alpha_delta manually
"""
#%%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from scipy.io import loadmat
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
import math as m

#two separate standard FRET samples
DAfolders = [r'C:\Users\GH\demo_data\DA_samples\10bp', 
             r'C:\Users\GH\demo_data\DA_samples\15bp',
             r'C:\Users\GH\demo_data\DA_samples\20bp'] #folders for image/trace data

DAimages = [2, 2, 2] #number of images in each folder

subsample_to_reweight = True #experimental; whether to subsample arrays of expts with more data to match the number in the expt with the fewest datapoints (equally weight the samples)

Corrections_Folder = r'C:\Users\GH\demo_data\Corrections_Folder'

alpha_delta_file =  'alpha_delta.txt'#'alpha_delta.txt'#None if no file, and then alpha, delta must be specified. File should be in corrections folder. expects file to be header row 1, alpha row 2, delta row 3
alpha = 0.034 #only used if alpha_delta_file not specified
delta = 0.136 #only used if alpha_delta_file not specified

if alpha_delta_file is not None:
    alpha = np.loadtxt(Corrections_Folder + r'\\' + alpha_delta_file, skiprows=1)[0]
    delta = np.loadtxt(Corrections_Folder + r'\\' + alpha_delta_file, skiprows=1)[1]

smoothing = True
window = 5
smooth = lambda array: np.convolve(array, np.ones(window)/window, mode='valid')

alldata={i:{} for i in range(len(DAfolders))}

#%%
index = 0
for i in DAfolders: #load all the data and calculate S, E
    os.chdir(i)
    alldata[index]['DD']=[]
    alldata[index]['AA']=[]
    alldata[index]['DA']=[]
    alldata[index]['E']=[]
    alldata[index]['S']=[]
    for i in range(DAimages[index]):
        if i==0:

            datatemp=loadmat(r'spool_sorted_FRET_FRET.mat', simplify_cells=True)['FRET_pairs']
        else:

            datatemp=loadmat(r'spool_{0}_sorted_FRET_FRET.mat'.format(i), simplify_cells=True)['FRET_pairs']
        alldata[index]['DD'].extend([i['LeftID'] for i in datatemp])
        alldata[index]['AA'].extend([i['RightIA'] for i in datatemp])
        alldata[index]['DA'].extend([i['RightID'] for i in datatemp])
    alldata[index]['DD']=np.array(alldata[index]['DD']).T
    alldata[index]['AA']=np.array(alldata[index]['AA']).T
    alldata[index]['DA']=np.array(alldata[index]['DA']).T    

    if smoothing:
        for i in ['DD', 'AA', 'DA']:
            alldata[index][i] = np.apply_along_axis(smooth, 0, alldata[index][i])
            
    alldata[index]['E']=(alldata[index]['DA']/(alldata[index]['DD']+alldata[index]['DA'])).T
    alldata[index]['S']=((alldata[index]['DA']+alldata[index]['DD'])/(alldata[index]['DD']+alldata[index]['DA']+alldata[index]['AA'])).T
    alldata[index]['E'][np.isnan(alldata[index]['E'])]=0  #just to handle weird values that give NaNs; minimal contribution and can be filtered out later
    alldata[index]['S'][np.isnan(alldata[index]['S'])]=0
    alldata[index]['E'][np.isinf(alldata[index]['E'])]=0
    alldata[index]['S'][np.isinf(alldata[index]['S'])]=0
    alldata[index]['E'][alldata[index]['E']>10**3]=0
    alldata[index]['S'][alldata[index]['S']>10**3]=0
    alldata[index]['E'][alldata[index]['E']<-10**3]=0
    alldata[index]['S'][alldata[index]['S']<-10**3]=0
    
    index += 1

#%% alpha, delta correction
index = 0
for i in DAfolders:

    alldata[index]['DA_corrAD']=alldata[index]['DA']-alpha*alldata[index]['DD']-delta*alldata[index]['AA']
    alldata[index]['S_corrAD']=(alldata[index]['DA_corrAD']+alldata[index]['DD'])/(alldata[index]['DA_corrAD']+alldata[index]['DD']+alldata[index]['AA'])
    alldata[index]['E_corrAD']=(alldata[index]['DA_corrAD'])/(alldata[index]['DA_corrAD']+alldata[index]['DD'])

    plt.hexbin(alldata[index]['E_corrAD'].flatten(),alldata[index]['S_corrAD'].flatten(), extent=(-.25,1.25,-.25,1.25), gridsize=(100,100),mincnt=1, bins='log', cmap='hsv')

    plt.show()
    index += 1

#%% establish filters

thresh_E=[-.25,1.1] #apparent efficiency threshold
thresh_S=[.4,.9] #apparent stoichiometry thresholds
thresh_D=0 #min D brightness (if high E, D can be low brightness)
thresh_A=80 #min A brightness during direct excitation
thresh_DA = 300 #minimum total brightness in D excitation

fbo = '' #filter based on data of type with given suffix '' or '_corrAD' 

Efilts = [(i['E' + fbo].flatten()>thresh_E[0])*(i['E' + fbo].flatten()<thresh_E[1]) for i in alldata.values()]
Sfilts = [(i['S' + fbo].flatten()>thresh_S[0])*(i['S' + fbo].flatten()<thresh_S[1]) for i in alldata.values()]
Dfilts = [(i['DD'].flatten()>thresh_D) for i in alldata.values()]
Afilts = [(i['AA'].flatten()>thresh_A) for i in alldata.values()]
DAfilts = [((i['DD'].flatten() + i['DA'].flatten())>thresh_DA) for i in alldata.values()]

filts = {}
for i in range(len(Efilts)):
    filts[i] = (Efilts[i] * Sfilts[i] * Dfilts[i] * Afilts[i] * DAfilts[i])

if subsample_to_reweight:   
    counts = [sum(i) for i in filts.values()] 
    fewest = min(counts)
    subsamp_strides = {i: m.ceil(counts[i]/fewest) for i in filts}
else:
    subsamp_strides = {i: 1 for i in filts}

EsUn=np.concatenate([alldata[i]['E'].flatten()[filts[i]][::subsamp_strides[i]] for i in alldata])
SsUn=np.concatenate([alldata[i]['S'].flatten()[filts[i]][::subsamp_strides[i]] for i in alldata])
    
EsAD=np.concatenate([alldata[i]['E_corrAD'].flatten()[filts[i]][::subsamp_strides[i]] for i in alldata])
SsAD=np.concatenate([alldata[i]['S_corrAD'].flatten()[filts[i]][::subsamp_strides[i]] for i in alldata])
#SinvAD=1./SsAD #calculated in case one instead wants to do the linear fit method

#%% fit SvE function to S vs E to calculate b, g
def SvE(x,beta,gamma):
    return (1.+gamma*beta+(1.-gamma)*beta*x)**(-1.)

popt,pcov=curve_fit(SvE, EsAD,SsAD, p0=(.9, .5))#, bounds=((0,0.7),(2.,2.5)))

plt.hexbin(EsAD,SsAD, extent=(-.25,1.25,-.25,1.25), gridsize=(100,100),mincnt=30, bins='log', cmap='hsv')

plt.plot(np.array([-.25,1.25]), SvE(np.array([-.25,1.25]), popt[0], popt[1]), 'k--')

plt.xlabel('E')
plt.ylabel('S')
plt.xlim(-.25, 1.25)
plt.ylim(-.25, 1.25)

plt.show()


beta=popt[0] #0.5808900489279684
gamma=popt[1] #0.4965545599446537

#%%calculate corrected data
index = 0
for i in DAfolders:

    alldata[index]['DD_corrG']=gamma*alldata[index]['DD']
    alldata[index]['AA_corrB']=1./beta*alldata[index]['AA']
    alldata[index]['S_corrABGD']=(alldata[index]['DA_corrAD']+alldata[index]['DD_corrG'])/(alldata[index]['DA_corrAD']+alldata[index]['DD_corrG']+alldata[index]['AA_corrB'])
    alldata[index]['E_corrABGD']=(alldata[index]['DA_corrAD'])/(alldata[index]['DA_corrAD']+alldata[index]['DD_corrG'])
    index += 1

EsABGD=np.concatenate([alldata[i]['E_corrABGD'].flatten()[filts[i]][::subsamp_strides[i]] for i in alldata])
SsABGD=np.concatenate([alldata[i]['S_corrABGD'].flatten()[filts[i]][::subsamp_strides[i]] for i in alldata])


#%% save uncorrected figure
fig = plt.figure()
gs = GridSpec(5,5,figure=fig)

ax1 = fig.add_subplot(gs[1:,:-1])
plt.xlabel('Apparent FRET Efficiency')
plt.ylabel('Apparent Stoichiometry')
ax2 = fig.add_subplot(gs[0, :-1])
ax2.set_xticklabels([])
ax3 = fig.add_subplot(gs[1:, -1])
ax3.set_yticklabels([])
ax1.hexbin(EsUn,SsUn, extent=(-.25,1.5,-.25,1.5), gridsize=(100,100),mincnt=50, bins='log', cmap='hsv')
ax2.hist(EsUn, bins = np.linspace(-.25,1.25,101), density = True, color = 'grey')
ax3.hist(SsUn, bins = np.linspace(-.25,1.25,101), density = True, color = 'grey', orientation = 'horizontal')


ax1.hlines(.5, -.25,1.25, color='gray')
ax1.set_xlim(-.25,1.25)
ax1.set_ylim(-.25,1.25)
ax2.set_xlim(-.25,1.25)
ax3.set_ylim(-.25,1.25)

plt.savefig(Corrections_Folder + r'\uncorrected_FRET_Standards.png')
plt.savefig(Corrections_Folder + r'\uncorrected_FRET_Standards.pdf')
plt.show()

#%% save ad corrected figure
fig = plt.figure()
gs = GridSpec(5,5,figure=fig)

ax1 = fig.add_subplot(gs[1:,:-1])
plt.xlabel('Apparent FRET Efficiency')
plt.ylabel('Apparent Stoichiometry')
ax2 = fig.add_subplot(gs[0, :-1])
ax2.set_xticklabels([])
ax3 = fig.add_subplot(gs[1:, -1])
ax3.set_yticklabels([])
ax1.hexbin(EsAD,SsAD, extent=(-.25,1.5,-.25,1.5), gridsize=(100,100),mincnt=50, bins='log', cmap='hsv')
ax2.hist(EsAD, bins = np.linspace(-.25,1.25,101), density = True, color = 'grey')
ax3.hist(SsAD, bins = np.linspace(-.25,1.25,101), density = True, color = 'grey', orientation = 'horizontal')
ax1.plot(np.array([-.25,1.25]), SvE(np.array([-.25,1.25]), popt[0], popt[1]), color = 'black', linestyle = 'dashed')

ax1.hlines(.5, -.25,1.25, color='gray')
ax1.set_xlim(-.25,1.25)
ax1.set_ylim(-.25,1.25)
ax2.set_xlim(-.25,1.25)
ax3.set_ylim(-.25,1.25)

plt.savefig(Corrections_Folder + r'\ADcorrected_FRET_Standards.png')
plt.savefig(Corrections_Folder + r'\ADcorrected_FRET_Standards.pdf')
plt.show()

#%% save abgd corrected
fig = plt.figure()
gs = GridSpec(5,5,figure=fig)

ax1 = fig.add_subplot(gs[1:,:-1])
plt.xlabel('Apparent FRET Efficiency')
plt.ylabel('Apparent Stoichiometry')
ax2 = fig.add_subplot(gs[0, :-1])
ax2.set_xticklabels([])
ax3 = fig.add_subplot(gs[1:, -1])
ax3.set_yticklabels([])
ax1.hexbin(EsABGD,SsABGD, extent=(-.25,1.5,-.25,1.5), gridsize=(100,100),mincnt=50, bins='log', cmap='hsv')
ax2.hist(EsABGD, bins = np.linspace(-.25,1.25,101), density = True, color = 'grey')
ax3.hist(SsABGD, bins = np.linspace(-.25,1.25,101), density = True, color = 'grey', orientation = 'horizontal')


ax1.hlines(.5, -.25,1.25, color='gray')
ax1.set_xlim(-.25,1.25)
ax1.set_ylim(-.25,1.25)
ax2.set_xlim(-.25,1.25)
ax3.set_ylim(-.25,1.25)

plt.savefig(Corrections_Folder + r'\ABDGcorrected_FRET_Standards.png')
plt.savefig(Corrections_Folder + r'\ABDGcorrected_FRET_Standards.pdf')
plt.show()
#%% save corrections
with open(Corrections_Folder + r'\beta_gamma.txt','w+') as f:
    f.write('#beta and gamma calculated with images from {0}\n'.format(DAfolders))
    f.write('{0}\n{1}'.format(beta, gamma))
    

















# %%
