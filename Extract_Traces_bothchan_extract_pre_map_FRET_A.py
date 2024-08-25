# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:50:15 2024

@author: GH
"""

import os
import numpy as np
import skimage as sk
import tkinter as tk
from tkinter import filedialog
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.draw import polygon
import tifffile as tf
import matplotlib.pyplot as plt
#import screeninfo as si
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors
import matplotlib.gridspec as gridspec
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, SpectralClustering, OPTICS, DBSCAN, AffinityPropagation
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import squareform
import pickle

# =============================================================================
# folderpath=r'D:\Data\DNA_RAD51_ALEX\DATA\DNA_ONLY'
# =============================================================================
# =============================================================================
# folderpath=r'D:\Data\DNA_RAD51_ALEX\DATA\DNA_RAD51_400nM'
# =============================================================================
folderpath=r'D:\Data\SUDIPTA_ALEX_Correction\03282024\DATA\ALEX\10 bp'

# =============================================================================
# folderpath=r'D:\Data\DNA_standards\data\AO_for_RAD51exp'
# =============================================================================

prefix='spool'
suffix='_sorted.tif'
folders=10
frame_time=.200 #seconds
centerdist=3
squaresize=5
radius=int((squaresize-1)/2)
ringsize=1 #thickness in addition to initial square to be used for BG estimation
ringgap=0 #pizels between box and ring
ringradius=int(radius+ringsize+ringgap)
innerradius=int(radius+ringgap)
#LSP finds the element in the box satsifying the given expected percentile (given as a decimal)
bg_method='LSP_np'#'median'#'mean'#'min'#'LSP'#'LSP_np
LSP_X=.53
invmapparaloc=r'D:\Data\Alex_DNA_Standards\cw\map\spool.tif_R_to_G(inv)skPolyTrans2.dat'
transorder=1
invmappara=np.loadtxt(invmapparaloc)
alpha=0.01
delta=0.2
use_corrections=False
save_data=True

mapparaloc=r'D:\Data\Alex_DNA_Standards\cw\map\spool.tif_G_to_RskPolyTrans2.dat'
mappara=np.loadtxt(mapparaloc)

mean_bg=False
do_extract_on_unmapped=True
showpeaks=False

use_unmatched=False #True, False, 'DO' or 'AO'
#used to keep all matched peaks but also take all unmatched peaks and pair it up with coordinates in the other channel based on the transformation
#this allows inclusion of DO, AO molecules
#setting DO or AO includes ONLY AO or DO molecules
exclude_matched=False #if use_unmatched and either 'DO' or 'AO', exclude the matched coordinates

thresh_D=.2
thresh_A=.2
maxmol=300

take_top=False
top_N=9 #number of brightest pixels to consider when calculating traces

def PolyTransCoord2D(inarray, transformation_params, order=3):
    if len(transformation_params.flatten())!=(order+1)*(order+2):
        print('Check that the number of input parameters is correct for the desired order. N*2==(ord+1)*(ord+2)')
        return
    paramsarr=np.zeros((2,order+1, order+1))
    ind=0
    for j in range(order+1):
        for i in range(j+1):
            paramsarr[0,j,i]=transformation_params[0][ind]
            paramsarr[1,j,i]=transformation_params[1][ind]
            ind+=1
    outarray=np.zeros(inarray.shape)
    ind=0
    for j in range(order+1):
       for i in range(j+1):
           outarray[:,0]+=paramsarr[1,j,i]*inarray[:,1]**(j-i)*inarray[:,0]**i
           outarray[:,1]+=paramsarr[0,j,i]*inarray[:,1]**(j-i)*inarray[:,0]**i
           ind+=1
    return outarray
def make_square(inarray):
    if int(len(inarray)**.5)!=len(inarray)**.5:
        print("Array must be a square value")
        return
    outarray=np.zeros((int(len(inarray)**.5),int(len(inarray)**.5)))
    for i in range(int(len(inarray)**.5)):
        outarray[:,i]=inarray[i::int(len(inarray)**.5)]
    return outarray
def get_percentile(array, percentile):
    array=array.flatten()
    inds=np.argsort(array)
    return array[inds[int(len(inds)*percentile)]]

os.chdir(folderpath)
data={}

for j in range(folders):
# =============================================================================
#     os.chdir(prefix+f'{j+1}')
# =============================================================================
    data[j]={}
    if j==0:
        image=tf.imread('mapped_'+prefix+f'{suffix}')
        D_D_untrans=tf.imread('D_'+prefix+f'{suffix}')[0::2]
    else:
        image=tf.imread('mapped_'+prefix+f'_{j}{suffix}')
        D_D_untrans=tf.imread('D_'+prefix+f'_{j}{suffix}')[0::2]
# =============================================================================
#     image=tf.imread('mapped_'+prefix+f'{j+1}{suffix}')
# =============================================================================
    D_D=image[0::2,1]
    D_A=image[0::2,0]
    A_A=image[1::2,0]
    del(image)
    
    
    if do_extract_on_unmapped:
        D_sel=gaussian(np.max(D_D_untrans, axis=0), sigma=1)
    else:
        D_sel=gaussian(np.max(D_D, axis=0), sigma=1)
    A_sel=gaussian(np.max(A_A, axis=0), sigma=1)
    peaksD_all=peak_local_max(D_sel, threshold_rel=thresh_D, min_distance=radius, exclude_border=True, num_peaks=maxmol)
    peaksA_all=peak_local_max(A_sel, threshold_rel=thresh_A, min_distance=radius, exclude_border=True, num_peaks=maxmol)
    if do_extract_on_unmapped:
        pairdist=pairwise_distances(PolyTransCoord2D(peaksD_all, invmappara, order=transorder), peaksA_all)
    else:
        pairdist=pairwise_distances(peaksD_all, peaksA_all)
    if len(peaksA_all)<=len(peaksD_all):
        minmatch=np.min(pairdist, axis=0)
        ordered=np.argwhere(pairdist[:,minmatch<=centerdist]==minmatch[minmatch<=centerdist])
        peaksA=peaksA_all[minmatch<=centerdist][ordered[:,1]]
        peaksD=peaksD_all[ordered[:,0]]
        if use_unmatched:
            unmatchedA=peaksA_all[minmatch>centerdist]
            unmatchedA_to_G=np.round(PolyTransCoord2D(unmatchedA, mappara, order=transorder)).astype(int)
            unmatchedG=peaksD_all[list(set(range(len(peaksD_all)))-set(ordered[:,0]))]
            unmatchedG_to_A=np.round(PolyTransCoord2D(unmatchedG, invmappara, order=transorder)).astype(int)
            
            if use_unmatched=='DO':
                peaksA=np.concatenate((peaksA, unmatchedG_to_A))
                peaksD=np.concatenate((peaksD, unmatchedG))
                
            elif use_unmatched=='AO':
                peaksA=np.concatenate((peaksA, unmatchedA))
                peaksD=np.concatenate((peaksD, unmatchedA_to_G))
            else:
                peaksA=np.concatenate((peaksA, unmatchedA, unmatchedG_to_A)) #concatenate matched peaks, unmatched A, and unmatched G
                peaksD=np.concatenate((peaksD, unmatchedA_to_G, unmatchedG))
            
    elif len(peaksD_all)<len(peaksA_all):
        minmatch=np.min(pairdist, axis=1)
        ordered=np.argwhere((pairdist[minmatch<=centerdist,:].T==minmatch[minmatch<=centerdist]).T)
        peaksD=peaksD_all[minmatch<=centerdist][ordered[:,0]]
        peaksA=peaksA_all[ordered[:,1]]
        if use_unmatched:
            unmatchedA=peaksA_all[list(set(range(len(peaksA_all)))-set(ordered[:,1]))]
            unmatchedA_to_G=np.round(PolyTransCoord2D(unmatchedA, mappara, order=transorder)).astype(int)
            unmatchedG=peaksD_all[minmatch>centerdist]
            unmatchedG_to_A=np.round(PolyTransCoord2D(unmatchedG, invmappara, order=transorder)).astype(int)
            
            if use_unmatched=='DO':
                if exclude_matched:
                    peaksA=unmatchedG_to_A
                    peaksD=unmatchedG
                else:
                    peaksA=np.concatenate((peaksA, unmatchedG_to_A))
                    peaksD=np.concatenate((peaksD, unmatchedG))
                
            elif use_unmatched=='AO':
                if exclude_matched:
                    peaksA=unmatchedA
                    peaksD=unmatchedA_to_G
                else:
                    peaksA=np.concatenate((peaksA, unmatchedA))
                    peaksD=np.concatenate((peaksD, unmatchedA_to_G))
            else:
                peaksA=np.concatenate((peaksA, unmatchedA, unmatchedG_to_A)) #concatenate matched peaks, unmatched A, and unmatched G
                peaksD=np.concatenate((peaksD, unmatchedA_to_G, unmatchedG))
            
    #make sure peaks can fit the ring around them
    A_to_keep=((peaksA[:,0]<A_sel.shape[0]-(radius+ringgap+ringsize+1)) & (peaksA[:,0]>(radius+ringgap+ringsize)) & (peaksA[:,1]<A_sel.shape[1]-(radius+ringgap+ringsize+1)) & (peaksA[:,1]>(radius+ringgap+ringsize)))  
    D_to_keep=((peaksD[:,0]<D_sel.shape[0]-(radius+ringgap+ringsize+1)) & (peaksD[:,0]>(radius+ringgap+ringsize)) & (peaksD[:,1]<D_sel.shape[1]-(radius+ringgap+ringsize+1)) & (peaksD[:,1]>(radius+ringgap+ringsize)))  
    all_to_keep=A_to_keep*D_to_keep
    peaksA=peaksA[all_to_keep]
    peaksD=peaksD[all_to_keep]

    data[j]['raw_DD']=pd.DataFrame() #raw integrated intensity in each box
    data[j]['raw_DA']=pd.DataFrame()
    data[j]['raw_AA']=pd.DataFrame()
    data[j]['bg_DD']=pd.DataFrame() #average background per pixel in ring around box
    data[j]['bg_DA']=pd.DataFrame()
    data[j]['bg_AA']=pd.DataFrame()
    data[j]['sub_DD']=pd.DataFrame() #background subtracted integrated intensity
    data[j]['sub_DA']=pd.DataFrame()
    data[j]['sub_AA']=pd.DataFrame()
    data[j]['Prox_Rat']=pd.DataFrame()
    data[j]['Stoichiometry']=pd.DataFrame()
    for i in range(len(peaksA)):
        coor_peakA=polygon([peaksA[i,1]+radius, peaksA[i,1]+radius, peaksA[i,1]-radius, peaksA[i,1]-radius], [peaksA[i,0]+radius, peaksA[i,0]-radius, peaksA[i,0]-radius, peaksA[i,0]+radius])
        coor_peakD=polygon([peaksD[i,1]+radius, peaksD[i,1]+radius, peaksD[i,1]-radius, peaksD[i,1]-radius], [peaksD[i,0]+radius, peaksD[i,0]-radius, peaksD[i,0]-radius, peaksD[i,0]+radius])
        
        coor_ringA=polygon([peaksA[i,1]+ringradius, peaksA[i,1]+ringradius, peaksA[i,1]-ringradius, peaksA[i,1]-ringradius], [peaksA[i,0]+ringradius, peaksA[i,0]-ringradius, peaksA[i,0]-ringradius, peaksA[i,0]+ringradius])
        coor_ringD=polygon([peaksD[i,1]+ringradius, peaksD[i,1]+ringradius, peaksD[i,1]-ringradius, peaksD[i,1]-ringradius], [peaksD[i,0]+ringradius, peaksD[i,0]-ringradius, peaksD[i,0]-ringradius, peaksD[i,0]+ringradius])
        
        coor_peak_gapA=polygon([peaksA[i,1]+innerradius, peaksA[i,1]+innerradius, peaksA[i,1]-innerradius, peaksA[i,1]-innerradius], [peaksA[i,0]+innerradius, peaksA[i,0]-innerradius, peaksA[i,0]-innerradius, peaksA[i,0]+innerradius])
        coor_peak_gapD=polygon([peaksD[i,1]+innerradius, peaksD[i,1]+innerradius, peaksD[i,1]-innerradius, peaksD[i,1]-innerradius], [peaksD[i,0]+innerradius, peaksD[i,0]-innerradius, peaksD[i,0]-innerradius, peaksD[i,0]+innerradius])
        
        coor_ring_onlyA=[i for i in list(zip(coor_ringA[0],coor_ringA[1])) if i not in list(zip(coor_peak_gapA[0],coor_peak_gapA[1]))]
        coor_ring_onlyD=[i for i in list(zip(coor_ringD[0],coor_ringD[1])) if i not in list(zip(coor_peak_gapD[0],coor_peak_gapD[1]))]
        
        coor_ring_onlyA=[np.array([i[0] for i in coor_ring_onlyA]),np.array([i[1] for i in coor_ring_onlyA])]
        coor_ring_onlyD=[np.array([i[0] for i in coor_ring_onlyD]),np.array([i[1] for i in coor_ring_onlyD])]
                
        if showpeaks:
            plt.imshow(make_square(D_sel[coor_peakD[1], coor_peakD[0]]), cmap='Greens')
            plt.show()
            plt.imshow(make_square(A_sel[coor_peakA[1], coor_peakA[0]]), cmap='Reds')
            plt.show()
        
        if take_top:
            if do_extract_on_unmapped:
                data[j]['raw_DD']['molecule_{0}'.format(i)]=np.mean(np.take_along_axis(D_D_untrans[:,coor_peakD[1], coor_peakD[0]], np.argsort(D_D_untrans[:,coor_peakD[1], coor_peakD[0]], axis=1)[:,-top_N:], axis=1), axis=1)
                
            else:
                data[j]['raw_DD']['molecule_{0}'.format(i)]=np.mean(np.take_along_axis(D_D[:,coor_peakD[1], coor_peakD[0]], np.argsort(D_D[:,coor_peakD[1], coor_peakD[0]], axis=1)[:,-top_N:], axis=1), axis=1)
            data[j]['raw_DA']['molecule_{0}'.format(i)]=np.mean(np.take_along_axis(D_A[:,coor_peakA[1], coor_peakA[0]], np.argsort(D_A[:,coor_peakA[1], coor_peakA[0]], axis=1)[:,-top_N:], axis=1), axis=1)
            data[j]['raw_AA']['molecule_{0}'.format(i)]=np.mean(np.take_along_axis(A_A[:,coor_peakA[1], coor_peakA[0]], np.argsort(A_A[:,coor_peakA[1], coor_peakA[0]], axis=1)[:,-top_N:], axis=1), axis=1)
        else:
            if do_extract_on_unmapped:
                data[j]['raw_DD']['molecule_{0}'.format(i)]=np.mean(D_D_untrans[:,coor_peakD[1], coor_peakD[0]], axis=1)
                
            else:
                data[j]['raw_DD']['molecule_{0}'.format(i)]=np.mean(D_D[:,coor_peakD[1], coor_peakD[0]], axis=1)
            data[j]['raw_DA']['molecule_{0}'.format(i)]=np.mean(D_A[:,coor_peakA[1], coor_peakA[0]],axis=1)
            data[j]['raw_AA']['molecule_{0}'.format(i)]=np.mean(A_A[:,coor_peakA[1], coor_peakA[0]], axis=1)
            
        if bg_method=='mean':
            if do_extract_on_unmapped:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.mean(D_D_untrans[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], axis=1)
            else:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.mean(D_D[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], axis=1)
            data[j]['bg_DA']['molecule_{0}'.format(i)]=np.mean(D_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], axis=1)
            data[j]['bg_AA']['molecule_{0}'.format(i)]=np.mean(A_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], axis=1)
        elif bg_method=='median':
            if do_extract_on_unmapped:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.median(D_D_untrans[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], axis=1)
            else:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.median(D_D[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], axis=1)
            data[j]['bg_DA']['molecule_{0}'.format(i)]=np.median(D_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], axis=1)
            data[j]['bg_AA']['molecule_{0}'.format(i)]=np.median(A_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], axis=1)
        elif bg_method=='min':
            if do_extract_on_unmapped:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.min(D_D_untrans[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], axis=1)
            else:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.min(D_D[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], axis=1)
            data[j]['bg_DA']['molecule_{0}'.format(i)]=np.min(D_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], axis=1)
            data[j]['bg_AA']['molecule_{0}'.format(i)]=np.min(A_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], axis=1)
        elif bg_method=='LSP':
            if do_extract_on_unmapped:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.ones(len(D_D_untrans[:,coor_ring_onlyD[1], coor_ring_onlyD[0]]))*get_percentile(D_D_untrans[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], LSP_X)
            else:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.ones(len(D_D_untrans[:,coor_ring_onlyD[1], coor_ring_onlyD[0]]))*get_percentile(D_D[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], LSP_X)
            data[j]['bg_DA']['molecule_{0}'.format(i)]=np.ones(len(D_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]]))*get_percentile(D_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], LSP_X)
            data[j]['bg_AA']['molecule_{0}'.format(i)]=np.ones(len(A_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]]))*get_percentile(A_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], LSP_X)
        elif bg_method=='LSP_np':
            if do_extract_on_unmapped:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.percentile(D_D_untrans[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], LSP_X*100, axis=1)
            else:
                data[j]['bg_DD']['molecule_{0}'.format(i)]=np.percentile(D_D[:,coor_ring_onlyD[1], coor_ring_onlyD[0]], LSP_X*100, axis=1)
            data[j]['bg_DA']['molecule_{0}'.format(i)]=np.percentile(D_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], LSP_X*100, axis=1)
            data[j]['bg_AA']['molecule_{0}'.format(i)]=np.percentile(A_A[:,coor_ring_onlyA[1], coor_ring_onlyA[0]], LSP_X*100, axis=1)
            
        if mean_bg:
            data[j]['bg_DD']['molecule_{0}'.format(i)]=np.mean(data[j]['bg_DD']['molecule_{0}'.format(i)])
            data[j]['bg_DA']['molecule_{0}'.format(i)]=np.mean(data[j]['bg_DA']['molecule_{0}'.format(i)])
            data[j]['bg_AA']['molecule_{0}'.format(i)]=np.mean(data[j]['bg_AA']['molecule_{0}'.format(i)])
        
        data[j]['sub_DD']['molecule_{0}'.format(i)]=data[j]['raw_DD']['molecule_{0}'.format(i)]-data[j]['bg_DD']['molecule_{0}'.format(i)]#*len(coor_peak[0])
        data[j]['sub_DA']['molecule_{0}'.format(i)]=data[j]['raw_DA']['molecule_{0}'.format(i)]-data[j]['bg_DA']['molecule_{0}'.format(i)]#*len(coor_peak[0])
        data[j]['sub_AA']['molecule_{0}'.format(i)]=data[j]['raw_AA']['molecule_{0}'.format(i)]-data[j]['bg_AA']['molecule_{0}'.format(i)]#*len(coor_peak[0])
        if use_corrections:
            data[j]['sub_DA']['molecule_{0}'.format(i)]=data[j]['sub_DA']['molecule_{0}'.format(i)]-alpha*data[j]['sub_DD']['molecule_{0}'.format(i)]-delta*data[j]['sub_AA']['molecule_{0}'.format(i)]
        
        data[j]['Prox_Rat']['molecule_{0}'.format(i)]=data[j]['sub_DA']['molecule_{0}'.format(i)]/(data[j]['sub_DA']['molecule_{0}'.format(i)]+data[j]['sub_DD']['molecule_{0}'.format(i)])
        data[j]['Stoichiometry']['molecule_{0}'.format(i)]=(data[j]['sub_DD']['molecule_{0}'.format(i)]+data[j]['sub_DA']['molecule_{0}'.format(i)])/(data[j]['sub_DD']['molecule_{0}'.format(i)]+data[j]['sub_DA']['molecule_{0}'.format(i)]+data[j]['sub_AA']['molecule_{0}'.format(i)])
    if save_data:
        if j==0:
            with open('traces_'+prefix+f'.pkl', 'wb') as file:
                pickle.dump(data[j], file, protocol=pickle.HIGHEST_PROTOCOL)
    
        else:
            with open('traces_'+prefix+f'_{j}.pkl', 'wb') as file:
                pickle.dump(data[j], file, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    #os.chdir(r'..')

    #%%
plt.imshow(A_sel)
plt.scatter(peaksA_all[:,1], peaksA_all[:,0], color='black', s=1)
plt.scatter(peaksA[:,1], peaksA[:,0], color='red', s=.1)
plt.show() 
    
    
    #%%
plt.imshow(D_sel)
plt.scatter(peaksD_all[:,1], peaksD_all[:,0], color='black', s=1)
plt.scatter(peaksD[:,1], peaksD[:,0], color='red', s=.1)
plt.show() 
      
#%%

peaksD_trans=peak_local_max(gaussian(np.max(D_D, axis=0),sigma=1), threshold_rel=.4, min_distance=ringradius, exclude_border=True, num_peaks=200)
peaksD_untrans=peak_local_max(gaussian(np.max(D_D_untrans, axis=0),sigma=1), threshold_rel=.4, min_distance=ringradius, exclude_border=True, num_peaks=200)
peaksD_retrans=PolyTransCoord2D(peaksD_untrans, invmappara, order=transorder)

plt.scatter(peaksD_trans[:,1],peaksD_trans[:,0],s=3,color='k')
plt.scatter(peaksD_retrans[:,1],peaksD_retrans[:,0],s=.5,color='g')
plt.ylim(512,0)
plt.xlim(0,256)
plt.show()
    
    #%%
start_frame=0
acc_E=[]
acc_S=[]
plot=True

for i in range(len(peaksA)):
    if plot:
        try:
            fig=plt.figure(constrained_layout=True)
            gs=fig.add_gridspec(nrows=4, ncols=7)
            ax1=fig.add_subplot(gs[-2,2:-1])
            ax2=fig.add_subplot(gs[-1,2:-1])
            ax3=fig.add_subplot(gs[-2,-1])
            ax4=fig.add_subplot(gs[-1,-1])
            ax5=fig.add_subplot(gs[-3,2:-1])
            ax6=fig.add_subplot(gs[-3,-1])
            ax7=fig.add_subplot(gs[-4,2:-1])
            ax8=fig.add_subplot(gs[-4,-1])
            ax9=fig.add_subplot(gs[:2,:2])
            ax10=fig.add_subplot(gs[2:,:2])
            ax1.plot(data[j]['sub_AA']['molecule_{0}'.format(i)][start_frame:], color='black',label='I_AA')
            ax1.plot(data[j]['sub_DA']['molecule_{0}'.format(i)][start_frame:], color='red',label='I_DA')
            ax1.plot(data[j]['sub_DD']['molecule_{0}'.format(i)][start_frame:], color='green',label='I_DD')
            ax1.legend(loc='upper right', fontsize="5")
            ax3.hist(data[j]['sub_AA']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='black', alpha=.5, bins=25)
            ax3.hist(data[j]['sub_DA']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='red', alpha=.5, bins=25)
            ax3.hist(data[j]['sub_DD']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='green', alpha=.5, bins=25)
            
            ax2.plot(data[j]['Prox_Rat']['molecule_{0}'.format(i)][start_frame:], color='purple',label='E')
            ax2.plot(data[j]['Stoichiometry']['molecule_{0}'.format(i)][start_frame:], color='orange', label='S')
            ax2.set_ylim(-.2,1.2)
            ax2.legend(loc='upper right', fontsize="5")
            ax4.hist(data[j]['Prox_Rat']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='purple', alpha=.5, bins=np.linspace(-.2,1.2,25))
            ax4.hist(data[j]['Stoichiometry']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='orange', alpha=.5, bins=np.linspace(-.2,1.2,25))
            ax4.set_ylim(-.2,1.2)
            
            ax5.plot(data[j]['bg_AA']['molecule_{0}'.format(i)][start_frame:], color='black',label='BG_AA')
            ax5.plot(data[j]['bg_DA']['molecule_{0}'.format(i)][start_frame:], color='red',label='BG_DA')
            ax5.plot(data[j]['bg_DD']['molecule_{0}'.format(i)][start_frame:], color='green',label='BG_DD')
            ax5.legend(loc='upper right', fontsize="5")
            ax6.hist(data[j]['bg_AA']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='black', alpha=.5, bins=25)
            ax6.hist(data[j]['bg_DA']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='red', alpha=.5, bins=25)
            ax6.hist(data[j]['bg_DD']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='green', alpha=.5, bins=25)
            
            ax7.plot(data[j]['raw_AA']['molecule_{0}'.format(i)][start_frame:], color='black',label='RAW_AA')
            ax7.plot(data[j]['raw_DA']['molecule_{0}'.format(i)][start_frame:], color='red',label='RAW_DA')
            ax7.plot(data[j]['raw_DD']['molecule_{0}'.format(i)][start_frame:], color='green',label='RAW_DD')
            ax7.legend(loc='upper right', fontsize="5")
            ax8.hist(data[j]['raw_AA']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='black', alpha=.5, bins=25)
            ax8.hist(data[j]['raw_DA']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='red', alpha=.5, bins=25)
            ax8.hist(data[j]['raw_DD']['molecule_{0}'.format(i)][start_frame:], orientation='horizontal', color='green', alpha=.5, bins=25)
            
            ax3.set_yticks([])
            ax4.set_yticks([])
            ax6.set_yticks([])
            ax8.set_yticks([])
            ax3.set_xticks([])
            ax4.set_xticks([])
            ax6.set_xticks([])
            ax8.set_xticks([])
            
            ax1.set_xticks([])
            ax5.set_xticks([])
            ax7.set_xticks([])
            
            coor_peakA=polygon([peaksA[i,1]+radius, peaksA[i,1]+radius, peaksA[i,1]-radius, peaksA[i,1]-radius], [peaksA[i,0]+radius, peaksA[i,0]-radius, peaksA[i,0]-radius, peaksA[i,0]+radius])
            coor_ringA=polygon([peaksA[i,1]+ringradius, peaksA[i,1]+ringradius, peaksA[i,1]-ringradius, peaksA[i,1]-ringradius], [peaksA[i,0]+ringradius, peaksA[i,0]-ringradius, peaksA[i,0]-ringradius, peaksA[i,0]+ringradius])
            coor_peak_gapA=polygon([peaksA[i,1]+innerradius, peaksA[i,1]+innerradius, peaksA[i,1]-innerradius, peaksA[i,1]-innerradius], [peaksA[i,0]+innerradius, peaksA[i,0]-innerradius, peaksA[i,0]-innerradius, peaksA[i,0]+innerradius])
            coor_ring_onlyA=[i for i in list(zip(coor_ringA[0],coor_ringA[1])) if i not in list(zip(coor_peak_gapA[0],coor_peak_gapA[1]))]
            coor_ring_onlyA=[np.array([i[0] for i in coor_ring_onlyA]),np.array([i[1] for i in coor_ring_onlyA])]
            
            coor_peakD=polygon([peaksD[i,1]+radius, peaksD[i,1]+radius, peaksD[i,1]-radius, peaksD[i,1]-radius], [peaksD[i,0]+radius, peaksD[i,0]-radius, peaksD[i,0]-radius, peaksD[i,0]+radius])
            coor_ringD=polygon([peaksD[i,1]+ringradius, peaksD[i,1]+ringradius, peaksD[i,1]-ringradius, peaksD[i,1]-ringradius], [peaksD[i,0]+ringradius, peaksD[i,0]-ringradius, peaksD[i,0]-ringradius, peaksD[i,0]+ringradius])
            coor_peak_gapD=polygon([peaksD[i,1]+innerradius, peaksD[i,1]+innerradius, peaksD[i,1]-innerradius, peaksD[i,1]-innerradius], [peaksD[i,0]+innerradius, peaksD[i,0]-innerradius, peaksD[i,0]-innerradius, peaksD[i,0]+innerradius])
            coor_ring_onlyD=[i for i in list(zip(coor_ringD[0],coor_ringD[1])) if i not in list(zip(coor_peak_gapD[0],coor_peak_gapD[1]))]
            coor_ring_onlyD=[np.array([i[0] for i in coor_ring_onlyD]),np.array([i[1] for i in coor_ring_onlyD])]
            
            ax9.imshow(make_square(D_sel[coor_peakD[1], coor_peakD[0]]), cmap='Greens')
            ax10.imshow(make_square(A_sel[coor_peakA[1], coor_peakA[0]]), cmap='Reds')
            ax9.set_title('D_D Peak\nMolecule {0}'.format(i))
            ax10.set_title('A_A Peak\nMolecule {0}'.format(i))
            ax9.set_xticks([])
            ax10.set_xticks([])
            ax9.set_yticks([])
            ax10.set_yticks([])
            plt.show()
            plt.close(fig)
        except:
            continue
    try:
        if not any(data[j]['Prox_Rat']['molecule_{0}'.format(i)][start_frame:]==np.inf) and not any(data[j]['Stoichiometry']['molecule_{0}'.format(i)][start_frame:]==np.inf):
            acc_E.extend(data[j]['Prox_Rat']['molecule_{0}'.format(i)][start_frame:])
            acc_S.extend(data[j]['Stoichiometry']['molecule_{0}'.format(i)][start_frame:])
    except:
        continue
    

#%%
fig=plt.figure(constrained_layout=True)
gs=fig.add_gridspec(nrows=5, ncols=5)
ax1=fig.add_subplot(gs[1:,:-1])
ax2=fig.add_subplot(gs[0,:-1])
ax3=fig.add_subplot(gs[1:,-1])

ax1.hexbin(acc_E, acc_S, gridsize=(400,400), bins='log', extent=(-.2,1.2,-.2,1.2), cmap='jet')
ax1.set_xlabel('E')
ax1.set_ylabel('S')

ax2.hist(acc_E, bins=np.linspace(-.2,1.2,400), color='purple')
ax2.set_xticks([])
ax3.hist(acc_S, bins=np.linspace(-.2,1.2,400), orientation='horizontal', color='orange')
ax3.set_yticks([])
plt.show()
    

# =============================================================================
# #%% prep for clustering
# inrange=(np.array(acc_E)>-.2)*(np.array(acc_E)<1.2)*(np.array(acc_S)>-.2)*(np.array(acc_S)<1.2)
# 
# acc_E=np.array(acc_E)[inrange]
# acc_S=np.array(acc_S)[inrange]
# SE=[[i[0], i[1]] for i in zip(acc_S, acc_E)]
# 
# colors=['cyan', 'magenta', 'yellow']
# 
# #%%Kmeans
# clusters=KMeans(n_clusters=3, random_state=10, init='random', n_init=10, algorithm='full').fit(SE)
#     
# for i in np.unique(clusters.labels_):
#     plt.scatter(np.array(acc_E)[clusters.labels_==i], np.array(acc_S)[clusters.labels_==i], color=colors[i%3], s=1)
#     plt.scatter(clusters.cluster_centers_[:,0], clusters.cluster_centers_[:,1], color='black', s=10)
# plt.xlim(-.2, 1.2)
# plt.ylim(-.2, 1.2)
# plt.show()
#     
# #%% GaussMix
# 
# clusters=GaussianMixture(n_components=3, random_state=10, init_params='kmeans', n_init=10, ).fit(SE)
# labels=clusters.predict(SE)    
# for i in np.unique(labels):
#     plt.scatter(np.array(acc_E)[labels==i], np.array(acc_S)[labels==i], color=colors[i%3], s=1)
#     plt.scatter(clusters.means_[:,0], clusters.means_[:,1], color='black', s=10)
# plt.xlim(-.2, 1.2)
# plt.ylim(-.2, 1.2)
# plt.show()
# 
# #%%SpectralClustering
# clusters=SpectralClustering(n_clusters=3, random_state=10, n_init=10, affinity='rbf').fit(SE)
# 
# for i in np.unique(clusters.labels_):
#     plt.scatter(np.array(acc_E)[clusters.labels_==i], np.array(acc_S)[clusters.labels_==i], color=colors[i%3], s=1)
#     center_E=np.mean(np.array(acc_E)[clusters.labels_==i])
#     center_S=np.mean(np.array(acc_S)[clusters.labels_==i])
#     plt.scatter(center_E,center_S, color='black', s=10)
# plt.xlim(-.2, 1.2)
# plt.ylim(-.2, 1.2)
# plt.show()
# 
# =============================================================================

































# =============================================================================
# #%%Optics
# clusters=OPTICS(min_samples=5).fit(SE)
#     
# for i in np.unique(clusters.labels_):
#     plt.scatter(np.array(acc_E)[clusters.labels_==i], np.array(acc_S)[clusters.labels_==i], color=colors[i%3], s=1)
# # =============================================================================
# #     plt.scatter(clusters.cluster_centers_[:,0], clusters.cluster_centers_[:,1], color='black', s=10)
# # =============================================================================
# plt.xlim(-.2, 1.2)
# plt.ylim(-.2, 1.2)
# plt.show()
# =============================================================================

# =============================================================================
# #%%DBSCAN
# clusters=DBSCAN(eps=.03,min_samples=5).fit(SE)
#     
# for i in np.unique(clusters.labels_):
#     plt.scatter(np.array(acc_E)[clusters.labels_==i], np.array(acc_S)[clusters.labels_==i], color=colors[i%3], s=1)
# # =============================================================================
# #     plt.scatter(clusters.cluster_centers_[:,0], clusters.cluster_centers_[:,1], color='black', s=10)
# # =============================================================================
# plt.xlim(-.2, 1.2)
# plt.ylim(-.2, 1.2)
# plt.show()
# =============================================================================

# =============================================================================
# #%%AffinityPropagation
# clusters=AffinityPropagation(damping=.5, random_state=0).fit(SE)
#     
# for i in np.unique(clusters.labels_):
#     plt.scatter(np.array(acc_E)[clusters.labels_==i], np.array(acc_S)[clusters.labels_==i], color=colors[i%3], s=1)
#     plt.scatter(clusters.cluster_centers_[:,0], clusters.cluster_centers_[:,1], color='black', s=10)
# plt.xlim(-.2, 1.2)
# plt.ylim(-.2, 1.2)
# plt.show()
# 
# 
# 
# 
# 
# 
# =============================================================================


    




