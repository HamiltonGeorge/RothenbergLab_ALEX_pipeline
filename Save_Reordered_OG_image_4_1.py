# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:12:34 2024

@author: GH
"""


import os
import numpy as np
import skimage as sk
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter
import tifffile as tf
import matplotlib.pyplot as plt
#import screeninfo as si
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors
from skimage.filters import gaussian
from skimage.feature import peak_local_max
import copy

folderpath=r'Z:\rothenberglab\archive\Xue\2024 FRET paper_data analysis\04-03-2024 Alex\data'
folders = ['S2_Ch2_left7 Cy5']
prefix='spool'
suffix='.tif'


sortchannel='R'
for k in folders:
    os.chdir(folderpath + '\\' +k)
    filelist = [i for i in os.listdir('.') if i[-3:]=='tif']
    files = len(filelist)
    for i in range(files):
        if i==0:
            image=tf.imread(prefix+f'{suffix}')
        else:
            image=tf.imread(prefix+f'_{i}{suffix}')
        
        #have to sort images cuz excitation order sometimes switches???
        G=image[:,:, :int(np.shape(image)[2]/2)] #y is first dimension, x is second. Grab left
        R=image[:,:, int(np.shape(image)[2]/2):] #y is first dimension, x is second Grab right
        OG_image_sorted=copy.copy(image)
        image_Gexc=np.ones(image.shape)[::2]
        image_Rexc=np.ones(image.shape)[::2]
        if sortchannel.lower() in ['g', 'r', 'both']:
            sorted_G=copy.copy(G)
            sorted_R=copy.copy(R)
            if sortchannel.lower()=='g':
                #checks the mean deviation from the median
                tempG=copy.copy(G).astype(float)
                Gmeds = np.median(tempG, axis=(1,2))
                for l in range(len(tempG)):
                    tempG[l] = tempG[l] - Gmeds[l]
# =============================================================================
#                 threshold=np.average(tempG, axis=(0,1,2))*1.05
# =============================================================================
                thresh_window = 31
                threshold = np.convolve(np.average(tempG, axis=(1,2)), np.ones(thresh_window)/thresh_window, mode='same')
                threshold[:int((thresh_window-1)/2)] = threshold[int((thresh_window-1)/2)] 
                threshold[-int((thresh_window-1)/2):] = threshold[-(int((thresh_window-1)/2)+1)]
                Gexc=np.average(tempG, axis=(1,2))>threshold
                Rexc=np.average(tempG, axis=(1,2))<threshold
            elif sortchannel.lower()=='r':
                tempR=copy.copy(R).astype(float)
                Rmeds = np.median(tempR, axis=(1,2))
                for l in range(len(tempR)):
                    tempR[l] = tempR[l] - Rmeds[l]
                threshold=np.average(tempR, axis=(0,1,2))
                thresh_window = 31
                threshold = np.convolve(np.average(tempR, axis=(1,2)), np.ones(thresh_window)/thresh_window, mode='same')
                threshold[:int((thresh_window-1)/2)] = threshold[int((thresh_window-1)/2)] 
                threshold[-int((thresh_window-1)/2):] = threshold[-(int((thresh_window-1)/2)+1)]
                Gexc=np.average(tempR, axis=(1,2))<threshold
                Rexc=np.average(tempR, axis=(1,2))>threshold
            elif sortchannel.lower()=='both':
                #dont use
                tempGR=copy.copy((G+R)).astype(float)
                GRmeds = np.median(tempGR, axis=(1,2))
                for l in range(len(tempGR)):
                    tempGR[l] = tempGR[l] - GRmeds[l]
                threshold=np.average(tempGR, axis=(0,1,2))
                Gexc=np.average(tempGR, axis=(1,2))>threshold
                Rexc=np.average(tempGR, axis=(1,2))<threshold 
    
            for j in range(len(G[Gexc])): #box above not working, so looping explicitly
                sorted_G[2*j]=G[Gexc][j]
                sorted_G[2*j+1]=G[Rexc][j]
                sorted_R[2*j]=R[Gexc][j]
                sorted_R[2*j+1]=R[Rexc][j]
                image_Gexc[j]=image[Gexc][j]
                image_Rexc[j]=image[Rexc][j]
                OG_image_sorted[2*j]=image[Gexc][j]
                OG_image_sorted[2*j+1]=image[Rexc][j]
            G=sorted_G
            R=sorted_R
            del(sorted_G)
            del(sorted_R)
    
        if i==0:
            tf.imsave(f'{prefix}_sorted{suffix}',OG_image_sorted)
    
        else:
            tf.imsave(f'{prefix}_{i}_sorted{suffix}',OG_image_sorted)
    

