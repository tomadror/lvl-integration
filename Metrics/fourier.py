#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import fftpack, ndimage, linalg
from .utils import findFiles, getField, rSquared

def getRad(data):
    h  = data.shape[0];  hc = h//2
    w  = data.shape[1];  wc = w//2
   
    # create an array of integer radial distances from the center
    Y, X = np.ogrid[0:h, 0:w]
    r    = np.hypot(X - wc, Y - hc).astype(np.int)
    
    return r

def getPsd1D(psd2D):
    r  = getRad(psd2D)
    sh = np.min(psd2D.shape)//2
    N  = np.ones(psd2D.shape)

    # SUM all psd2D pixels with label 'r' for 0<=r<=wc
    # Will miss power contributions in 'corners' r>wc
    rran    = np.arange(1, sh+1)
    psd1D   = ndimage.sum(psd2D, r, index=rran)
    Ns      = ndimage.sum(N    , r, index=rran)
    psd1D   = psd1D/Ns*rran*2*np.pi

    return psd1D

def getPsd1DAz(psd2D, dTheta=5, rMin=0, rMax=255):
    h  = psd2D.shape[0]
    w  = psd2D.shape[1]
    wc = w//2
    hc = h//2
    
    # note that displaying PSD as image inverts Y axis
    # create an array of integer angular slices of dTheta
    Y, X  = np.ogrid[0:h, 0:w]
    theta = np.rad2deg(np.arctan2(-(Y-hc), (X-wc)))
    theta = np.mod(theta + dTheta/2 + 360, 360)
    theta = dTheta * (theta//dTheta)
    theta = theta.astype(np.int)
    
    # mask below rMin and above rMax by setting to -100
    R     = np.hypot(-(Y-hc), (X-wc))
    mask  = np.logical_and(R > rMin, R < rMax)
    theta = theta + 100
    theta = np.multiply(mask, theta)
    theta = theta - 100
    
    # SUM all psd2D pixels with label 'theta' for 0<=theta<360 between rMin 
    # and rMax
    angF  = np.arange(0, 360, int(dTheta))
    psd1D = ndimage.sum(psd2D, theta, index=angF)
    
    # normalize each sector to the total sector power
    pwrTotal = np.sum(psd1D)
    psd1D    = psd1D/pwrTotal
    
    return angF, psd1D

def detrend(data, regressors):
    # From https://neurohackweek.github.io/image-processing/02-detrending/
    regressors = np.vstack([r.ravel() for r in regressors]).T
    solution = linalg.lstsq(regressors, data.ravel())
    beta_hat = solution[0]
    trend = np.dot(regressors, beta_hat)
    detrended = data - np.reshape(trend, data.shape)
    return detrended, beta_hat

def hannRad(data):
    r = getRad(data)
    sh  = np.min(data.shape)
    shc = sh//2
     
    fac  = (3.*np.pi/8 - 2/np.pi)**(-0.5)
    # fac  = 1
    whan = fac*(1 + np.cos(2*np.pi*r/sh))
    whan[r>=shc] = 0
    
    return data*whan    

def welchRad(data):
    r = getRad(data)
    sh = np.min(data.shape)
    shc = sh//2

    wwel = 1 - (r/shc)**2
    wwel[r>=shc] = 0
    
    return data*wwel

def planckRad(data,eps=0.1):
    r = getRad(data)
    N = np.min(data.shape)
    N2 = N//2
    epsN = int(eps*N)
    n = N2 - r
    
    wPlanck = np.zeros(data.shape)
    wPlanck[(n<epsN) & (n>=1)] = (1+np.exp(epsN/n[(n<epsN) & (n>=1)] - epsN/
                                          (epsN - n[(n<epsN) & (n>=1)])))**(-1)
    wPlanck[epsN<=n] = 1

    return data*wPlanck    

class FourierMetrics():
    '''
    Class for computing metrics from Fourier-transformed fields. Can compute 
    spectral slope directly (beta) or from bin-averaged data (betaa), as well
    as spectral anisotropy and a spectral length scale (de Roode et al. 2004).
    
    Parameters
    ----------
    mpar : Dict (optional, but necessary for using the compute method)
       Specifies the following parameters:
           loadPath : Path to load .h5 files that contain a pandas dataframe
                      with a cloud mask field as one of the columns.
           savePath : Path to a .h5 containing a pandas dataframe whose columns
                      contain metrics and whose indices are scenes. Two of 
                      these columns can be filled by 'beta', 'betaa', 'azVar'
                      and 'specL'.
           save     : Boolean to specify whether to store the variables in
                      savePath/Metrics.h5
           resFac   : Resolution factor (e.g. 0.5), to coarse-grain the field.
           plot     : Boolean to specify whether to make plot with details on
                      this metric for each scene.
           con      : Connectivitiy for segmentation (1 - 4 seg, 2 - 8 seg)
           areaMin  : Minimum cloud size considered in computing metric
           fMin     : First scene to load
           fMax     : Last scene to load. If None, is last scene in set.
           fields   : Naming convention for fields, used to set the internal
                      field to be used to compute each metric. Must be of the 
                      form:
                           {'cm'  : CloudMaskName, 
                            'im'  : imageName, 
                            'cth' : CloudTopHeightName,
                            'cwp' : CloudWaterPathName}
                     
    '''
    def __init__(self, mpar=None):
        # Metric-specific parameters
        self.field  = 'Cloud_Mask_1km'
        self.window = 'Planck'         # Choose in [Planck, Welch, Hann, None]
        self.detrend= True
        self.k0     = 0                # Zero bin length
        self.kMax   = 8                # Max wavenumber
        self.nBin   = 10
        self.bins   = np.logspace(self.k0,self.kMax,self.nBin+1,base=2)
        self.binsA  = np.exp((np.log(self.bins[1:]) + np.log(self.bins[:-1]))/2)
        self.plot   = False
        
        # General parameters
        if mpar is not None:
            self.loadPath = mpar['loadPath']
            self.savePath = mpar['savePath']
            self.save     = mpar['save']
            self.saveExt  = mpar['saveExt']
            self.resFac   = mpar['resFac']
            self.plot     = mpar['plot']
            self.con      = mpar['con']
            self.areaMin  = mpar['areaMin']
            self.fMin     = mpar['fMin']
            self.fMax     = mpar['fMax']
            self.field    = mpar['fields']['cm']

    def metric(self,field):
        '''
        Compute metric(s) for a single field

        Parameters
        ----------
        field : numpy array of shape (npx,npx) - npx is number of pixels
            Cloud mask field.

        Returns
        -------
        beta : float
            Directly computed spectral slope.
        betaa : float
            Bin-averaged spectral slope.
        azVar : float
            Measure of variance in the azimuthal power spectrum (anisotropy)
        lSpec : float
            Spectral length scale (de Roode et al, 2004)

        '''
        # Spectral analysis - general observations
        # Windowing   : Capturing more information is beneficial 
        #               (Planck>Welch>Hann window)
        # Detrending  : Mostly imposes unrealistic gradients
        # Using image : Less effective at reproducing trends in 2D org plane 
        # Binning     : Emphasises lower k and ignores higher k
        # Assumptions : 2D Fourier spectrum is isotropic
        #               Impact of small-scale errors is small
        
        [X,Y] = np.meshgrid(np.arange(field.shape[0]),
                            np.arange(field.shape[1]), indexing='ij')
        
        # Detrend
        if self.detrend:
            field,bDt  = detrend(field,[X,Y])
        
        # Windowing
        if self.window == 'Planck':
            field = planckRad(field)        # Planck-taper window
        elif self.window == 'Welch':
            field = welchRad(field)         # Welch window
        elif self.window == 'Hann':
            field = hannRad(field)          # Hann window  
        
        # FFT
        F       = fftpack.fft2(field)       # 2D FFT (no prefactor)
        F       = fftpack.fftshift(F)       # Shift so k0 is centred
        psd2    = np.abs(F)**2/\
                  np.prod(field.shape)      # Get the energy-preserving 2D PSD
        psd1    = getPsd1D(psd2)            # Azimuthal integral-> 1D PSD
        psd1Az  = getPsd1DAz(psd2)          # Radial integral -> Sector 1D PSD
        azVar   = 2*(np.max(psd1Az[1]) - 
                     np.min(psd1Az[1]))     # Spectrum anisotropy (0-1)
        
        # Direct beta
        k1d     = np.arange(1,len(psd1)+1)  
        beta,b0 = np.polyfit(np.log(k1d),
                             np.log(psd1),1)# Spectral slope beta
        rSqb    = rSquared(np.log(k1d),
                           np.log(psd1),
                           [beta,b0])       # rSquared of the fit
        
        # Average over bins
        mns = np.zeros(len(self.bins)-1); sts = np.zeros(len(self.bins-1))
        for i in range(len(self.bins)-1):
            imax   = np.where(k1d <  self.bins[i+1])[0][-1]
            imin   = np.where(k1d >= self.bins[i])[0]
            if len(imin) == 0:
                continue # You have gone beyond the available wavenumbers
            else:
                imin = imin[0]
            if imin == imax:
                psdi = psd1[imin]
            else:
                psdi   = psd1[imin:imax]
            mns[i] = np.mean(psdi)
            sts[i] = np.std (psdi)
        
        binsA = self.binsA[mns!=0]
        mns   = mns[mns!=0]
        
        # betaa
        betaa,b0a = np.polyfit(np.log(binsA[1:-1]),np.log(mns[1:-1]),1)        # Spectral slope beta
        rSqba = rSquared(np.log(binsA[1:-1]),np.log(mns[1:-1]),[betaa,b0a])    # rSquared of the fit
        
        # Spectral length scale as de Roode et al. (2004), using true median
        sumps = np.cumsum(psd1); sumps/=sumps[-1]
        kcrit = np.where(sumps>1/2)[0][0]
        lSpec = 1./kcrit
        
        # Plotting
        if self.plot:
            fig,axs = plt.subplots(ncols=3,figsize=(12,4))
            axs[0].imshow(field,'gray'); axs[0].axis('off')
            axs[0].set_title('Clouds')
            axs[1].imshow(np.log(psd2)); axs[1].axis('off')
            axs[1].set_title('2D PSD - Anisotropy: %.3f' %azVar)
            axs[2].scatter(np.log(k1d),np.log(psd1),s=2.5,c='k')
            axs[2].plot(np.log(k1d),b0+beta*np.log(k1d),c='k')
            axs[2].scatter(np.log(binsA),np.log(mns),s=2.5,c='C1')
            axs[2].plot(np.log(binsA),b0a+betaa*np.log(binsA),c='C1')
            axs[2].annotate('Direct',(0.7,0.9), xycoords='axes fraction',
                            fontsize=10)
            axs[2].annotate(r'$R^2$='+str(round(rSqb,3)),(0.7,0.8), 
                            xycoords='axes fraction',fontsize=10)
            axs[2].annotate(r'$\beta=$'+str(round(beta,3)),(0.7,0.7), 
                            xycoords='axes fraction',fontsize=10)
            axs[2].annotate(r'$\Lambda=$'+str(round(lSpec,3)),(0.7,0.6), 
                            xycoords='axes fraction',fontsize=10)
            axs[2].annotate('Bin-averaged',(0.4,0.9), xycoords='axes fraction',
                            color='C1',fontsize=10)
            axs[2].annotate(r'$R^2$='+str(round(rSqba,3)),(0.4,0.8), 
                            xycoords='axes fraction',color='C1',fontsize=10)
            axs[2].annotate(r'$\beta_a=$'+str(round(betaa,3)),(0.4,0.7), 
                            xycoords='axes fraction',color='C1',fontsize=10)
            axs[2].set_xlabel(r'$\ln k$',fontsize=10)
            axs[2].set_ylabel(r'$\ln E(k)$',fontsize=10)
            axs[2].grid()
            axs[2].set_title('1D Spectrum')
            plt.tight_layout()
            plt.show()
        
        return beta, betaa, azVar, lSpec
        
    def verify(self):
        return 'Verification not implemented for FourierMetrics'
        
    def compute(self):
        '''
        Main loop over scenes. Loads fields, computes metric, and stores it.

        '''
        files, dates = findFiles(self.loadPath)
        files = files[self.fMin:self.fMax]
        dates = dates[self.fMin:self.fMax]

        if self.save:
            saveSt    = self.saveExt
            dfMetrics = pd.read_hdf(self.savePath+'/Metrics'+saveSt+'.h5')
        
        ## Main loop over files
        for f in range(len(files)):
            cm = getField(files[f], self.field, self.resFac, binary=False)
            print('Scene: '+files[f]+', '+str(f+1)+'/'+str(len(files)))
            
            beta, betaa, azVar, lSpec  = self.metric(cm)
            print('beta:   ', beta)
            print('betaa:  ', betaa)
            print('azVar:  ', azVar)
            print('lSpec:  ', lSpec)

            if self.save:
                dfMetrics['beta'].loc[dates[f]]     = beta
                dfMetrics['betaa'].loc[dates[f]]    = betaa
                dfMetrics['psdAzVar'].loc[dates[f]] = azVar
                dfMetrics['specL'].loc[dates[f]]    = lSpec
        
        if self.save:
            dfMetrics.to_hdf(self.savePath+'/Metrics'+saveSt+'.h5', 'Metrics',
                             mode='w')        
        
if  __name__ == '__main__':
    mpar = {
            'loadPath' : '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Filtered',
            'savePath' : '/Users/martinjanssens/Documents/Wageningen/Patterns-in-satellite-images/testEnv/Data/Metrics',
            'save'     : True, 
            'resFac'   : 1,     # Resolution factor (e.g. 0.5)
            'plot'     : True,  # Plot with details on each metric computation
            'con'      : 1,     # Connectivity for segmentation (1:4 seg, 2:8 seg)
            'areaMin'  : 4,     # Minimum cloud size considered for object metrics
            'fMin'     : 0,     # First scene to load
            'fMax'     : None,  # Last scene to load. If None, is last scene in set
           }
    fourierMetrics = FourierMetrics(mpar)
    fourierMetrics.verify()
    fourierMetrics.compute()
   
    