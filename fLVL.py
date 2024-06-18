# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import ks_2samp
from PIL import Image
from scipy.ndimage import label
import cv2



def fLVL(A):
    # Convert A to double
    A = A.astype(float)
    sz = A.shape
    mnsz = min(sz)  # minimum scale of the field
    szA = sz[0] * sz[1]
    p = np.sum(A) / szA  # cloud fraction

    # estimating the ideal length to capture almost 100% of the histogram
    # we want that the error < exp(-12) for the perfect rand case
    mxln = int(np.floor(abs(12 / np.log(p))) + 1)
    mxln = min(mxln, mnsz)

    # The cloud part
    # Flatenning along the two directions
    # then we need to divide c1 and c2 by two
    
    B = A.flatten()  # rows
    C = A.flatten(order='F') #columns 
    B = np.concatenate((C, B))
    
    L, num_labels = label(B)
    # Label connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))
    
    # Extract area of each connected component
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Adding one to the max area
    mx_ar = np.max(areas) + 1   

    # To avoid single histogram
    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar
    
    # Get the cloud chord length counts - c1    
    c1, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    # Correct for flattening in two directions
    c1 = c1 / 2
    s1 = np.sum(c1)
    
    # Now, the theortical calculations ct, nt for a given cloud fraction (p)
    nt1 = np.arange(1, mxln + 1)
    ct1 = (szA * (1 - p) ** 2) * p ** nt1
    st1 = np.sum(ct1)
    nt1 = nt1.astype(int)
    
    
    # Get the KS score for the cloud part
    adf1 = np.abs(np.cumsum(ct1 / st1) - np.cumsum(c1 / s1))
    KS1 = np.max(adf1)


    # The void part
    q = 1 - p # void fraction
    mxln = int(np.floor(abs(12 / np.log(q))) + 1)
    mxln = min(mxln, mnsz)
    B = -B + 1
    
    L, num_labels = label(B)
    # Label connected components in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8))
    
    # Extract area of each connected component
    areas = stats[1:, cv2.CC_STAT_AREA]
    # Adding one to the max area
    mx_ar = np.max(areas) + 1   
    
    if mx_ar < mxln:
        mx_ar = mxln
    else:
        mxln = mx_ar

    c2, _ = np.histogram(areas, bins=np.arange(1, mx_ar + 2))
    c2 = c2 / 2
    s2 = np.sum(c2)

    nt2 = np.arange(1, mxln + 1)
    ct2 = (szA * (1 - q) ** 2) * q ** nt2
    st2 = np.sum(ct2)
    nt2 = nt2.astype(int)
    
    # Get the KS score for the void part
    adf2 = np.abs(np.cumsum(ct2 / st2) - np.cumsum(c2 / s2))
    KS2 = np.max(adf2)

    return KS1, KS2