import os
import multiprocessing
import numpy as np
import nibabel as nib
from skimage import filters
import scipy.ndimage as ndimage
from util import *

def processLiver(liver):
    stages = os.listdir('data/nifti/'+liver)
    stages = sorted(stages)
    mat = None
    for i in range(len(stages)):
        stage = stages[i]

        raw = nib.load('data/nifti/'+liver+'/'+stage)
        data = raw.get_fdata()

        data = (data-np.min(data))/(np.max(data)-np.min(data))
        threshold = filters.threshold_otsu(data)

        mask = np.copy(data)
        mask[mask <  threshold] = 0
        mask[mask >= threshold] = 1
        mask = np.array(mask, np.bool_)

        s = 5
        kernel = generateSphere(s)
        mask = ndimage.binary_opening(mask, kernel)

        mask = selectLargestContour(mask)

        s = 31
        kernel = generateSphere(s)[:,:,s//2:s//2+1]
        mask = ndimage.binary_closing(mask, kernel)

        s = 11
        kernel = generateSphere(s)[:,:,s//2:s//2+1]
        mask = ndimage.binary_opening(mask, kernel)

        mask = selectLargestContour(mask)

        s = 11
        kernel = generateSphere(s)
        mask = ndimage.binary_closing(mask, kernel)

        data = data * mask

        nonzero = data.flatten()
        nonzero = nonzero[nonzero != 0]

        mean = np.mean(nonzero)
        std = np.std(nonzero)

        mi = mean-3*std
        ma = mean+3*std

        data = (data-mi)/(ma-mi)
        data[data < 0] = 0
        data[data > 1] = 1

        data = ndimage.gaussian_filter(data, sigma=0.5)

        if mat is None:
            center = ndimage.center_of_mass(mask)
            mat = np.eye(4)
            mat[0:3,3] = np.array(center) * -1

        os.makedirs('data/preprocessed/'+liver, exist_ok=True)
        nib.save(nib.MGHImage(data,mat,raw.header),'data/preprocessed/'+liver+'/'+stage)

livers = os.listdir('data/nifti')
livers = sorted(livers)
# with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
#     pool.map(processLiver, livers)

def downsampleLiver(liver):
    stages = os.listdir('data/preprocessed/'+liver)
    stages = sorted(stages)
    for i in range(len(stages)):
        stage = stages[i]

        raw = nib.load('data/preprocessed/'+liver+'/'+stage)
        data = raw.get_fdata()
        mat = raw.get_sform()

        data = ndimage.zoom(data, 0.5, order=1)
        data[data < 0] = 0
        data[data > 1] = 1
        center = mat[0:3,3]
        mat[0:3,3] = center / 2.0

        os.makedirs('data/downsampled/'+liver, exist_ok=True)
        nib.save(nib.MGHImage(data,mat,raw.header),'data/downsampled/'+liver+'/'+stage)

# with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
#     pool.map(downsampleLiver, livers)

def cropLiverPairs(liver):
    stages = os.listdir('data/downsampled/'+liver)
    stages = sorted(stages)
    for i in range(len(stages)-1):
        stage0 = stages[i]
        stage1 = stages[i+1]

        raw0 = nib.load('data/downsampled/'+liver+'/'+stage0)
        data0 = raw0.get_fdata()
        mask0 = np.zeros(data0.shape, np.bool_)
        mask0[data0 > 0] = 1
        raw1 = nib.load('data/downsampled/'+liver+'/'+stage1)
        data1 = raw1.get_fdata()
        mask1 = np.zeros(data1.shape, np.bool_)
        mask1[data1 > 0] = 1
        mat = raw0.get_sform()

        mask = np.logical_or(mask0, mask1)
        bounds = findMaskBounds(mask)

        l = bounds[:,0]
        u = bounds[:,1]
        data0 = data0[l[0]:u[0],l[1]:u[1],l[2]:u[2]]
        data1 = data1[l[0]:u[0],l[1]:u[1],l[2]:u[2]]

        center = mat[0:3,3] * -1
        center = center - l
        mat[0:3,3] = center * -1

        os.makedirs('data/cropped/'+liver, exist_ok=True)
        nib.save(nib.MGHImage(data0,mat,raw0.header),'data/cropped/'+liver+'/'+stage0[:-7]+stage0)
        nib.save(nib.MGHImage(data1,mat,raw0.header),'data/cropped/'+liver+'/'+stage0[:-7]+stage1)

with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(cropLiverPairs, livers)