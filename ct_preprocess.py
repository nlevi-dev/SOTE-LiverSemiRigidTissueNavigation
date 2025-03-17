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

        os.makedirs('data/preprocessed/'+liver, exist_ok=True)
        nib.save(nib.MGHImage(data,raw.get_sform(),raw.header),'data/preprocessed/'+liver+'/'+stage)

livers = os.listdir('data/nifti')
livers = sorted(livers)
with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)