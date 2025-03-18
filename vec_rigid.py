import os
import multiprocessing
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from params import *

KERNEL = 0.1

def processStage(inp):
    liver, stage = inp
    mask = nib.load('data/idxarr/'+liver+'/0000.nii.gz').get_fdata()
    mask = np.array(mask < 0, np.bool_)
    raw = nib.load('data/warpstack/'+liver+'/'+stage+'.nii.gz')
    data = raw.get_fdata()
    data = np.linalg.norm(data, axis=3)
    k = int(round(float(data.shape[0])*KERNEL))
    k += (k+1)%2
    data = ndimage.generic_filter(data, np.std, size=k)
    data[mask] = 0
    os.makedirs('data/rigid/'+liver, exist_ok=True)
    nib.save(nib.MGHImage(data,raw.get_sform(),raw.header),'data/rigid/'+liver+'/'+stage+'.nii.gz')

def processLiver(liver):
    stages = os.listdir('data/warpstack/'+liver)
    stages = sorted(stages)
    stages = [[liver,s[0:-7]] for s in stages]
    if DEBUG:
        stages = stages[0:1]
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        pool.map(processStage, stages)

livers = os.listdir('data/points_raw')
livers = sorted(livers)
for liver in livers:
    processLiver(liver)