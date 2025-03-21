import os
import multiprocessing
import numpy as np
import nibabel as nib
from params import *
from util import generateSphere

KERNEL = 0.1

def processStage(inp):
    liver, stage = inp
    mask = nib.load('data/idxarr/'+liver+'/0000.nii.gz').get_fdata()
    mask = np.array(mask >= 0, np.bool_)
    raw = nib.load('data/warpstack/'+liver+'/'+stage+'.nii.gz')
    data = raw.get_fdata()
    data = np.linalg.norm(data, axis=3)
    k = int(round(float(data.shape[0])*KERNEL))//2
    k2 = k*2+1
    sphere = np.array(generateSphere(k2),np.bool_)
    coords = np.argwhere(mask)
    res = np.zeros_like(data)
    for x, y, z in coords:
        lx = x-k
        ly = y-k
        lz = z-k
        p_mask  = np.logical_and(sphere,mask[lx:lx+k2,ly:ly+k2,lz:lz+k2])
        p_data  = data[lx:lx+k2,ly:ly+k2,lz:lz+k2][p_mask]
        res[x,y,z] = np.std(p_data)
    os.makedirs('data/rigid/'+liver, exist_ok=True)
    nib.save(nib.MGHImage(res,raw.get_sform(),raw.header),'data/rigid/'+liver+'/'+stage+'.nii.gz')

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