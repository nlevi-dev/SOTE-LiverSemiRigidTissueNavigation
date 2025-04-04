# Copyright 2025 Levente Zsolt Nagy & Katalin Anna Olasz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import multiprocessing
import numpy as np
import nibabel as nib
from params import *
from util import generateSphere

KERNEL_R = 5.0  #mm

W = 256.0 #mm (constant)

A = 1 # axis      (0-2)
D = 0 # direction (0: 0->L; 1: L->0)

def processStage(inp):
    liver, stage = inp
    mask = nib.load('data/idxarr/'+liver+'/0000.nii.gz').get_fdata()
    mask = np.array(mask >= 0, np.bool_)
    raw = nib.load('data/warpstack/'+liver+'/'+stage+'.nii.gz')
    data = raw.get_fdata()
    RATIO = W/float(data.shape[0]) # mm/vox
    kr2 = KERNEL_R/RATIO*2.0       # vox
    kr = int(kr2)//2
    kr2 = kr*2+1
    data = np.linalg.norm(data, axis=3)*RATIO
    data[np.logical_not(mask)] = 0
    os.makedirs('data/length/'+liver, exist_ok=True)
    nib.save(nib.MGHImage(data,raw.get_sform(),raw.header),'data/length/'+liver+'/'+stage+'.nii.gz')
    circle = np.array(generateSphere(kr2),np.bool_)
    circle = circle[:,:,kr]
    if A == 1:
        trans0 = [1,0,2]
        trans1 = [1,0,2]
    elif A == 2:
        trans0 = [2,0,1]
        trans1 = [1,2,0]
    else:
        trans0 = [0,1,2]
        trans1 = [0,1,2]
    data = np.transpose(data,trans0)
    mask = np.transpose(mask,trans0)
    cylinder = np.repeat(np.expand_dims(circle,0), data.shape[0], 0)
    coords = np.argwhere(mask)
    res = np.zeros_like(data)
    for x, y, z in coords:
        if D == 0:
            lx = 0
            ux = x+1
        else:
            lx = x
            ux = data.shape[0]
        ly = y-kr
        uy = ly+kr2
        lz = z-kr
        uz = lz+kr2
        p_mask  = np.logical_and(cylinder[lx:ux,:,:],mask[lx:ux,ly:uy,lz:uz])
        p_data  = data[lx:ux,ly:uy,lz:uz][p_mask]
        res[x,y,z] = np.std(p_data)*2
    res = np.transpose(res,trans1)
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