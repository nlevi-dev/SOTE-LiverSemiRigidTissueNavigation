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
import shutil
import multiprocessing
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage
from util import inpaint
from params import *

def processLiver(liver):
    stages = os.listdir('data/warp/'+liver)
    stages = sorted(stages)
    os.makedirs('data/warpstack/'+liver, exist_ok=True)
    shutil.copy('data/warp/'+liver+'/'+stages[0],'data/warpstack/'+liver+'/'+stages[0])
    stages = [s[0:-7] for s in stages]
    stages = stages[1:]
    if DEBUG:
        stages = stages[0:1]
    mask = nib.load('data/idxarr/'+liver+'/0000.nii.gz').get_fdata()
    mask = np.array(mask < 0, np.bool_)
    for i in range(len(stages)):
        T1 = stages[i]
        T0 = str(int(T1)-1).zfill(4)
        raw = nib.load('data/warpstack/'+liver+'/'+T0+'.nii.gz')
        I0 = raw.get_fdata()
        I1 = nib.load('data/warp/'+liver+'/'+T1+'.nii.gz').get_fdata()
        coords = np.zeros(I0.shape, np.float32)
        for x in range(coords.shape[0]):
            coords[x,:,:,0] = x
        for y in range(coords.shape[1]):
            coords[:,y,:,1] = y
        for z in range(coords.shape[2]):
            coords[:,:,z,2] = z
        idxarr = coords+I0
        idxarr[mask] = 0
        idxarr = np.transpose(idxarr, [3,0,1,2])
        for j in range(I1.shape[3]):
            I1[:,:,:,j] = ndimage.map_coordinates(I1[:,:,:,j], idxarr, order=1)
        I1 = I1+I0
        I1 = inpaint(I1, mask)
        nib.save(nib.MGHImage(I1,raw.get_sform(),raw.header),'data/warpstack/'+liver+'/'+T1+'.nii.gz')

livers = os.listdir('data/points_raw')
livers = sorted(livers)
with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)