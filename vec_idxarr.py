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
from scipy.spatial import Delaunay
import scipy.ndimage as ndimage
from params import *

def pointInTetrahedron(P, tet):
    A, B, C, D = tet
    T = np.column_stack([A - D, B - D, C - D])
    v = P - D
    barycentric = np.linalg.solve(T, v)
    w = 1 - np.sum(barycentric)
    return np.all(barycentric >= 0) and np.all(barycentric <= 1) and (0 <= w <= 1)

def findTetrahedra(tetrahedra, points):
    ret = []
    for i, P in enumerate(points):
        for j, tet in enumerate(tetrahedra):
            if pointInTetrahedron(P, tet):
                ret.append(j)
                break
        if len(ret) == i:
            ret.append(-1)
        if DEBUG:
            print(str(i+1)+' / '+str(len(points)))
    return np.array(ret, np.int16)

def processStage(inp):
    liver, stage = inp
    points = np.load('data/points/'+liver+'/'+stage+'.npy')[:,0:3] / DOWNSAMPLE
    tet_idx = Delaunay(points).simplices
    tet = np.zeros(tet_idx.shape+(3,), points.dtype)
    for i in range(len(tet_idx)):
        tet[i,:] = points[tet_idx[i]]
    raw = nib.load('data/preprocessed/'+liver+'/'+stage+'.nii.gz')
    mask = np.array(raw.get_fdata())
    mask[mask > 0] = 1
    mask = np.array(mask, np.bool_)
    if DOWNSAMPLE != 1.0:
        mask = ndimage.zoom(mask, 1.0/DOWNSAMPLE, order=0)
    shape = mask.shape
    idxarr = np.full(shape, -1, np.int16)
    coords = np.zeros(shape+(3,),np.int16)
    for x in range(coords.shape[0]):
        coords[x,:,:,0] = x
    for y in range(coords.shape[1]):
        coords[:,y,:,1] = y
    for z in range(coords.shape[2]):
        coords[:,:,z,2] = z
    mask = mask.reshape((-1,))
    idxarr = idxarr.reshape((-1,))
    coords = coords.reshape((-1, 3))
    points = coords[mask]
    idxarr[mask] = findTetrahedra(tet, points)
    idxarr = idxarr.reshape(shape)
    mat = raw.get_sform()
    mat[0,0] *= DOWNSAMPLE
    mat[1,1] *= DOWNSAMPLE
    mat[2,2] *= DOWNSAMPLE
    os.makedirs('data/idxarr/'+liver, exist_ok=True)
    nib.save(nib.MGHImage(idxarr,mat,raw.header),'data/idxarr/'+liver+'/'+stage+'.nii.gz')

def processLiver(liver):
    stages = os.listdir('data/points/'+liver)
    stages = sorted(stages)
    stages = [[liver,s[0:-4]] for s in stages]
    if DEBUG:
        stages = stages[0:1]
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        pool.map(processStage, stages)

livers = os.listdir('data/points_raw')
livers = sorted(livers)
for liver in livers:
    processLiver(liver)