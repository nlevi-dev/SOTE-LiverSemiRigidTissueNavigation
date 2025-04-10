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
from util import inpaint
from params import *

def intersectTetrahedron(tetrahedron, point):
    ret = []
    for i in range(4):
        V = tetrahedron[i]  # The vertex from which the ray originates
        plane_vertices = np.delete(tetrahedron, i, axis=0)  # The opposite face
        # Compute plane normal
        v1, v2, v3 = plane_vertices
        normal = np.cross(v2 - v1, v3 - v1)
        normal /= np.linalg.norm(normal)
        # Compute ray direction
        ray_dir = point - V
        # Plane equation: n . (X - v1) = 0 => solving for t in V + t * (P - V)
        denom = np.dot(normal, ray_dir)
        t = np.dot(normal, v1 - V) / denom
        intersection = V + t * ray_dir
        ret.append(np.linalg.norm(intersection - point))
    return np.array(ret, np.float32)

def interpolate(tetrahedra_idx, coords, points, idxarr, values):
    ret = []
    for i in range(len(points)):
        tet_values  = values[tetrahedra_idx[idxarr[i]]]
        point = points[i]
        tetrahedron = coords[tetrahedra_idx[idxarr[i]]]
        weights = intersectTetrahedron(tetrahedron, point)
        s = np.sum(weights)
        weights = np.repeat(np.expand_dims(weights,-1),3,-1)
        if np.any(np.isnan(weights)) or s == 0:
            print('Invalid weights!')
            weights = np.array([1,1,1,1],np.float32)
        ret.append(np.sum(tet_values * weights, 0) / s)
        if DEBUG:
            print(str(i+1)+' / '+str(len(points)))
    return np.array(ret, np.float32)

def processStage(inp):
    liver, T1 = inp
    T0 = str(int(T1)-1).zfill(4)
    P0 = np.load('data/points/'+liver+'/'+T0+'.npy')[:,0:3] / DOWNSAMPLE
    P1 = np.load('data/points/'+liver+'/'+T1+'.npy')[:,0:3] / DOWNSAMPLE
    tet_idx = Delaunay(P0).simplices
    raw = nib.load('data/idxarr/'+liver+'/'+T0+'.nii.gz')
    idxarr = np.array(raw.get_fdata(), np.int16)
    mask = np.array(idxarr >= 0, np.bool_)
    shape = idxarr.shape
    coords = np.zeros(shape+(3,),np.float32)
    for x in range(coords.shape[0]):
        coords[x,:,:,0] = x
    for y in range(coords.shape[1]):
        coords[:,y,:,1] = y
    for z in range(coords.shape[2]):
        coords[:,:,z,2] = z
    warp = np.zeros(shape+(3,),np.float32)
    idxarr = idxarr.reshape((-1,))
    mask   = mask.reshape((-1,))
    coords = coords.reshape((-1, 3))
    warp   = warp.reshape((-1, 3))
    idxarr = idxarr[mask]
    coords = coords[mask]
    diff = P1-P0
    warp[mask, :] = interpolate(tet_idx, P0, coords, idxarr, diff)
    warp = warp.reshape(shape+(3,))
    mask = mask.reshape(shape)
    warp = inpaint(warp, np.logical_not(mask))
    os.makedirs('data/warp/'+liver, exist_ok=True)
    nib.save(nib.MGHImage(warp,raw.get_sform(),raw.header),'data/warp/'+liver+'/'+T1+'.nii.gz')

def processLiver(liver):
    stages = os.listdir('data/points/'+liver)
    stages = sorted(stages)[1:]
    stages = [[liver,s[0:-4]] for s in stages]
    if DEBUG:
        stages = stages[0:1]
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        pool.map(processStage, stages)

livers = os.listdir('data/points_raw')
livers = sorted(livers)
for liver in livers:
    processLiver(liver)