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
import sys
import shutil
import subprocess
import numpy as np
import scipy.ndimage as ndimage

def dicom2nifti(pathFrom, pathTo):
    if os.path.exists('TMP_dcm2niix'):
        shutil.rmtree('TMP_dcm2niix')
        os.mkdir('TMP_dcm2niix')
    else:
        os.mkdir('TMP_dcm2niix')
    subprocess.call('dcm2niix -o TMP_dcm2niix -z y {}'.format(pathFrom), shell=True)
    names = os.listdir('TMP_dcm2niix')
    for n in names:
        if n[-7:] == '.nii.gz':
            os.makedirs(pathTo[0:pathTo.rfind('/')], exist_ok=True)
            shutil.move('TMP_dcm2niix/'+n, pathTo)
            shutil.rmtree('TMP_dcm2niix')
            return

def generateSphere(volumeSize):
    x_ = np.linspace(0,volumeSize, volumeSize)
    y_ = np.linspace(0,volumeSize, volumeSize)
    z_ = np.linspace(0,volumeSize, volumeSize)
    r = int(volumeSize/2) # radius can be changed by changing r value
    center = int(volumeSize/2) # center can be changed here
    u,v,w = np.meshgrid(x_, y_, z_, indexing='ij')
    a = np.power(u-center, 2)+np.power(v-center, 2)+np.power(w-center, 2)
    b = np.where(a<=r*r,1,0)
    return b

def getDistribution(data, bins=100, excludeZero=True):
    data = data.flatten()
    if excludeZero:
        data = data[data != 0]
    return np.histogram(data,bins)

def selectLargestContour(mask):
    label, n_labels = ndimage.label(mask)
    m = np.zeros((n_labels+1,),np.uint32)
    for i in range(n_labels+1):
        m[i] = np.sum(label == i)
    idx = np.argsort(m)[-2]
    region = np.argwhere(label == idx)
    ret = np.zeros_like(mask)
    ret[region[:, 0], region[:, 1], region[:, 2]] = 1
    return ret

def findMaskBounds(mask, axis=None):
    if axis is None:
        ret = np.zeros((len(mask.shape),2),np.uint16)
        for a in range(ret.shape[0]):
            ret[a,:] = findMaskBounds(mask, a)
        return ret
    mask_zero_columns = np.where(np.sum(mask, axis=axis) == 0, sys.maxsize, 0)
    lower_bound =                    np.min(np.argmax(mask, axis=axis)                     + mask_zero_columns)
    upper_bound = mask.shape[axis] - np.min(np.argmax(np.flip(mask, axis=axis), axis=axis) + mask_zero_columns)
    return np.array([lower_bound, upper_bound],np.uint16)

def inpaint(image, mask):
    indices = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    inpainted_image = image.copy()
    inpainted_image[mask] = image[tuple(indices[:, mask])]
    return inpainted_image