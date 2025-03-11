import os
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