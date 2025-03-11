import os
import multiprocessing
from util import dicom2nifti

def processLiver(liver):
    stages = os.listdir('data/raw/'+liver)
    stages = sorted(stages)
    for i in range(len(stages)):
        stage = stages[i]
        idx = str(i)
        while len(idx) < 4:
            idx = '0'+idx
        dicom2nifti('data/raw/'+liver+'/'+stage, 'data/nifti/'+liver+'/'+idx+'.nii.gz')

livers = os.listdir('data/raw')
livers = sorted(livers)
with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)