import os
import multiprocessing
import numpy as np
import nibabel as nib
import numpy as np

def processLiver(liver):
    stages = os.listdir('data/points_delaunay/'+liver)
    stages = sorted(stages)
    raw = nib.load('data/preprocessed/'+liver+'/0000.nii.gz')
    for i in range(len(stages)):
        stage = stages[i][0:-4]
        points = np.load('data/points/'+liver+'/'+stage+'.npy')
        test = np.zeros(raw.get_fdata().shape, np.bool_)
        for p in points:
            p = np.array(np.round(p),np.uint16)
            test[p[0]-3:p[0]+3,p[1]-3:p[1]+3,p[2]-3:p[2]+3] = 1
        os.makedirs('data/test/'+liver, exist_ok=True)
        nib.save(nib.MGHImage(test,raw.get_sform(),raw.header),'data/test/'+liver+'/'+stage+'.nii.gz')

livers = os.listdir('data/points_raw')
livers = sorted(livers)

with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)