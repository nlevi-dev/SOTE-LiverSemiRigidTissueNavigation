import os
import re
import json
import multiprocessing
import numpy as np
import nibabel as nib

ts = ['R','D','V','P']

def processLiver(liver):
    stages = os.listdir('data/points_raw/'+liver)
    stages = sorted(stages)
    processed = [[] for _ in range(len(stages)//len(ts))]
    idxs = ['' for _ in range(len(processed))]
    mat = np.linalg.inv(nib.load('data/nifti/'+liver+'/0000.nii.gz').get_sform())
    for j in range(len(ts)):
        t = ts[j]
        st = [s for s in stages if s[0] == t]
        for i in range(len(st)):
            stage = st[i]
            with open('data/points_raw/'+liver+'/'+stage, 'r') as f:
                data = json.load(f)['markups'][0]['controlPoints']
            idx = re.sub('[^0-9]','',stage)
            while len(idx) < 4:
                idx = '0'+idx
            idxs[i] = idx
            processed[i] = processed[i]+[list(np.dot(mat,np.array(data[d]['position']+[1])*np.array([-1,-1,1,1]))[0:3])+[j,d] for d in range(len(data))]
    processed = np.array(processed,np.float32)
    for i in range(len(idxs)):
        os.makedirs('data/points/'+liver, exist_ok=True)
        np.save('data/points/'+liver+'/'+idxs[i],processed[i])

livers = os.listdir('data/points_raw')
livers = sorted(livers)
with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)