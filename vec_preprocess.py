import os
import re
import json
import multiprocessing
import numpy as np
from scipy.spatial import Delaunay
import nibabel as nib

ts = ['R','D','V','P']

def processLiver(liver):
    stages = os.listdir('data/points_raw/'+liver)
    stages = sorted(stages)
    processed = [[] for _ in range(len(stages)//len(ts))]
    idxs = ['' for _ in range(len(processed))]
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
            mat = np.linalg.inv(nib.load('data/nifti/'+liver+'/'+idx+'.nii.gz').get_sform())
            processed[i] = processed[i]+[list(np.dot(mat,np.array(data[d]['position']+[1])*np.array([-1,-1,1,1]))[0:3])+[j,d] for d in range(len(data))]
    processed = np.array(processed,np.float16)
    for i in range(len(idxs)):
        os.makedirs('data/points/'+liver, exist_ok=True)
        np.save('data/points/'+liver+'/'+idxs[i],processed[i])

livers = os.listdir('data/points_raw')
livers = sorted(livers)
with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)

def checkDelaunay(polygs, points):
    for i in range(len(polygs)):
        for j in range(len(points)):
            if j not in polygs[i] and Delaunay(points[polygs[i]]).find_simplex(points[j]) >= 0:
                return False
    return True

def delaunayLiver(liver):
    stages = os.listdir('data/points/'+liver)
    stages = sorted(stages)
    dels = []
    pnts = []
    for i in range(len(stages)):
        stage = stages[i]
        points = np.load('data/points/'+liver+'/'+stage)[:,0:3]
        delu = Delaunay(points).simplices
        dels.append(delu)
        pnts.append(points)
    dels2 = []
    for i in range(len(stages)):
        tmp = []
        for j in range(len(stages)):
            cor = checkDelaunay(dels[i], pnts[j])
            if j == 0 and not cor:
                tmp = [False for _ in range(len(stages))]
                break
            tmp.append(cor)
        dels2.append(tmp)
    dels2 = np.array(dels2, np.uint8)
    score = np.sum(dels2, axis=1)
    ranks = np.argsort(score)[::-1]
    dels3 = [None for _ in range(len(stages))]
    for i in range(len(stages)):
        for j in range(len(stages)):
            if dels3[j] is None and dels2[ranks[i],j] == 1:
                dels3[j] = dels2[ranks[i],j]
    dels3[0] = None
    for i in range(len(stages)):
        if dels3[i] is not None:
            os.makedirs('data/points_delaunay/'+liver, exist_ok=True)
            np.save('data/points_delaunay/'+liver+'/'+stages[i],dels[dels3[i]])

with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(delaunayLiver, livers)