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

def delaunaySelect(mat, idx):
    idxarr = []
    for i in range(len(mat)):
        if idx+i < len(mat):
            idxarr.append(idx+i)
        if idx-i-1 >= 0:
            idxarr.append(idx-i-1)
    idxarr = np.array(idxarr)
    for i in range(len(mat)):
        if np.sum(mat[idxarr[i],idx-1:idx+1]) == 2:
            return idxarr[i]
    return None

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
            tmp.append(cor)
        dels2.append(tmp)
        print(str(i+1)+' / '+str(len(stages)))
    dels2 = np.array(dels2, np.uint8)
    print(dels2)
    dels3 = [None]+[delaunaySelect(dels2, i) for i in range(1, len(stages))]
    for i in range(len(stages)):
        if dels3[i] is not None:
            os.makedirs('data/points_delaunay/'+liver, exist_ok=True)
            np.save('data/points_delaunay/'+liver+'/'+stages[i],dels[dels3[i]])

with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(delaunayLiver, livers)