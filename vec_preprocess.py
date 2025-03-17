import os
import re
import json
import multiprocessing
import numpy as np
from scipy.spatial import Delaunay

ts = ['R','D','V','P']

def processLiver(liver):
    stages = os.listdir('data/points_raw/'+liver)
    stages = sorted(stages)
    processed = [[] for _ in range(len(stages)//len(ts))]
    idxs = ['' for _ in range(len(stages)//len(ts))]
    for t in ts:
        st = [s for s in stages if s[0] == t]
        for i in range(len(st)):
            stage = st[i]
            with open('data/points_raw/'+liver+'/'+stage, 'r') as f:
                data = json.load(f)['markups'][0]['controlPoints']
            idx = re.sub('[^0-9]','',stage)
            while len(idx) < 4:
                idx = '0'+idx
            idxs[i] = idx
            processed[i] = processed[i]+[d['position'] for d in data]
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
        points = np.load('data/points/'+liver+'/'+stage)
        delu = Delaunay(points).simplices
        dels.append(delu)
        pnts.append(points)
    # dels2 = [None for _ in range(len(stages))]
    # for i in range(len(stages)):
    #     for j in range(len(stages)):
    #         cor = checkDelaunay(dels[i], pnts[j])
    #         if j == 0 and not cor:
    #             break
    #         if j > 0 and cor and dels2[j] is None:
    #             dels2[j] = i
    dels2 = [None, 1, None, 1, 1, None]
    for i in range(len(stages)):
        if dels2[i] is not None:
            os.makedirs('data/points_delaunay/'+liver, exist_ok=True)
            np.save('data/points_delaunay/'+liver+'/'+stages[i],dels[dels2[i]])

with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(delaunayLiver, livers)