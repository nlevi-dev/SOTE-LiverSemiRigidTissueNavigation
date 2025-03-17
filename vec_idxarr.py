import os
import multiprocessing
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage

DOWNSAMPLE = 4.0

def pointInTetrahedron(P, A, B, C, D):
    T = np.column_stack([A - D, B - D, C - D])
    v = P - D
    barycentric = np.linalg.solve(T, v)
    w = 1 - np.sum(barycentric)
    return np.all(barycentric >= 0) and np.all(barycentric <= 1) and (0 <= w <= 1)

def findTetrahedra(tetrahedra, points):
    ret = []
    for i, P in enumerate(points):
        for j, tet in enumerate(tetrahedra):
            A, B, C, D = tet
            if pointInTetrahedron(P, A, B, C, D):
                ret.append(j)
                break
        if len(ret) == i:
            ret.append(-1)
    return np.array(ret, np.int16)

def processStage(inp):
    liver, T1 = inp
    T0 = str(int(T1)-1).zfill(4)
    P0 = np.load('data/points/'+liver+'/'+T0+'.npy')[:,0:3] / DOWNSAMPLE
    P1 = np.load('data/points/'+liver+'/'+T1+'.npy')[:,0:3] / DOWNSAMPLE
    tet_idx = np.load('data/points_delaunay/'+liver+'/'+T1+'.npy')
    tet = np.zeros(tet_idx.shape+(3,), P0.dtype)
    for i in range(len(tet_idx)):
        tet[i,:] = P0[tet_idx[i]]
    raw = nib.load('data/preprocessed/'+liver+'/'+T0+'.nii.gz')
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
    nib.save(nib.MGHImage(idxarr,mat,raw.header),'data/idxarr/'+liver+'/'+T0+'.nii.gz')

def processLiver(liver):
    stages = os.listdir('data/points_delaunay/'+liver)
    stages = sorted(stages)
    stages = [[liver,s[0:-4]] for s in stages]
    stages = stages[0:1]
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        pool.map(processStage, stages)

livers = os.listdir('data/points_raw')
livers = sorted(livers)
for liver in livers:
    processLiver(liver)