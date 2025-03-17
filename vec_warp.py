import os
import multiprocessing
import numpy as np
import nibabel as nib

DOWNSAMPLE = 4.0

DEBUG = False

def intersectTetrahedron(tetrahedron, point):
    ret = []
    for i in range(4):
        V = tetrahedron[i]  # The vertex from which the ray originates
        plane_vertices = np.delete(tetrahedron, i, axis=0)  # The opposite face
        
        # Compute plane normal
        v1, v2, v3 = plane_vertices
        normal = np.cross(v2 - v1, v3 - v1)
        normal /= np.linalg.norm(normal)  # Normalize
        
        # Compute ray direction
        ray_dir = point - V
        
        # Plane equation: n . (X - v1) = 0 => solving for t in V + t * (P - V)
        denom = np.dot(normal, ray_dir)
        
        t = np.dot(normal, v1 - V) / denom
        intersection = V + t * ray_dir

        ret.append(np.linalg.norm(intersection - point))
    ret = np.array(ret, np.float32)
    ret = 10 ** ret
    if np.any(np.isnan(ret)):
        ret[:] = 1
    return ret

def interpolate(tetrahedra_idx, points, idxarr, values):
    ret = []
    for i in range(len(points)):
        tet_values  = values[tetrahedra_idx[idxarr[i]]]
        point = points[i]
        tetrahedron = points[tetrahedra_idx[idxarr[i]]]
        weights = intersectTetrahedron(tetrahedron, point)
        s = np.sum(weights)
        weights = np.repeat(np.expand_dims(weights,-1),3,-1)
        ret.append(np.sum(tet_values * weights, 0) / s)
        # ret.append(np.sum(tet_values, 0) / 4.0)
        if DEBUG:
            print(str(i+1)+' / '+str(len(points)))
    return np.array(ret, np.float32)

def processStage(inp):
    liver, T1 = inp
    T0 = str(int(T1)-1).zfill(4)
    P0 = np.load('data/points/'+liver+'/'+T0+'.npy')[:,0:3] / DOWNSAMPLE
    P1 = np.load('data/points/'+liver+'/'+T1+'.npy')[:,0:3] / DOWNSAMPLE
    tet_idx = np.load('data/points_delaunay/'+liver+'/'+T1+'.npy')
    raw = nib.load('data/idxarr/'+liver+'/'+T1+'.nii.gz')
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
    points = coords[mask]
    diff = P1-P0
    warp[mask, :] = interpolate(tet_idx, points, idxarr, diff)
    warp = warp.reshape(shape+(3,))
    os.makedirs('data/warp/'+liver, exist_ok=True)
    nib.save(nib.MGHImage(warp,raw.get_sform(),raw.header),'data/warp/'+liver+'/'+T1+'.nii.gz')

def processLiver(liver):
    stages = os.listdir('data/points_delaunay/'+liver)
    stages = sorted(stages)
    stages = [[liver,s[0:-4]] for s in stages]
    if DEBUG:
        stages = stages[0:1]
    with multiprocessing.Pool(multiprocessing.cpu_count()-1) as pool:
        pool.map(processStage, stages)

livers = os.listdir('data/points_raw')
livers = sorted(livers)
for liver in livers:
    processLiver(liver)

# print(intersectTetrahedron(np.array([[0,0,0],[0,0,1],[0,1,1],[1,0,0]],np.float32),np.array([0.5,0,0],np.float32)))