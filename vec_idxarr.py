import os
import multiprocessing
import numpy as np
import nibabel as nib
import numpy as np

def matrix1(start_points, dest_points):
    A = []
    b = []
    for i in range(4):
        x, y, z = start_points[i]
        xp, yp, zp = dest_points[i]
        A.append([x, y, z, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, 0, x, y, z, 1, 0, 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, 0, 0, x, y, z, 1])
        b.append(xp)
        b.append(yp)
        b.append(zp)
    A = np.array(A)
    b = np.array(b)
    x = np.linalg.solve(A, b)
    return np.vstack([x.reshape(3, 4), [0, 0, 0, 1]])

def matrix2(start_points, dest_points):
    centroid_start = np.mean(start_points, axis=0)
    centroid_dest = np.mean(dest_points, axis=0)
    start_centered = start_points - centroid_start
    dest_centered = dest_points - centroid_dest
    U, S, Vt = np.linalg.svd(start_centered.T @ dest_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / np.sum(start_centered**2)
    T = centroid_dest - scale * (R @ centroid_start)
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = scale * R
    affine_matrix[:3, 3] = T
    return affine_matrix

def matrix3(start_points, dest_points):
    centroid_start = np.mean(start_points, axis=0)
    centroid_dest = np.mean(dest_points, axis=0)
    start_centered = start_points - centroid_start
    dest_centered = dest_points - centroid_dest
    U, _, Vt = np.linalg.svd(start_centered.T @ dest_centered)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = centroid_dest - R @ centroid_start
    rigid_matrix = np.eye(4)
    rigid_matrix[:3, :3] = R
    rigid_matrix[:3, 3] = T
    return rigid_matrix

start = [
    [146, 149, -84],  #R
    [51,  120, -176], #D1
    [207, 108, -60],  #D23
    [55,  128, -181], #V1
]
dest  = [
    [280, 216, 228],
    [89,  270, 67],
    [387, 253, 298],
    [117, 275, 77],
]

# print(np.round(matrix1(start,dest),1))
# print(np.round(matrix2(start[0:3],dest[0:3]),1))
# print(np.round(matrix3(start[0:3],dest[0:3]),1))

mat = matrix1(start,dest)

# s = np.array(list(start[0])+[1])
# print(s)
# d = np.dot(mat,s)
# print(d)
# print(dest[0])

def processLiver(liver):
    stages = os.listdir('data/points_delaunay/'+liver)
    stages = sorted(stages)
    for i in range(len(stages)):
        stage = stages[i][0:-4]
        points = np.load('data/points/'+liver+'/'+stage+'.npy')
        raw = nib.load('data/preprocessed/'+liver+'/'+stage+'.nii.gz')
        data = raw.get_fdata()
        test = np.zeros(data.shape, np.bool_)
        for p in points:
            p = np.dot(mat,np.array(list(p)+[1]))[0:3]
            print(p)
            p = np.array(np.round(p),np.uint16)
            test[p[0]-3:p[0]+3,p[1]-3:p[1]+3,p[2]-3:p[2]+3] = 1
        os.makedirs('data/test/'+liver, exist_ok=True)
        nib.save(nib.MGHImage(test,raw.get_sform(),raw.header),'data/test/'+liver+'/'+stage+'.nii.gz')

livers = os.listdir('data/points_raw')
livers = sorted(livers)

with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)