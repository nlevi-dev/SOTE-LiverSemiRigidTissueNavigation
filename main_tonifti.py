import os
from util import dicom2nifti

livers = os.listdir('data/raw')
for l in livers:
    stages = os.listdir('data/raw/'+l)
    stages = sorted(stages)
    for i in range(len(stages)):
        s = stages[i]
        idx = str(i)
        while len(idx) < 4:
            idx = ' '+idx
        dicom2nifti('data/raw/'+l+'/'+s, 'data/nifti/'+l+'/'+idx+'.nii.gz')