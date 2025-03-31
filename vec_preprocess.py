# Copyright 2025 Levente Zsolt Nagy & Katalin Anna Olasz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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