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
import multiprocessing
from util import dicom2nifti

def processLiver(liver):
    stages = os.listdir('data/raw/'+liver)
    stages = sorted(stages)
    for i in range(len(stages)):
        stage = stages[i]
        idx = str(i)
        while len(idx) < 4:
            idx = '0'+idx
        dicom2nifti('data/raw/'+liver+'/'+stage, 'data/nifti/'+liver+'/'+idx+'.nii.gz')

livers = os.listdir('data/raw')
livers = sorted(livers)
with multiprocessing.Pool(multiprocessing.cpu_count()-2) as pool:
    pool.map(processLiver, livers)