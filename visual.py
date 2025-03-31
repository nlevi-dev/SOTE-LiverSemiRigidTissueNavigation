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

try:
    tmp = get_ipython().__class__.__name__
    if tmp == 'ZMQInteractiveShell':
        import matplotlib.pyplot as plt
        def pltshow(_):
            plt.show()
    else:
        raise Exception('err')
except:
    from matplotlib_terminal import plt
    def pltshow(renderer):
        plt.show(renderer)
        plt.close()

def showSlices(data, slices=None):
    if slices is None:
        slices = [data.shape[0]//2,data.shape[1]//2,data.shape[2]//2]
    fig, (p0,p1,p2) = plt.subplots(1,3)
    fig.suptitle('')
    fig.set_size_inches(16, 7)
    p0.imshow(data[slices[0],:,:])
    p1.imshow(data[:,slices[1],:])
    p2.imshow(data[:,:,slices[2]])
    pltshow()

def showDistribution(hist):
    fig, p = plt.subplots(1,1)
    fig.suptitle('')
    fig.set_size_inches(8, 7)
    bins, edges = hist
    if len(bins) == len(edges):
        bins = bins[0:len(bins)-1]
    p.stairs(bins,edges,fill=True,color='blue')
    pltshow()