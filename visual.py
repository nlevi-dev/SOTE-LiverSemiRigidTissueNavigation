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