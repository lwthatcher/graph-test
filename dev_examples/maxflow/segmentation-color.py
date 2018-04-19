from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as plt
import datetime


img = imread("../../img/astronaut.png")[::2, ::2]
print('IMAGE', img.shape)

mrk = imread("../../img/astronaut_marking.png")
T = (255-mrk[..., 0]) / 255
S = (255-mrk[..., 2]) / 255


print(mrk.shape)
print(S.shape, T.shape)
print('PERCENTS', np.unique(S, return_counts=True)[1][0] / (256*256), np.unique(T, return_counts=True)[1][0] / (256*256))
print('S', np.unique(S, return_counts=True))
print('T', np.unique(T, return_counts=True))


# PARAMS
λf = 1.1
λb = 1
λn = 2


def roll_dist(_x, _xr, dmeter):
    x = _x.reshape((-1, 3))
    xr = _xr.reshape((-1, 3))
    dr = np.sqrt(np.sum((x-xr)**2, axis=1))
    return dmeter(dr).reshape((_x.shape[:-1]))


# t-links weights
def t_weights(x, msk):
    σ2 = np.var(x)
    s = x.shape
    def dmeter(d):
        return np.exp(-((d**2)/(2*σ2)))
    y1 = x.reshape((-1, 3))
    d1 = cdist(y1, x[msk])
    ed1 = dmeter(d1)
    e1 = np.mean(ed1,axis=1)
    return e1.reshape((s[0],s[1]))


def add_n_weights(graph, _nodeids, _img, λ=1.):
    σ2 = np.var(_img)
    rdirs = [(1,1), (-1,1), (-1,0), (1,0)]
    def dmeter(d):
        return 1 - np.exp(-((d**2)/(2*σ2)))
    def d_dist(x, x2):
        result = [cdist(x[idx].reshape(1,3), x2[idx].reshape(1,3)) for idx, _ in np.ndenumerate(x[...,0])]
        return np.array(result).reshape(x[...,0].shape)
    rolls = [np.roll(_img, *r) for r in rdirs]
    st1 = datetime.datetime.now()
    D = [d_dist(img, d) for d in rolls]
    dl, dr, du, dd = [dmeter(d)*λ for d in D]
    et1 = datetime.datetime.now()
    d_time = (et1-st1).total_seconds()
    st2 = datetime.datetime.now()
    el, er, eu, ed = [roll_dist(_img, xr, dmeter)*λ for xr in tqdm(rolls, total=len(rolls))]
    et2 = datetime.datetime.now()
    e_time = (et2 - st2).total_seconds()
    print('d time: {}'.format(d_time))
    print('e time: {}'.format(e_time))
    print(np.array_equal(dl,el), np.array_equal(dr,er), np.array_equal(du,eu), np.array_equal(dd,ed))
    # print percentiles:
    print('dl', [np.percentile(dl, i) for i in [25, 50, 75, 100]])
    print('dr', [np.percentile(dr, i) for i in [25, 50, 75, 100]])
    print('du', [np.percentile(du, i) for i in [25, 50, 75, 100]])
    print('dd', [np.percentile(dd, i) for i in [25, 50, 75, 100]])

    print('el', [np.percentile(el, i) for i in [25, 50, 75, 100]])
    print('er', [np.percentile(er, i) for i in [25, 50, 75, 100]])
    print('eu', [np.percentile(eu, i) for i in [25, 50, 75, 100]])
    print('ed', [np.percentile(ed, i) for i in [25, 50, 75, 100]])
    # add edges
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])
    graph.add_grid_edges(_nodeids, er, structure=structure, symmetric=False)
    structure = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0]])
    graph.add_grid_edges(_nodeids, el, structure=structure, symmetric=False)
    structure = np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
    graph.add_grid_edges(_nodeids, eu, structure=structure, symmetric=False)
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])
    graph.add_grid_edges(_nodeids, ed, structure=structure, symmetric=False)
    return graph


# masks
s_mask = S == 1
t_mask = T == 1

# set foreground/background weights
F = t_weights(img, s_mask)
B = t_weights(img, t_mask)
print('F/B', F.shape, B.shape, np.mean(F), np.mean(B))
print('F', np.mean(F))
print('B', np.mean(B))
F[s_mask] = np.inf
B[t_mask] = np.inf
# print('F', [np.percentile(F, i) for i in [25, 50, 75, 100]])
# print('B', [np.percentile(B, i) for i in [25, 50, 75, 100]])

# Create the graph.
g = maxflow.Graph[float]()
# Add the nodes.
nodeids = g.add_grid_nodes(img.shape[:-1])
print('NODES', nodeids.shape)
# Add non-terminal edges with respective capacities.
add_n_weights(g, nodeids, img, λ=λn)
# Add the terminal edges.
g.add_grid_tedges(nodeids, F*λf, B*λb)


# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)
print('SEGMENTS', sgm.shape, np.unique(sgm, return_counts=True))

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
print('IMG2', img2.shape)
# Show the result.

plt.imshow(img2, cmap=plt.cm.gray, interpolation='nearest')
plt.show()