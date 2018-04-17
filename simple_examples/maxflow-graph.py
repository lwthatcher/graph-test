import numpy as np
from scipy.spatial.distance import cdist
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as plt


img = imread("../img/astronaut.png")[::2, ::2]
print('IMAGE', img.shape)

mrk = imread("../img/astronaut_marking.png")
S = (255-mrk[..., 0]) / 255
T = (255-mrk[..., 2]) / 255

print(mrk.shape)
print(S.shape, T.shape)
print('PERCENTS', np.unique(S, return_counts=True)[1][0] / (256*256), np.unique(T, return_counts=True)[1][0] / (256*256))
print('S', np.unique(S, return_counts=True))
print('T', np.unique(T, return_counts=True))


# variance
σ2 = np.var(img)


# t-links weights
def t_weights(_img, mask):
    _σ2 = np.var(_img[mask])
    def _dist(a, b):
        d = cdist(a,b)
        return np.exp(-((d**2) / (2*_σ2)))
    result = np.empty(_img.shape[:-1])
    for index, _ in np.ndenumerate(_img[0]):
        x = _img[index].reshape(1,3)
        result[index] = np.mean(_dist(x, _img[mask]))
    return result


def add_n_weights(graph, _nodeids, _img):
    rdirs = [(1,1), (-1,1), (-1,0), (1,0)]
    def dmeter(d):
        return 1 - np.exp(-((d**2)/(2*σ2)))
    _il, _ir, _iu, _id = [np.roll(_img, *r) for r in rdirs]
    D = np.mean(img-_il), np.mean(img-_ir), np.mean(img-_iu), np.mean(img-_id)
    dl, dr, du, dd = [dmeter(d) for d in D]
    # print percentiles:
    print('dl', [np.percentile(dl, i) for i in [25, 50, 75, 100]])
    print('dr', [np.percentile(dr, i) for i in [25, 50, 75, 100]])
    print('du', [np.percentile(du, i) for i in [25, 50, 75, 100]])
    print('dd', [np.percentile(dd, i) for i in [25, 50, 75, 100]])
    # add edges
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])
    graph.add_grid_edges(_nodeids, dr, structure=structure, symmetric=False)
    structure = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0]])
    graph.add_grid_edges(_nodeids, dl, structure=structure, symmetric=False)
    structure = np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]])
    graph.add_grid_edges(_nodeids, du, structure=structure, symmetric=False)
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]])
    graph.add_grid_edges(_nodeids, dd, structure=structure, symmetric=False)
    return graph


# masks
s_mask = S == 1
t_mask = T == 1

# set foreground/background weights
F = t_weights(img, s_mask)
B = t_weights(img, t_mask)
print('F/B', F.shape, B.shape, np.mean(F), np.mean(B))
F[s_mask] = np.inf
B[t_mask] = np.inf
print('F', [np.percentile(F, i) for i in [25, 50, 75, 100]])
print('B', [np.percentile(B, i) for i in [25, 50, 75, 100]])

# Create the graph.
g = maxflow.Graph[float]()
# Add the nodes.
nodeids = g.add_grid_nodes(img.shape[:-1])
print('NODES', nodeids.shape)
# Add non-terminal edges with respective capacities.
add_n_weights(g, nodeids, img)
# Add the terminal edges.
g.add_grid_tedges(nodeids, F, B)


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