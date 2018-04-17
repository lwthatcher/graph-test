import numpy as np
import scipy
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


def av_dist(_img, mask):
    def _dist(a, b):
        return np.exp(-(((a - b) ** 2) / (2 * σ2)))
    result = np.empty(_img.shape[:-1])
    for index, _ in np.ndenumerate(_img[0]):
        x = _img[index]
        result[index] = np.mean(_dist(x, _img[mask]))
    return result


s_mask = S == 1
t_mask = T == 1


F = av_dist(img, s_mask)
B = av_dist(img, t_mask)
F[s_mask] = np.inf
B[t_mask] = np.inf



# Create the graph.
g = maxflow.Graph[int]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape[:-1])
print('NODES', nodeids.shape)
# Add non-terminal edges with the same capacity.
g.add_grid_edges(nodeids, .5)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
g.add_grid_tedges(nodeids, F, B)


# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)
print('SEGMENTS', sgm.shape)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
print('IMG2', img2.shape)
# Show the result.

plt.imshow(img2, cmap=plt.cm.gray, interpolation='nearest')
plt.show()