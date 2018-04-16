import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as plt


img = imread("../img/astronaut.png")[::2, ::2]
print('IMAGE', img.shape)

mrk = imread("../img/astronaut_marking.png")
S = mrk[..., 0]
T = mrk[..., 2]

print(mrk.shape)
print(S.shape, T.shape)
print('PERCENTS', np.unique(S, return_counts=True)[1][0] / (256*256), np.unique(T, return_counts=True)[1][0] / (256*256))
print('S', np.unique(S, return_counts=True))
print('T', np.unique(T, return_counts=True))

S2 = (255 - S) / 255
T2 = (255 - T) / 255
print('S2', np.unique(S2, return_counts=True))
print('T2', np.unique(T2, return_counts=True))

# Create the graph.
g = maxflow.Graph[int]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape)
print('NODES', nodeids.shape)
# Add non-terminal edges with the same capacity.
g.add_grid_edges(nodeids, 50)
# Add the terminal edges. The image pixels are the capacities
# of the edges from the source node. The inverted image pixels
# are the capacities of the edges to the sink node.
g.add_grid_tedges(nodeids, img, 255-img)


# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)


# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
# Show the result.

plt.imshow(img2, cmap=plt.cm.gray, interpolation='nearest')
plt.show()