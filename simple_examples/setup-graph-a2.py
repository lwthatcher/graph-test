import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as plt

a = imread("../img/a.png")
img = imread("../img/a2.png")

print(a.shape)


def circle_mask(_img, origin, r):
    a, b = origin
    n0, n1 = _img.shape
    y,x = np.ogrid[-a:n0-a, -b:n1-b]
    mask = x*x + y*y <= r*r
    return mask


# create figure
fig = plt.figure(figsize=(24, 16))

# A
ax = fig.add_subplot(131)
ax.imshow(a, cmap=plt.cm.gray, interpolation='nearest')
# foreground
ax = fig.add_subplot(132)
fg = np.ones(a.shape) * 255
msk = circle_mask(a, (35,35), 7)
fg[msk] = a[msk]
ax.imshow(fg, cmap=plt.cm.gray, interpolation='nearest')
# background
ax = fig.add_subplot(133)
bg = np.zeros(a.shape)
msk = circle_mask(a, (45,15), 6)
bg[msk] = a[msk]
ax.imshow(bg, cmap=plt.cm.gray, interpolation='nearest')
plt.show()



# # Create the graph.
# g = maxflow.Graph[int]()
# # Add the nodes. nodeids has the identifiers of the nodes in the grid.
# nodeids = g.add_grid_nodes(img.shape)
# # Add non-terminal edges with the same capacity.
# g.add_grid_edges(nodeids, 50)
# # Add the terminal edges. The image pixels are the capacities
# # of the edges from the source node. The inverted image pixels
# # are the capacities of the edges to the sink node.
# g.add_grid_tedges(nodeids, img, 255-img)
#
#
# # Find the maximum flow.
# g.maxflow()
# # Get the segments of the nodes in the grid.
# sgm = g.get_grid_segments(nodeids)
#
#
# # The labels should be 1 where sgm is False and 0 otherwise.
# img2 = np.int_(np.logical_not(sgm))
# # Show the result.
#
# plt.imshow(img2, cmap=plt.cm.gray, interpolation='nearest')
# plt.show()