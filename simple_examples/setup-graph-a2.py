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


def a2_markings(plot=True):
    # foreground
    fg = np.ones(a.shape) * 255
    msk1 = circle_mask(a, (35, 35), 7)
    fg[msk1] = a[msk1]
    # background
    bg = np.zeros(a.shape)
    msk2 = circle_mask(a, (45, 15), 6)
    bg[msk2] = a[msk2]
    # r/b markings
    markings = np.ones((*a.shape, 3)) * 255
    markings[fg == 0, 1:] = 0     # red
    markings[bg == 255, :-1] = 0  # blue
    # plot if specified
    if plot:
        plt.imshow(markings)
        plt.show()
    # return markings
    return markings


seeds = a2_markings()

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