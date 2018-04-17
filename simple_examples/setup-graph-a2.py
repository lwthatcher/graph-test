import numpy as np
import scipy
from scipy.misc import imread
import maxflow
from matplotlib import pyplot as plt

a = imread("../img/a.png")
img = imread("../img/a2.png")
print(img.shape)

# variance
σ2 = np.var(img)


# region Helper Functions
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


def av_dist(_img, mask):
    result = np.empty(_img.shape)
    for index, x in np.ndenumerate(_img):
        result[index] = np.mean(_dist(x, _img[mask]))
    return result


def _dist(a,b):
    return np.around(1 - np.exp(-(((a-b)**2)/(2*σ2))), 4)
# endregion


# get seeds
seeds = a2_markings(plot=False)
S = (255 - seeds[..., 0]) / 255
T = (255 - seeds[..., 2]) / 255
print('S', np.unique(S, return_counts=True))
print('T', np.unique(T, return_counts=True))
# s/t masks
s_mask = S == 1
t_mask = T == 1
print('S mask')
print(img[s_mask])
print('T mask')
print(img[t_mask])

# define foreground/background
F = av_dist(img, s_mask)
B = av_dist(img, t_mask)

plt.imshow(F,  cmap=plt.cm.gray)
plt.show()

plt.imshow(B,  cmap=plt.cm.gray)
plt.show()

# infinite cost for marked spots
F[s_mask] = np.inf
B[t_mask] = np.inf

print(F)

# Create the graph.
g = maxflow.Graph[int]()
# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(img.shape)
# Add non-terminal edges with the same capacity.
g.add_grid_edges(nodeids, 1)

g.add_grid_tedges(nodeids, F, B)
#
#
# Find the maximum flow.
g.maxflow()
# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
# Show the result.

plt.imshow(img2, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
