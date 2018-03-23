import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import maxflow
from scipy.spatial import Delaunay
import argparse
from PIL import Image
from matplotlib import path
from matplotlib.widgets import LassoSelector


# region Graph-Cut Segmentation
# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segs = slic(img, n_segments=500, compactness=20)
    seg_ids = np.unique(segs)

    # centers
    centers = np.array([np.mean(np.nonzero(segs==i),axis=1) for i in seg_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segs==i), bins, ranges).flatten() for i in seg_ids])
    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)
    return centers, colors_hists, segs, tri.vertex_neighbor_vertices


# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return fg_segments, bg_segments


# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()


# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask


# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])


# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)
    hist_comp_alg = cv2.HISTCMP_KL_DIV
    # Smoothness term: cost between neighbors
    indptr, indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))
    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))
    g.maxflow()
    return g.get_grid_segments(nodes)


def segment(img, img_marking, outfile, fig_file=None, show_figure=True):
    centers, colors_hists, segments, neighbors = superpixels_histograms_neighbors(img)
    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, segments)
    # get cumulative BG/FG histograms, before normalization
    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, colors_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, colors_hists)
    # get histograms
    norm_hists = normalize_histograms(colors_hists)
    # perform graph-cut
    graph_cut = do_graph_cut((fg_cumulative_hist, bg_cumulative_hist),
                             (fg_segments, bg_segments),
                             norm_hists,
                             neighbors)
    # segmentation plot
    plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
    plt.title('segmentation')
    segmask = pixels_for_segment_selection(segments, np.nonzero(graph_cut))
    cv2.imwrite(outfile, np.uint8(segmask * 255))
    plt.imshow(segmask)
    # SLIC + markings plot
    plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
    img = mark_boundaries(img, segments)
    img[img_marking[:, :, 0] != 255] = (1, 0, 0)
    img[img_marking[:, :, 2] != 255] = (0, 0, 1)
    plt.imshow(img)
    plt.title("SLIC + markings")
    # display result
    if fig_file:
        plt.savefig(fig_file, bbox_inches='tight', dpi=96)
    if show_figure:
        plt.show()
# endregion


# region Interactive Seed Drawing
class DrawingInterface:
    def __init__(self, img):
        # setup drawing sub-figure
        self.fig = plt.figure(figsize=(24, 16))
        self.ax = self.fig.add_subplot(121)
        self.ax.imshow(img)
        # setup annotations sub-figure
        self.ax2 = self.fig.add_subplot(122)
        self.seeds = np.zeros(img.shape)
        self.msk = self.ax2.imshow(self.seeds, origin='upper', interpolation='nearest')
        print("Image shape", img.shape, self.seeds.shape)
        # get pixel indices
        xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        self.idx = np.vstack((xv.flatten(), yv.flatten())).T
        # setup line formats
        self.lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
        self.lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)

    def _update_array(self, indices, val):
        R = self.seeds[:, :, val]
        lin = np.arange(R.size)
        result = R.flatten()
        result[lin[indices]] = 255
        msk = result.reshape(R.shape).astype(int)
        self.seeds[:, :, val] = msk
        return self.seeds

    def _callback(self, side='left'):
        val = 2
        if side == 'right':
            val = 0

        def onselect(verts):
            p = path.Path(verts)
            ind = p.contains_points(self.idx, radius=5)
            print('side:', side)
            # selections
            array = self._update_array(ind, val)
            self.msk.set_data(array)
            self.fig.canvas.draw_idle()
        return onselect

    def run(self):
        self.lasso_left = LassoSelector(self.ax, self._callback('left'), lineprops=self.lpl, button=[1])
        self.lasso_right = LassoSelector(self.ax, self._callback('right'), lineprops=self.lpr, button=[3])
        plt.show()
        return self.seeds
# endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default='gymnastics.jpg', help='the image to segment')
    parser.add_argument('-f', dest='folder', nargs='*', default=['img'], help='list to the image folder path')
    parser.add_argument('--save-figure', default=None,
                        help='if specified the slic + segmentation figure will be saved to this path')
    args = parser.parse_args()
    # specify source image path
    _path = args.folder or []
    image_path = os.path.join(*_path, args.image)
    print('SOURCE IMAGE:', image_path)
    # load image
    img = np.asarray(Image.open(image_path))
    # user-defined seeds
    seed_drawer = DrawingInterface(img)
    seeds = seed_drawer.run()
    print('collected seeds:', np.count_nonzero(seeds[:,:,0]), np.count_nonzero(seeds[:,:,2]))
    # specify output files
    _name = args.image.split('.')[0]
    _outfile = os.path.join(*_path, _name + '_segmentation.png')
    # start the graph-cut segmentation
    segment(img, seeds, _outfile, fig_file=args.save_figure)
    print('segmentation complete')
