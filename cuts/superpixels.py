import cv2
import maxflow
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from scipy.spatial import Delaunay


class SuperPixelCut:

    def __init__(self, img, seeds, outfile, **kwargs):
        self.img = img
        self.seeds = seeds
        # output files
        self.outfile = outfile
        self.fig_file = kwargs.get('fig_file', None)
        self.show_figure = kwargs.get('show_figure', True)
        # SLIC params
        self.n_segments = kwargs.get('n_segments', 500)
        self.compactness = kwargs.get('compactness', 20)

    def segment(self):
        img = self.img
        centers, colors_hists, segments, neighbors = self._superpixels_histograms_neighbors()
        fg_segments, bg_segments = self._find_superpixels_under_marking(self.seeds, segments)
        # get cumulative BG/FG histograms, before normalization
        fg_cumulative_hist = self._cumulative_histogram_for_superpixels(fg_segments, colors_hists)
        bg_cumulative_hist = self._cumulative_histogram_for_superpixels(bg_segments, colors_hists)
        # get histograms
        norm_hists = self._normalize_histograms(colors_hists)
        # perform graph-cut
        graph_cut = self._do_graph_cut((fg_cumulative_hist, bg_cumulative_hist),
                                       (fg_segments, bg_segments),
                                       norm_hists,
                                       neighbors)
        # segmentation plot
        plt.subplot(1, 2, 2), plt.xticks([]), plt.yticks([])
        plt.title('segmentation')
        segmask = self._pixels_for_segment_selection(segments, np.nonzero(graph_cut))
        cv2.imwrite(self.outfile, np.uint8(segmask * 255))
        plt.imshow(segmask)
        # SLIC + markings plot
        plt.subplot(1, 2, 1), plt.xticks([]), plt.yticks([])
        img = mark_boundaries(img, segments)
        img[self.seeds[:, :, 0] != 255] = (1, 0, 0)
        img[self.seeds[:, :, 2] != 255] = (0, 0, 1)
        plt.imshow(img)
        plt.title("SLIC + markings")
        # display result
        if self.fig_file:
            plt.savefig(self.fig_file, bbox_inches='tight', dpi=96)
        if self.show_figure:
            plt.show()

    # Calculate the SLIC superpixels, their histograms and neighbors
    def _superpixels_histograms_neighbors(self):
        # SLIC
        segs = slic(self.img, n_segments=self.n_segments, compactness=self.compactness)
        seg_ids = np.unique(segs)

        # centers
        centers = np.array([np.mean(np.nonzero(segs == i), axis=1) for i in seg_ids])

        # H-S histograms for all superpixels
        hsv = cv2.cvtColor(self.img.astype('float32'), cv2.COLOR_BGR2HSV)
        bins = [20, 20]  # H = S = 20
        ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]
        colors_hists = np.float32(
            [cv2.calcHist([hsv], [0, 1], np.uint8(segs == i), bins, ranges).flatten() for i in seg_ids])
        # neighbors via Delaunay tesselation
        tri = Delaunay(centers)
        return centers, colors_hists, segs, tri.vertex_neighbor_vertices

    # Get superpixels IDs for FG and BG from marking
    @staticmethod
    def _find_superpixels_under_marking(marking, superpixels):
        fg_segments = np.unique(superpixels[marking[:, :, 0] != 255])
        bg_segments = np.unique(superpixels[marking[:, :, 2] != 255])
        return fg_segments, bg_segments

    # Sum up the histograms for a given selection of superpixel IDs, normalize
    def _cumulative_histogram_for_superpixels(self, ids, histograms):
        h = np.sum(histograms[ids], axis=0)
        return h / h.sum()

    # Get a bool mask of the pixels for a given selection of superpixel IDs
    def _pixels_for_segment_selection(self, superpixels_labels, selection):
        pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
        return pixels_mask

    # Get a normalized version of the given histograms (divide by sum)
    def _normalize_histograms(self, histograms):
        return np.float32([h / h.sum() for h in histograms])

    # Perform graph cut using superpixels histograms
    def _do_graph_cut(self, fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
        num_nodes = norm_hists.shape[0]
        # Create a graph of N nodes, and estimate of 5 edges per node
        g = maxflow.Graph[float](num_nodes, num_nodes * 5)
        # Add N nodes
        nodes = g.add_nodes(num_nodes)
        hist_comp_alg = cv2.HISTCMP_KL_DIV
        # Smoothness term: cost between neighbors
        indptr, indices = neighbors
        for i in range(len(indptr) - 1):
            N = indices[indptr[i]:indptr[i + 1]]  # list of neighbor superpixels
            hi = norm_hists[i]  # histogram for center
            for n in N:
                if (n < 0) or (n > num_nodes):
                    continue
                # Create two edges (forwards and backwards) with capacities based on
                # histogram matching
                hn = norm_hists[n]  # histogram for neighbor
                g.add_edge(nodes[i], nodes[n], 20 - cv2.compareHist(hi, hn, hist_comp_alg),
                           20 - cv2.compareHist(hn, hi, hist_comp_alg))
        # Match term: cost to FG/BG
        for i, h in enumerate(norm_hists):
            if i in fgbg_superpixels[0]:
                g.add_tedge(nodes[i], 0, 1000)  # FG - set high cost to BG
            elif i in fgbg_superpixels[1]:
                g.add_tedge(nodes[i], 1000, 0)  # BG - set high cost to FG
            else:
                g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                            cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))
        g.maxflow()
        return g.get_grid_segments(nodes)
