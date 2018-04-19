import cv2
import maxflow
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from scipy.spatial import Delaunay
from .base import BaseCut


# noinspection PyTypeChecker
class SuperPixelCut(BaseCut):

    def __init__(self, img, seeds, outfile, **kwargs):
        super().__init__(img, seeds, outfile, **kwargs)
        # SLIC params
        self.n_segments = kwargs.get('n_segments', 500)
        self.compactness = kwargs.get('compactness', 20)
        self.epn = kwargs.get('edges_per_node', 5)

    def segment(self):
        img = self.img
        centers, colors_hists, segments, neighbors = self._superpixels_histograms_neighbors()
        fg, bg = self._find_superpixels_under_marking(self.seeds, segments)
        # get cumulative BG/FG histograms, before normalization
        fg_hist = self._cumulative_histogram_for_superpixels(fg, colors_hists)
        bg_hist = self._cumulative_histogram_for_superpixels(bg, colors_hists)
        # get histograms
        norm_hists = self._normalize_histograms(colors_hists)
        # perform graph-cut
        graph_cut = self._do_graph_cut((fg_hist, bg_hist), (fg, bg), norm_hists, neighbors)
        segmask = self._pixels_for_segment_selection(segments, np.nonzero(graph_cut))
        # store internal variables
        self.graph_cut = graph_cut
        self.segmask = segmask
        self.segments = segments
        return graph_cut, segmask

    def plot_results(self):
        # SLIC + markings plot
        img = mark_boundaries(self.img, self.segments)
        img[self.seeds[:, :, 0] != 255] = (1, 0, 0)
        img[self.seeds[:, :, 2] != 255] = (0, 0, 1)
        self.draw_figures(('segmentation', self.segmask), ('SLIC + markings', img))

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
        g = maxflow.Graph[float](num_nodes, num_nodes * self.epn)
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
