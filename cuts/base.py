import cv2
import numpy as np
import matplotlib.pyplot as plt


class BaseCut:

    def __init__(self, img, seeds, outfile, **kwargs):
        self.img = img
        self.seeds = seeds
        self.outfile = outfile
        # display options
        self.fig_file = kwargs.get('fig_file', None)
        self.show_figure = kwargs.get('show_figure', True)
        # store additional key-word arguments
        self.kwargs = kwargs

    def segment(self):
        pass

    def plot_results(self):
        pass

    def save_segmentation(self, segmask):
        cv2.imwrite(self.outfile, np.uint8(segmask * 255))

    def draw_figures(self, *figures, draw_ticks=False):
        fig, axes = plt.subplots(1, len(figures))
        for f, ax in zip(figures, axes):
            title, img = f
            ax.set_title(title)
            if draw_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            ax.imshow(img)
        # display result
        if self.fig_file:
            plt.savefig(self.fig_file, bbox_inches='tight', dpi=96)
        if self.show_figure:
            plt.show()
