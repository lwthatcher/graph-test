import numpy as np
import cv2


class BaseCut:

    def __init__(self, img, seeds, outfile, **kwargs):
        self.img = img
        self.seeds = seeds
        self.outfile = outfile
        # store additional key-word arguments
        self.kwargs = kwargs

    def segment(self):
        pass

    def plot_results(self):
        pass

    def save_segmentation(self, segmask):
        cv2.imwrite(self.outfile, np.uint8(segmask * 255))
