import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from matplotlib import path
from matplotlib.widgets import LassoSelector
from cuts.superpixels import SuperPixelCut


# region Interactive Seed Drawing
class DrawingInterface:
    def __init__(self, img):
        # setup drawing sub-figure
        self.fig = plt.figure(figsize=(24, 16))
        self.ax = self.fig.add_subplot(121)
        self.ax.imshow(img)
        # setup annotations sub-figure
        self.ax2 = self.fig.add_subplot(122)
        self.seeds = np.ones(img.shape).astype(int) * 255  # white background
        self.msk = self.ax2.imshow(self.seeds, origin='upper', interpolation='nearest')
        print("Image shape", img.shape, self.seeds.shape)
        # get pixel indices
        xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        self.idx = np.vstack((xv.flatten(), yv.flatten())).T
        # setup line formats
        self.lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
        self.lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)
        # RGB channels
        self.c = np.arange(3)

    def _update_array(self, ind, dim):
        """draws the red or blue seeds on a white background"""
        a,b = ind.T
        xi = self.c[self.c != dim]
        we = np.meshgrid(a,xi)[1]
        self.seeds[b, a, we] = 0
        return self.seeds

    def _callback(self, dim):
        def onselect(verts):
            p = path.Path(verts)
            ind = p.contains_points(self.idx, radius=5)
            # selections
            array = self._update_array(self.idx[ind], dim)
            self.msk.set_data(array)
            self.fig.canvas.draw_idle()
        return onselect

    def run(self):
        self.lasso_left = LassoSelector(self.ax, self._callback(2), lineprops=self.lpl, button=[1])
        self.lasso_right = LassoSelector(self.ax, self._callback(0), lineprops=self.lpr, button=[3])
        plt.show()
        return self.seeds
# endregion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default='gymnastics.jpg', help='the image to segment')
    parser.add_argument('-f', dest='folder', nargs='*', default=['img'], help='list to the image folder path')
    parser.add_argument('--save-figure', default=None, help='if specified the result figure will be saved to this path')
    parser.add_argument('--segments', '-s', default=500, type=int, dest='n_segments',
                        help='number of segments used in SLIC')
    parser.add_argument('--edges_per_node', '-epn', default=5, type=int,
                        help='average edges per superpixel node estimation')
    parser.add_argument('--compactness', '-c', default=20, type=int, help='the compactness param used in SLIC')
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
    # TODO: fix seed counts
    print('collected seeds:', np.count_nonzero(seeds[:,:,0]), np.count_nonzero(seeds[:,:,2]))
    # specify output files
    _name = args.image.split('.')[0]
    _outfile = os.path.join(*_path, _name + '_segmentation.png')
    # start the graph-cut segmentation
    d_args = vars(args)
    gc = SuperPixelCut(img, seeds, _outfile, **d_args)
    gc.segment()
    print('segmentation complete')
