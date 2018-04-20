import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.widgets import LassoSelector


class SelectionInterface:
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
