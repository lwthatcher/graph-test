from matplotlib.widgets import RectangleSelector, LassoSelector, RadioButtons, Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib import path


class MultiModalInterface:
    def __init__(self, img):
        self.img = img
        # seeds mask
        self.seeds = (np.ones((img.shape[0], img.shape[1], img.shape[2] + 1)) * 255).astype(int)
        self.seeds[:, :, 3] = 0
        print("Image shape", img.shape, self.seeds.shape)
        # misc variables/constants
        self.channels = np.arange(3)
        self.radius = 5
        # pixel indices
        xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        self.idx = np.vstack((xv.flatten(), yv.flatten())).T
        # line formats
        self.lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
        self.lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)
        self.lp_eraser = dict(color='white', linestyle='-', linewidth=5, alpha=0.8)
        # initialize figures and axes
        tool_color = 'deepskyblue'
        self.fig = plt.figure(1, figsize=(24, 10))
        self.ax1 = plt.subplot2grid((3, 3), (0, 0), facecolor=tool_color)
        self.ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan=3, colspan=2)
        self.ax4 = plt.subplot2grid((3, 3), (1, 0), facecolor=tool_color)
        self.ax2.set_xticks([]), self.ax2.set_yticks([])
        # additional components
        self.rect = None
        self.toggle_selector = ToggleSelector(self._toggle_selector)
        self.r_slider = Slider(self.ax4, 'Brush Radius', 1., 30.0, valstep=1, valinit=self.radius)
        self.radio = RadioButtons(self.ax1, ('rectangle', 'lasso', 'draw', 'eraser'), active=0)
        # drawing layers
        self._img = self.ax2.imshow(img, zorder=0, alpha=1.)
        self._msk = self.ax2.imshow(self.seeds, origin='upper', interpolation='nearest', zorder=3, alpha=.5)

    def run(self):
        # setup selectors
        toggle_selector = self.toggle_selector
        toggle_selector.RS = RectangleSelector(self.ax2, self.line_select_callback,
                                               drawtype='box', useblit=True,
                                               button=[1], minspanx=5, minspany=5,
                                               spancoords='data', interactive=True)
        toggle_selector.LL = LassoSelector(self.ax2, self.lasso_callback(2), lineprops=self.lpl, button=[1])
        toggle_selector.LR = LassoSelector(self.ax2, self.lasso_callback(0), lineprops=self.lpr, button=[3])
        toggle_selector.DL = LassoSelector(self.ax2, self.draw_callback(2), lineprops=self.lpl, button=[1])
        toggle_selector.DR = LassoSelector(self.ax2, self.draw_callback(0), lineprops=self.lpr, button=[3])
        toggle_selector.ERASER = LassoSelector(self.ax2, self.erase_callback, lineprops=self.lp_eraser, button=[1, 3])
        toggle_selector.LL.set_active(False)
        toggle_selector.LR.set_active(False)
        toggle_selector.DL.set_active(False)
        toggle_selector.DR.set_active(False)
        toggle_selector.ERASER.set_active(False)
        # link additional call-backs
        self.radio.on_clicked(self.radio_callback)
        self.r_slider.on_changed(self.update_radius)
        # start
        plt.connect('key_press_event', toggle_selector)
        plt.show()

    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x, y = min(x1, x2), min(y1, y2)
        width, height = max(x1, x2) - x, max(y1, y2) - y
        if self.rect:
            self.rect.remove()
        self.rect = patches.Rectangle((x, y), width, height, color='blue', visible=True, fill=False, alpha=.5, zorder=1)
        # add rectangle
        self.ax2.add_patch(self.rect)
        self.ax2.draw_artist(self.rect)
        self.fig.canvas.blit(self.ax2.bbox)

    def radio_callback(self, label):
        print('selected', label)
        if label == 'rectangle':
            self.toggle_selector.RS.set_active(True)
            self.toggle_selector.LL.set_active(False)
            self.toggle_selector.LR.set_active(False)
            self.toggle_selector.DL.set_active(False)
            self.toggle_selector.DR.set_active(False)
            self.toggle_selector.ERASER.set_active(False)
        elif label == 'lasso':
            self.toggle_selector.RS.set_active(False)
            self.toggle_selector.LL.set_active(True)
            self.toggle_selector.LR.set_active(True)
            self.toggle_selector.DL.set_active(False)
            self.toggle_selector.DR.set_active(False)
            self.toggle_selector.ERASER.set_active(False)
        elif label == 'draw':
            self.toggle_selector.RS.set_active(False)
            self.toggle_selector.LL.set_active(False)
            self.toggle_selector.LR.set_active(False)
            self.toggle_selector.DL.set_active(True)
            self.toggle_selector.DR.set_active(True)
            self.toggle_selector.ERASER.set_active(False)
        elif label == 'eraser':
            self.toggle_selector.RS.set_active(False)
            self.toggle_selector.LL.set_active(False)
            self.toggle_selector.LR.set_active(False)
            self.toggle_selector.DL.set_active(False)
            self.toggle_selector.DR.set_active(False)
            self.toggle_selector.ERASER.set_active(True)

    def lasso_callback(self, dim):
        def onselect(verts):
            p = path.Path(verts)
            ind = p.contains_points(self.idx, radius=self.radius)
            self.seeds = self._update_array(self.idx[ind], dim)
            self._msk.set_data(self.seeds)
            self.fig.canvas.draw_idle()

        return onselect

    def draw_callback(self, dim):
        def ondraw(verts):
            _r = [np.sum((self.idx - v) ** 2, axis=1) < self.radius ** 2 for v in verts]
            ind = np.logical_or.reduce(_r)
            self.seeds = self._update_array(self.idx[ind], dim)
            self._msk.set_data(self.seeds)
            self.fig.canvas.draw_idle()
        return ondraw

    def erase_callback(self, verts):
        _r = [np.sum((self.idx - v) ** 2, axis=1) < self.radius ** 2 for v in verts]
        ind = np.logical_or.reduce(_r)
        e_mask = ind.reshape(self.seeds.shape[:-1])
        self.seeds[e_mask, 0:3] = 255
        self.seeds[e_mask, 3] = 0
        self._msk.set_data(self.seeds)
        self.fig.canvas.draw_idle()

    def update_radius(self, val):
        print('new brush radius:', val)
        self.radius = val
        self.toggle_selector.LL.line.set_linewidth(self.radius)
        self.toggle_selector.LR.line.set_linewidth(self.radius)
        self.toggle_selector.DL.line.set_linewidth(self.radius)
        self.toggle_selector.DR.line.set_linewidth(self.radius)
        self.toggle_selector.ERASER.line.set_linewidth(self.radius)

    def _update_array(self, ind, dim):
        """draws the red or blue seeds with invisible background"""
        a, b = ind.T
        xi = self.channels[self.channels != dim]
        we = np.meshgrid(a, xi)[1]
        print('updating', self.seeds[b, a, we].shape)
        self.seeds[b, a, we] = 0  # only make dim 255, other 2 color channels to 0
        self.seeds[b, a, 3] = 255  # only make these spots visible
        return self.seeds

    def _toggle_selector(self, event):
        print(' Key pressed.', event.key)
        if event.key == 'escape' and self.rect:
            print('CLEAR rect!', self.rect)
            self.rect.remove()
            self.rect = None
            self.fig.canvas.draw_idle()


class ToggleSelector:
    def __init__(self, func):
        self.func = func
        self.RS = None
        self.LL = None
        self.LR = None
        self.DL = None
        self.DR = None
        self.ERASER = None

    def __call__(self):
        return self.func


if __name__ == "__main__":
    img = np.asarray(Image.open("../img/gymnastics.jpg"))
    interface = MultiModalInterface(img)
    interface.run()
