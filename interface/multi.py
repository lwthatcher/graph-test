from matplotlib.widgets import RectangleSelector, LassoSelector, RadioButtons, Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib import path


class MultiModalInterface:
    # region Constructor
    def __init__(self, imgs, masks=None):
        self._i = 0
        self._imgs = imgs
        self._overlays = [self._mask_to_overlay(mask) for mask in self._as_array(masks)]
        print("Images", len(imgs), self.img.shape, self.overlay.shape)
        # misc variables/constants
        self.channels = np.arange(3)
        self.radius = 5
        tool_color = 'deepskyblue'
        # pixel indices
        xv, yv = np.meshgrid(np.arange(self.img.shape[1]), np.arange(self.img.shape[0]))
        self.idx = np.vstack((xv.flatten(), yv.flatten())).T
        # line formats
        self.lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
        self.lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)
        self.lp_eraser = dict(color='white', linestyle='-', linewidth=5, alpha=0.8)
        # specify figure
        self.fig = plt.figure(1, figsize=(24, 10))
        # setup axes
        gs = gridspec.GridSpec(3, 3,
                               width_ratios=[1, 1, 8],
                               height_ratios=[2, 1, 1])
        self.ax_brushes = plt.subplot(gs[0, :2], facecolor=tool_color)
        self.ax_img = plt.subplot(gs[:, 2])
        self.ax_img.set_xticks([]), self.ax_img.set_yticks([])
        self.ax_slider = plt.subplot(gs[1, :2], facecolor=tool_color)
        self.ax_nav = plt.subplot(gs[2, :2], facecolor=tool_color)
        # additional components
        self._rects = [None for _ in imgs]
        self.toggle_selector = ToggleSelector(self._toggle_selector)
        self.slider = Slider(self.ax_slider, 'Brush Radius', 1., 30.0, valstep=1, valinit=self.radius)
        self.radio = RadioButtons(self.ax_brushes, ('rectangle', 'lasso', 'interface', 'eraser'), active=0)
        self.navs = RadioButtons(self.ax_nav, [str(i) for i in range(len(imgs))], active=0)
        # drawing layers
        self._img = self.ax_img.imshow(self.img, zorder=0, alpha=1.)
        self._msk = self.ax_img.imshow(self.overlay, origin='upper', interpolation='nearest', zorder=3, alpha=.5)
    # endregion

    # region Property Accessors
    @property
    def img(self):
        return self._imgs[self._i]

    @property
    def overlay(self):
        return self._overlays[self._i]

    @overlay.setter
    def overlay(self, value):
        self._overlays[self._i] = value

    @property
    def rect(self):
        return self._rects[self._i]

    @rect.setter
    def rect(self, value):
        self._rects[self._i] = value
    # endregion

    # region Public Methods
    def run(self):
        rs_kwargs = {'drawtype': 'box', 'useblit': True, 'button': [1], 'minspanx': 5, 'minspany': 5,
                     'spancoords': 'pixels', 'interactive': True}
        # setup selectors
        toggle_selector = self.toggle_selector
        toggle_selector.RS = RectangleSelector(self.ax_img, self.on_rect, **rs_kwargs)
        toggle_selector.LL = LassoSelector(self.ax_img, self.on_lasso(2), lineprops=self.lpl, button=[1])
        toggle_selector.LR = LassoSelector(self.ax_img, self.on_lasso(0), lineprops=self.lpr, button=[3])
        toggle_selector.DL = LassoSelector(self.ax_img, self.on_draw(2), lineprops=self.lpl, button=[1])
        toggle_selector.DR = LassoSelector(self.ax_img, self.on_draw(0), lineprops=self.lpr, button=[3])
        toggle_selector.ERASER = LassoSelector(self.ax_img, self.on_erase, lineprops=self.lp_eraser, button=[1, 3])
        toggle_selector.set_active('rectangle')
        # link additional call-backs
        self.radio.on_clicked(self.on_change_brush)
        self.navs.on_clicked(self.on_nav)
        self.slider.on_changed(self.on_change_radius)
        # start
        plt.connect('key_press_event', toggle_selector)
        plt.show()
        # noinspection PyTypeChecker
        return [(self.format_rect(r), self.format_mask(o, r)) for r, o in zip(self._rects, self._overlays)]
    # endregion

    # region Callbacks
    def on_rect(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        x, y = min(x1, x2), min(y1, y2)
        w, h = max(x1, x2) - x, max(y1, y2) - y
        if self.rect:
            self.rect.remove()
        self.rect = patches.Rectangle((x, y), w, h, color='blue', visible=True, fill=False, alpha=.5, zorder=1)
        # add rectangle
        self.ax_img.add_patch(self.rect)
        self.ax_img.draw_artist(self.rect)
        self.fig.canvas.blit(self.ax_img.bbox)

    def on_change_brush(self, label):
        print('mode:', label)
        self.toggle_selector.set_active(label)

    def on_change_radius(self, val):
        print('new brush radius:', val)
        self.radius = val
        self.toggle_selector.update_width(self.radius)

    def on_lasso(self, dim):
        def onselect(verts):
            p = path.Path(verts)
            ind = p.contains_points(self.idx, radius=self.radius)
            self.overlay = self._update_array(self.idx[ind], dim)
            self._msk.set_data(self.overlay)
            self.fig.canvas.draw_idle()

        return onselect

    def on_draw(self, dim):
        def ondraw(verts):
            _r = [np.sum((self.idx - v) ** 2, axis=1) < self.radius ** 2 for v in verts]
            ind = np.logical_or.reduce(_r)
            self.overlay = self._update_array(self.idx[ind], dim)
            self._msk.set_data(self.overlay)
            self.fig.canvas.draw_idle()
        return ondraw

    def on_erase(self, verts):
        _r = [np.sum((self.idx - v) ** 2, axis=1) < self.radius ** 2 for v in verts]
        ind = np.logical_or.reduce(_r)
        e_mask = ind.reshape(self.overlay.shape[:-1])
        self.overlay[e_mask, 0:3] = 255
        self.overlay[e_mask, 3] = 0
        self._msk.set_data(self.overlay)
        self.fig.canvas.draw_idle()

    def on_nav(self, i):
        i = int(i)
        print('selecting image', i)
        self._i = i
        self._img.set_data(self.img)
        self._msk.set_data(self.overlay)
        for j, rect in enumerate(self._rects):
            if rect is not None:
                rect.set_visible(j == i)
        self.fig.canvas.draw_idle()
    # endregion

    # region Update Methods
    def _update_array(self, ind, dim):
        """draws the red or blue seeds with invisible background"""
        a, b = ind.T
        xi = self.channels[self.channels != dim]
        we = np.meshgrid(a, xi)[1]
        print('updating', self.overlay[b, a, we].shape)
        self.overlay[b, a, we] = 0  # only make dim 255, other 2 color channels to 0
        self.overlay[b, a, 3] = 255  # only make these spots visible
        return self.overlay

    def _toggle_selector(self, event):
        print(' Key pressed.', event.key)
        if event.key == 'escape' and self.rect:
            print('removing rectangle', self.rect)
            self.rect.remove()
            self.rect = None
            self.fig.canvas.draw_idle()
    # endregion

    # region Helper Methods
    def _as_array(self, masks):
        if masks is None:
            masks = [None for _ in self._imgs]
        return masks

    def _new_overlay(self):
        shape = np.array(self.img.shape)
        shape[2] += 1
        overlay = (np.ones(shape) * 255).astype(int)
        overlay[:, :, 3] = 0
        return overlay

    def _mask_to_overlay(self, mask):
        overlay = self._new_overlay()
        if mask is not None:
            overlay[(mask==1) | (mask==3)] = (0,0,255,255)
        # for now, don't transfer background
        return overlay
    # endregion

    # Format Methods
    @staticmethod
    def format_rect(rect):
        if rect is None:
            return None
        result = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()
        return tuple(int(d) for d in result)

    @classmethod
    def format_mask(cls, overlay, rect=None):
        mask = np.ones(overlay.shape[:2], np.uint8) * 3  # set all unmarked as possible foreground
        if rect is not None:
            mask[:] = 0  # default to definite background
            x,y,w,h = cls.format_rect(rect)
            mask[y:y+h, x:x+w] = 3  # anything in the rectangle might be foreground
        mask[overlay[:, :, 2] != 255] = 0  # definite BACKGROUND pixels
        mask[overlay[:, :, 0] != 255] = 1  # definite FOREGROUND pixels
        return mask
    # endregion


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

    @property
    def brushes(self):
        return [(self.RS, 'rectangle'),
                (self.LL, 'lasso'),
                (self.LR, 'lasso'),
                (self.DL, 'interface'),
                (self.DR, 'interface'),
                (self.ERASER, 'eraser')]

    def set_active(self, label):
        for brush, _type in self.brushes:
            brush.set_active(_type == label)

    def update_width(self, width):
        for brush, _type in self.brushes:
            if hasattr(brush, 'line'):
                brush.line.set_linewidth(width)


if __name__ == "__main__":
    img = np.asarray(Image.open("../img/gymnastics.jpg"))
    interface = MultiModalInterface([img])
    interface.run()
