from matplotlib.widgets import RectangleSelector, LassoSelector, RadioButtons, Slider, Button
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib import path


BLUE = (0, 0, 255, 255)
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
YELLOW = (255, 255, 0, 255)
CLEAR = (255, 255, 255, 0)
CLEAR_GREEN = (0, 255, 0, 0)
CLEAR_YELLOW = (255, 255, 0, 0)


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
        # specify figure
        self.fig = plt.figure(1, figsize=(24, 10))
        # setup axes
        gs = gridspec.GridSpec(7, 3,
                               width_ratios=[1, 1, 8],
                               height_ratios=[2, 1, 2, 1, 1, 1, 1])
        self.ax_brushes = plt.subplot(gs[0, :2], facecolor=tool_color)
        self.ax_img = plt.subplot(gs[:, 2])
        self.ax_img.set_xticks([]), self.ax_img.set_yticks([])
        self.ax_slider = plt.subplot(gs[1, :2], facecolor=tool_color)
        self.ax_nav = plt.subplot(gs[2, :2], facecolor=tool_color)
        self.ax_btn1 = plt.subplot(gs[3, 0], facecolor=tool_color)
        self.ax_btn2 = plt.subplot(gs[3, 1], facecolor=tool_color)
        self.ax_btn3 = plt.subplot(gs[4, 0], facecolor=tool_color)
        self.ax_btn4 = plt.subplot(gs[4, 1], facecolor=tool_color)
        self.ax_btn5 = plt.subplot(gs[5, :2], facecolor=tool_color)
        # additional components
        self._rects = [None for _ in imgs]
        self.toggle_selector = ToggleSelector(self._toggle_selector)
        self.slider = Slider(self.ax_slider, 'Brush Radius', 1., 30.0, valstep=1, valinit=self.radius)
        self.radio = RadioButtons(self.ax_brushes, ('rectangle', 'lasso', 'draw', 'eraser'), active=0)
        self.navs = RadioButtons(self.ax_nav, [str(i) for i in range(len(imgs))], active=0)
        self.btn1 = Button(self.ax_btn1, 'Transfer Foreground')
        self.btn2 = Button(self.ax_btn2, 'Transfer Background')
        self.btn3 = Button(self.ax_btn3, 'Toggle Suggested')
        self.btn4 = Button(self.ax_btn4, 'Clear Background')
        self.btn5 = Button(self.ax_btn5, 'Submit')
        # drawing layers
        self._img = self.ax_img.imshow(self.img, zorder=0, alpha=1.)
        self._msk = self.ax_img.imshow(self.overlay, origin='upper', interpolation='nearest', zorder=3, alpha=.5)
        # line formats
        r_axis = self.ax_img.transData.transform([[5, 0], [10, 0]])
        self._t = (r_axis[1][0] - r_axis[0][0]) / 5
        self.lpl = dict(color='blue', linestyle='-', linewidth=5*self._t, alpha=0.5)
        self.lpr = dict(color='black', linestyle='-', linewidth=5*self._t, alpha=0.5)
        self.lp_eraser = dict(color='white', linestyle='-', linewidth=5*self._t, alpha=0.8)
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
        toggle_selector.DE = LassoSelector(self.ax_img, self.on_erase, lineprops=self.lp_eraser, button=[2])
        toggle_selector.E = LassoSelector(self.ax_img, self.on_erase, lineprops=self.lp_eraser, button=[1, 3])
        toggle_selector.set_active('rectangle')
        # link additional call-backs
        self.radio.on_clicked(self.on_change_brush)
        self.navs.on_clicked(self.on_nav)
        self.slider.on_changed(self.on_change_radius)
        self.btn1.on_clicked(self._transfer_foreground)
        self.btn2.on_clicked(self._transfer_background)
        self.btn3.on_clicked(self._toggle_suggestions)
        self.btn4.on_clicked(self._clear_background)
        self.btn5.on_clicked(self._quit)
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
        print('new brush radius:', val, val*self._t)
        self.radius = val
        self.toggle_selector.update_width(self.radius*self._t)

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

    # region Buttons
    def _transfer_foreground(self, event):
        yellows = np.all(self.overlay==YELLOW, axis=-1)
        print(np.sum(np.all(self.overlay==YELLOW, axis=-1)))
        self.overlay[yellows] = BLUE
        self._msk.set_data(self.overlay)
        self.fig.canvas.draw_idle()

    def _transfer_background(self, event):
        greens = np.all(self.overlay == GREEN, axis=-1)
        print(np.sum(np.all(self.overlay == GREEN, axis=-1)))
        self.overlay[greens] = RED
        self._msk.set_data(self.overlay)
        self.fig.canvas.draw_idle()

    def _toggle_suggestions(self, event):
        yellows = np.all(self.overlay == YELLOW, axis=-1)
        greens = np.all(self.overlay == GREEN, axis=-1)
        c_greens = np.all(self.overlay == CLEAR_GREEN, axis=-1)
        c_yellows = np.all(self.overlay == CLEAR_YELLOW, axis=-1)
        has_yg = np.sum(yellows) + np.sum(greens) > 0
        has_hidden = np.sum(c_yellows) + np.sum(c_greens) > 0
        if has_yg:
            self.overlay[yellows] = CLEAR_YELLOW
            self.overlay[greens] = CLEAR_GREEN
        elif has_hidden:
            self.overlay[c_yellows] = YELLOW
            self.overlay[c_greens] = GREEN
        self._msk.set_data(self.overlay)
        self.fig.canvas.draw_idle()

    def _clear_background(self, event):
        reds = np.all(self.overlay == RED, axis=-1)
        self.overlay[reds] = CLEAR
        self._msk.set_data(self.overlay)
        self.fig.canvas.draw_idle()

    def _quit(self, event):
        plt.close(self.fig)
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
            overlay[mask == 0] = RED
            overlay[mask==1] = BLUE
            overlay[mask==2] = GREEN
            overlay[mask==3] = YELLOW
        return overlay
    # endregion

    # region Update Methods
    def _update_array(self, ind, dim):
        """draws the red or blue seeds with invisible background"""
        a, b = ind.T
        if dim == 0:
            color = RED
        else:
            color = BLUE
        self.overlay[b,a] = color
        return self.overlay

    def _toggle_selector(self, event, *args, **kwargs):
        print(' Key pressed.', event.key)
        if event.key == 'escape' and self.rect:
            print('removing rectangle', self.rect)
            self.rect.remove()
            self.rect = None
            self.fig.canvas.draw_idle()
        elif event.key == 't':
            print('axis scaling factor:', self._t)
        elif event.key == 'ctrl+o':
            cv.imwrite('overlay.png', self.overlay)
            print('saved overlay: overlay.png')
        elif event.key == 'ctrl+s':
            plt.savefig('figure.png', bbox_inches='tight')
            print('saved figure: figure.png')
        elif event.key == 'ctrl+p':
            extent = self.ax_img.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            plt.savefig('plot.png', bbox_inches=extent)
            print('saved image: plot.png')
        elif event.key == 'ctrl+m':
            _overlay = self.overlay
            yellows = np.all(_overlay == YELLOW, axis=-1)
            greens = np.all(_overlay == GREEN, axis=-1)
            _overlay[yellows] = BLUE
            _overlay[greens] = RED
            _mask = self.format_mask(_overlay, None)
            _mask[_mask == 1] = 255
            _mask[_mask == 3] = 255
            _mask[_mask == 2] = 0
            cv.imwrite('mask.png', _mask)
            print('saved mask: mask.png')
    # endregion

    # region Format Methods
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
        red = np.all(overlay == RED, axis=-1)
        blue = np.all(overlay == BLUE, axis=-1)
        mask[red] = 0  # definite BACKGROUND pixels
        mask[blue] = 1  # definite FOREGROUND pixels
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
        self.DE = None
        self.E = None

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def brushes(self):
        return [(self.RS, 'rectangle'),
                (self.LL, 'lasso'),
                (self.LR, 'lasso'),
                (self.DL, 'draw'),
                (self.DR, 'draw'),
                (self.DE, 'draw'),
                (self.E, 'eraser')]

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
