"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector, LassoSelector, RadioButtons, Slider
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib import path

img = np.asarray(Image.open("../img/gymnastics.jpg"))
msk = (np.ones(img.shape) * 255).astype(int)
print('IMG', img.shape)


xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
idx = np.vstack((xv.flatten(), yv.flatten())).T

print('IDX', idx.shape)

fig = plt.figure(figsize=(24, 10))
axcolor = 'lightgoldenrodyellow'
ax1 = plt.subplot2grid((3, 3), (0, 0), facecolor=axcolor)
ax2 = plt.subplot2grid((3, 3), (0, 1), rowspan=3, colspan=2)
ax4 = plt.subplot2grid((3, 3), (1, 0), facecolor=axcolor)

ax2.set_xticks([]), ax2.set_yticks([])


_img = ax2.imshow(img, zorder=0, alpha=1.)
_msk = ax2.imshow(msk, origin='upper', interpolation='nearest', zorder=3, alpha=.5)

rect = None

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    global rect
    x,y = min(x1,x2), min(y1,y2)
    width, height = max(x1,x2)-x, max(y1,y2)-y
    if rect:
        rect.remove()
    rect = patches.Rectangle((x,y), width, height, color='blue', visible=True, fill=False, alpha=.5, zorder=1)
    print('RECT:', rect)
    # add rectangle
    ax2.add_patch(rect)
    ax2.draw_artist(rect)
    fig.canvas.blit(ax2.bbox)


def toggle_selector(event):
    print(' Key pressed.', event.key)
    global rect
    if event.key == 'escape' and rect:
        print('CLEAR rect!', rect)
        rect.remove()
        rect = None
        fig.canvas.draw_idle()


def radio_callback(label):
    print('selected', label)
    if label == 'rectangle':
        toggle_selector.RS.set_active(True)
        toggle_selector.LL.set_active(False)
        toggle_selector.LR.set_active(False)
        toggle_selector.DL.set_active(False)
        toggle_selector.DR.set_active(False)
        toggle_selector.ERASER.set_active(False)
    elif label == 'lasso':
        toggle_selector.RS.set_active(False)
        toggle_selector.LL.set_active(True)
        toggle_selector.LR.set_active(True)
        toggle_selector.DL.set_active(False)
        toggle_selector.DR.set_active(False)
        toggle_selector.ERASER.set_active(False)
    elif label == 'draw':
        toggle_selector.RS.set_active(False)
        toggle_selector.LL.set_active(False)
        toggle_selector.LR.set_active(False)
        toggle_selector.DL.set_active(True)
        toggle_selector.DR.set_active(True)
        toggle_selector.ERASER.set_active(False)
    elif label == 'eraser':
        toggle_selector.RS.set_active(False)
        toggle_selector.LL.set_active(False)
        toggle_selector.LR.set_active(False)
        toggle_selector.DL.set_active(False)
        toggle_selector.DR.set_active(False)
        toggle_selector.ERASER.set_active(True)


def _update_array(ind, dim):
    """draws the red or blue seeds on a white background"""
    global msk
    channels = np.arange(3)
    a,b = ind.T
    xi = channels[channels != dim]
    we = np.meshgrid(a,xi)[1]
    print('updating', msk[b,a,we].shape)
    msk[b, a, we] = 0
    return msk


def lasso_callback(dim):
    def onselect(verts):
        global radius
        p = path.Path(verts)
        ind = p.contains_points(idx, radius=radius)
        print('contains', ind[True].shape)
        global msk, _msk
        msk = _update_array(idx[ind], dim)
        _msk.set_data(msk)
        fig.canvas.draw_idle()
    return onselect


def draw_callback(dim):
    def ondraw(verts):
        global radius
        _r = [np.sum((idx-v)**2,axis=1) < radius**2 for v in verts]
        print('circles', len(_r), 'radius', radius)
        ind = np.logical_or.reduce(_r)
        global msk, _msk
        msk = _update_array(idx[ind], dim)
        _msk.set_data(msk)
        fig.canvas.draw_idle()
    return ondraw


def erase_callback(verts):
    global radius, msk, _msk
    _r = [np.sum((idx - v) ** 2, axis=1) < radius ** 2 for v in verts]
    print('circles', len(_r), 'radius', radius)
    ind = np.logical_or.reduce(_r)
    e_mask = ind.reshape(msk.shape[:-1])
    msk[e_mask] = 255
    _msk.set_data(msk)
    fig.canvas.draw_idle()


def update_radius(val):
    print('new brush radius:', val)
    global radius
    radius = val
    toggle_selector.LL.line.set_linewidth(radius)
    toggle_selector.LR.line.set_linewidth(radius)
    toggle_selector.DL.line.set_linewidth(radius)
    toggle_selector.DR.line.set_linewidth(radius)
    toggle_selector.ERASER.line.set_linewidth(radius)


lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)
lp_eraser = dict(color='white', linestyle='-', linewidth=5, alpha=0.8)

radius = 5

r_slider = Slider(ax4, 'Brush Radius', 1., 30.0, valstep=1, valinit=radius)
r_slider.on_changed(update_radius)

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(ax2, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1], minspanx=5, minspany=5,
                                       spancoords='data', interactive=True)
toggle_selector.LL = LassoSelector(ax2, lasso_callback(2), lineprops=lpl, button=[1])
toggle_selector.LR = LassoSelector(ax2, lasso_callback(0), lineprops=lpr, button=[3])
toggle_selector.DL = LassoSelector(ax2, draw_callback(2), lineprops=lpl, button=[1])
toggle_selector.DR = LassoSelector(ax2, draw_callback(0), lineprops=lpr, button=[3])
toggle_selector.ERASER = LassoSelector(ax2, erase_callback, lineprops=lp_eraser, button=[1,3])
toggle_selector.LL.set_active(False)
toggle_selector.LR.set_active(False)
toggle_selector.DL.set_active(False)
toggle_selector.DR.set_active(False)
toggle_selector.ERASER.set_active(False)

# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(ax1, ('rectangle', 'lasso', 'draw', 'eraser'), active=0)
radio.on_clicked(radio_callback)

# plt.tight_layout()
plt.connect('key_press_event', toggle_selector)
plt.show()
