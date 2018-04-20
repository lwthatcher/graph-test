from __future__ import print_function
"""
Do a mouseclick somewhere, move the mouse to some destination, release
the button.  This class gives click- and release-events and also draws
a line or a box from the click-point to the actual mouseposition
(within the same axes) until the button is released.  Within the
method 'self.ignore()' it is checked whether the button from eventpress
and eventrelease are the same.

"""
from matplotlib.widgets import RectangleSelector, LassoSelector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RadioButtons
from PIL import Image
from matplotlib import path

img = np.asarray(Image.open("../img/gymnastics.jpg"))
msk = np.ones(img.shape) * 255


xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
idx = np.vstack((xv.flatten(), yv.flatten())).T

print('IDX', idx.shape)

fig = plt.figure(figsize=(10, 5))
axcolor = 'lightgoldenrodyellow'
ax1 = plt.subplot2grid((3, 5), (0, 0), facecolor=axcolor)
ax2 = plt.subplot2grid((3, 5), (0, 1), rowspan=3, colspan=2)
ax3 = plt.subplot2grid((3, 5), (0, 3), rowspan=3, colspan=2)

ax2.set_xticks([]), ax2.set_yticks([])
ax3.set_xticks([]), ax3.set_yticks([])

_img = ax2.imshow(img)
_msk = ax3.imshow(msk, origin='upper', interpolation='nearest')

rect = None


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))
    global rect
    x,y = min(x1,x2), min(y1,y2)
    width, height = max(x1,x2)-x, max(y1,y2)-y
    if rect:
        rect.remove()
    rect = patches.Rectangle((x,y), width, height, color='blue', visible=True, alpha=.5)
    print('RECT:', rect)
    # add rectangle
    ax3.add_patch(rect)
    ax3.draw_artist(rect)
    fig.canvas.blit(ax3.bbox)


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
    elif label == 'lasso':
        toggle_selector.RS.set_active(False)
        toggle_selector.LL.set_active(True)
        toggle_selector.LR.set_active(True)
    elif label == 'draw':
        toggle_selector.RS.set_active(False)
        toggle_selector.LL.set_active(False)
        toggle_selector.LR.set_active(False)


def _update_array(ind, dim):
    """draws the red or blue seeds on a white background"""
    global msk
    channels = np.arange(3)
    a,b = ind.T
    xi = channels[channels != dim]
    we = np.meshgrid(a,xi)[1]
    print('updating', len(msk[b,a,we]))
    msk[b, a, we] = 0
    return msk


def lasso_callback(dim):
    def onselect(verts):
        p = path.Path(verts)
        print('p', len(verts))
        ind = p.contains_points(idx, radius=5)
        print('contains', len(ind[True]))
        global msk, _msk
        msk = _update_array(idx[ind], dim)
        _msk.set_data(msk)
        fig.canvas.draw_idle()
    return onselect


lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(ax2, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1], minspanx=5, minspany=5,
                                       spancoords='data', interactive=True)
toggle_selector.LL = LassoSelector(ax2, lasso_callback(2), lineprops=lpl, button=[1])
toggle_selector.LR = LassoSelector(ax2, lasso_callback(0), lineprops=lpr, button=[3])
toggle_selector.LL.set_active(False)
toggle_selector.LR.set_active(False)

# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(ax1, ('rectangle', 'lasso', 'draw'), active=0)
radio.on_clicked(radio_callback)

# plt.tight_layout()
plt.connect('key_press_event', toggle_selector)
plt.show()
