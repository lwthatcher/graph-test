from PIL import Image
from matplotlib.widgets import LassoSelector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path


img = np.asarray(Image.open("../img/gymnastics.jpg"))


fig = plt.figure(figsize=(24, 16))
ax = fig.add_subplot(131)
ax.imshow(img)

ax2 = fig.add_subplot(132)
array = np.zeros(img.shape)
msk = ax2.imshow(array, origin='upper', interpolation='nearest')
print("Image shape", img.shape, array.shape)

ax3 = fig.add_subplot(133)
white = np.ones(img.shape).astype(int) * 255
msk2 = ax3.imshow(white, origin='upper', interpolation='nearest')

xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
idx = np.vstack((xv.flatten(), yv.flatten())).T
channels = np.arange(3)
print('IDX', idx.shape)

lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)


def updateArray(array, indices, val):
    a,b = indices.T
    array[b,a, val] = 1.
    return array

def update_white(white, ind, val):
    a,b = ind.T
    xi = channels[channels != val]
    we = np.meshgrid(a,xi)[1]
    white[b,a, we] = 0
    return white

def select_callback(side='left'):
    val = 2
    if side == 'right':
        val = 0

    def onselect(verts):
        p = path.Path(verts)
        ind = p.contains_points(idx, radius=5)
        v0 = np.array(verts)
        v1 = np.round(verts)
        v2 = np.unique(v1, axis=0)
        print('side:', side)
        # selections
        print(v0.shape, v1.shape, v2.shape, idx[ind].shape)
        global array, white
        array = updateArray(array, idx[ind], val)
        msk.set_data(array)
        white = update_white(white, idx[ind], val)
        msk2.set_data(white)
        fig.canvas.draw_idle()
    return onselect


lasso_left = LassoSelector(ax, select_callback('left'), lineprops=lpl, button=[1])
lasso_right = LassoSelector(ax, select_callback('right'), lineprops=lpr, button=[3])

plt.show()


