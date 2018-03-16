from PIL import Image
from matplotlib.widgets import LassoSelector
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path


img = np.asarray(Image.open("img/gymnastics.jpg"))


fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(img)

ax2 = fig.add_subplot(122)
array = np.zeros(img.shape[:-1])
msk = ax2.imshow(array, origin='upper', vmin=-10, vmax=10, interpolation='nearest')
print("Image shape", img.shape, array.shape)

xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
idx = np.vstack((xv.flatten(), yv.flatten())).T


lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)


def updateArray(array, indices, val):
    lin = np.arange(array.size)
    newArray = array.flatten()
    newArray[lin[indices]] = val
    return newArray.reshape(array.shape)


def select_callback(side='left'):
    val = 10
    if side == 'right':
        val = -10

    def onselect(verts):
        p = path.Path(verts)
        ind = p.contains_points(idx, radius=5)
        v0 = np.array(verts)
        v1 = np.round(verts)
        v2 = np.unique(v1, axis=0)
        print('side:', side)
        # selections
        print(v0.shape, v1.shape, v2.shape, idx[ind].shape)
        global array
        array = updateArray(array, ind, val)
        msk.set_data(array)
        fig.canvas.draw_idle()


        # draw the line again, just for fun!
        # line = Line2D(v0[:,0], v0[:,1], **lpl)
        # ax2.add_line(line)
        # ax2.draw_artist(line)
        # line.set_visible(True)
        # fig.canvas.draw_idle()
        # print(line, line.get_data()[0].shape)
    return onselect


lasso_left = LassoSelector(ax, select_callback('left'), lineprops=lpl, button=[1])
lasso_right = LassoSelector(ax, select_callback('right'), lineprops=lpr, button=[3])

plt.show()

print(array)

