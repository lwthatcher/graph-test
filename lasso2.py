from PIL import Image
from matplotlib.widgets import LassoSelector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path


img = np.asarray(Image.open("img/gymnastics.jpg"))
print("Image shape", img.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img)

xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
idx = np.vstack((xv.flatten(), yv.flatten())).T

def onselect(verts):
    p = path.Path(verts)
    ind = p.contains_points(idx, radius=5)
    verts = np.round(verts)
    v2 = np.unique(verts, axis=0)
    print(verts.shape, v2.shape, idx[ind].shape)
    print(idx[ind])


lpl = dict(color='blue', linestyle='-', linewidth=5, alpha=0.5)
lpr = dict(color='black', linestyle='-', linewidth=5, alpha=0.5)


lasso_left = LassoSelector(ax, onselect, lineprops=lpl, button=[1])
lasso_right = LassoSelector(ax, onselect, lineprops=lpr, button=[3])

plt.show()

