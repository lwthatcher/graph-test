from PIL import Image
from matplotlib.widgets import LassoSelector
import numpy as np
import matplotlib.pyplot as plt

img = np.asarray(Image.open("img/gymnastics.jpg"))
print("Image shape", img.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img)



def onselect(verts):
    verts = np.round(verts)
    v2 = np.unique(verts, axis=0)
    print(verts.shape, v2.shape)
    print(v2)


lasso = LassoSelector(ax, onselect)

plt.show()

