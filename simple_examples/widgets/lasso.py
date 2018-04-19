from matplotlib.widgets import LassoSelector
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

x, y = 4*(np.random.rand(2, 100) - .5)
ax.plot(x, y, 'o')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)


def onselect(verts):
    print(len(verts))
lasso = LassoSelector(ax, onselect)

plt.show()

