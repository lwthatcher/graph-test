from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cv2 as cv
from .multi import ToggleSelector


class EvaluateCutInterface:
    def __init__(self, img, mask):
        self.img = img
        self.mask = mask
        print('EVAL', img.shape, mask.shape)
        # setup figure
        self.fig = plt.figure(2, figsize=(24, 10))
        # setup axes
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[1, 1],
                               height_ratios=[8, 1])
        self.ax1 = plt.subplot(gs[0])
        self.ax2 = plt.subplot(gs[1])
        self.ax3 = plt.subplot(gs[2])
        self.ax4 = plt.subplot(gs[3])
        # remove ticks
        self.ax1.set_xticks([]), self.ax1.set_yticks([])
        self.ax2.set_xticks([]), self.ax2.set_yticks([])
        # default result
        self.result = None

    def run(self):
        self.ax1.imshow(self.img)
        im = self.ax2.imshow(self.mask)
        # legend
        values = ['definite background', 'definite foreground', 'possible background', 'possible foreground']
        colors = [im.cmap(im.norm(i)) for i,val in enumerate(values)]
        patches = [mpatches.Patch(color=colors[i], label=val) for i,val in enumerate(values)]
        self.ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # add buttons
        btn_accept = Button(self.ax3, 'Accept')
        btn_refine = Button(self.ax4, 'Refine further')
        # callbacks
        btn_accept.on_clicked(self._on_accept)
        btn_refine.on_clicked(self._on_refine)
        # listener for key-press events
        toggle_selector = ToggleSelector(self._on_keypress)
        # start
        plt.connect('key_press_event', toggle_selector)
        plt.show()
        # return result
        return self.result

    def _on_accept(self, event):
        self.result = 'accept'
        plt.close()

    def _on_refine(self, event):
        self.result = 'refine'
        plt.close()

    def _on_keypress(self, event, *args, **kwargs):
        print(' Key pressed.', event.key)
        if event.key == 'ctrl+s':
            plt.savefig('figure.png', bbox_inches='tight')
            print('saved figure: figure.png')
        elif event.key == 'ctrl+p':
            extent = self.ax1.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            plt.savefig('plot.png', bbox_inches=extent)
            print('saved image: plot.png')
        elif event.key == 'ctrl+m':
            _mask = self.mask
            _mask[_mask == 1] = 255
            _mask[_mask == 3] = 255
            _mask[_mask == 2] = 0
            cv.imwrite('mask.png', _mask)
            print('saved mask: mask.png')
