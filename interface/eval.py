from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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
        self.ax2.imshow(self.mask)
        btn_accept = Button(self.ax3, 'Accept')
        btn_refine = Button(self.ax4, 'Refine further')
        # callbacks
        btn_accept.on_clicked(self._on_accept)
        btn_refine.on_clicked(self._on_refine)
        plt.show()
        return self.result

    def _on_accept(self, event):
        print('Accepted!', event)
        self.result = 'accept'
        plt.close()

    def _on_refine(self, event):
        print('Needs further refinement', event)
        self.result = 'refine'
        plt.close()
