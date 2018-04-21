import os
import argparse
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from draw import MultiModalInterface


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default='gymnastics.jpg', help='the image to segment')
    parser.add_argument('-f', dest='folder', nargs='*', default=['img'], help='path list to the image folder path')
    args = parser.parse_args()
    # load image
    _path = args.folder or []
    image_path = os.path.join(*_path, args.image)
    print('SOURCE IMAGE:', image_path)
    img = np.asarray(Image.open(image_path))
    # open interface
    interface = MultiModalInterface(img)
    rect, seeds = interface.run()
    print('rect', rect)
    print('seeds', seeds.shape)
    if rect:
        print('rect info:', rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height())
        print('rect info:', rect.get_bbox().extents)
        r = int(rect.get_x()), int(rect.get_y()), int(rect.get_width()), int(rect.get_height())
        # setup grab-cut
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv.grabCut(img, mask, r, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        plt.imshow(img), plt.colorbar(), plt.show()
