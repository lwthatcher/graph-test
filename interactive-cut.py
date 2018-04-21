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
    parser.add_argument('--iters', '-i', type=int, default=5, help='the number of iterations to run grab-cut for')
    args = parser.parse_args()
    # load image
    _path = args.folder or []
    image_path = os.path.join(*_path, args.image)
    print('SOURCE IMAGE:', image_path)
    img = np.asarray(Image.open(image_path))
    # open interface
    interface = MultiModalInterface(img)
    rect, mask = interface.run()
    has_rect, has_mask = rect is not None, 0 in np.unique(mask) or 1 in np.unique(mask)
    print('rect', rect)
    print('mask', np.unique(mask, return_counts=True), 0 in np.unique(mask), 1 in np.unique(mask))
    print('has rect?', has_rect, 'has mask?', has_mask)
    # do grab-cut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    # if rectangle present, start with that
    if has_rect:
        print('Initialization with Rectangle:')
        rect_mask = np.zeros(img.shape[:2], np.uint8)
        cv.grabCut(img, rect_mask, rect, bgdModel, fgdModel, args.iters, mode=cv.GC_INIT_WITH_RECT)
        rect_mask[mask == 0] = 0
        rect_mask[mask == 1] = 1
        print(np.unique(rect_mask), rect_mask.dtype)
        mask = rect_mask
    if has_mask:
        print('Initialization with Mask:')
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, args.iters, mode=cv.GC_INIT_WITH_MASK)
    # inferred masked image
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    # draw
    plt.imshow(img), plt.show()
    plt.imshow(mask), plt.colorbar(ticks=[0,1,2,3], spacing='proportional'), plt.show()
