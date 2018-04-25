import os
import argparse
import numpy as np
import cv2 as cv
from osvos import extras
from PIL import Image
import matplotlib.pyplot as plt
from interface import MultiModalInterface, EvaluateCutInterface


def get_image(_img, path):
    path = path or []
    img_path = os.path.join(*path, _img)
    print('Loading source image:', img_path)
    return np.asarray(Image.open(img_path))


def load_mask(mask_path):
    mask = np.asarray(Image.open(mask_path)).astype(np.uint8)
    mask[mask == 255] = 3
    mask[mask == 0] = 2
    return mask


def save_results(results, output_frames):
    for result, path in zip(results, output_frames):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        result[result == 1] = 255
        result[result == 3] = 255
        result[result == 2] = 0
        print('saving', path, result.shape, result.dtype, np.unique(result, return_counts=True))
        plt.imshow(result)
        plt.show()
        cv.imwrite(path, result)

    print('saved:', [(path, os.path.exists(path)) for path in output_frames])


def masked_img(img, mask):
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return img * mask2[:, :, np.newaxis]


def annotate(imgs, masks=None):
    interface = MultiModalInterface(imgs, masks)
    results = interface.run()
    # iterate over results
    results = zip(imgs, results)
    result_masks = []
    for img, (rect, mask) in results:
        has_rect, has_mask = rect is not None, 0 in np.unique(mask) or 1 in np.unique(mask)
        # do grab-cut
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        if has_rect and not has_mask:
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
        result_masks.append(mask)
    return result_masks


def evaluate(imgs, masks):
    results = []
    for img, mask in zip(imgs, masks):
        img2 = masked_img(img, mask)
        eval_interface = EvaluateCutInterface(img2, mask)
        result = eval_interface.run()
        results.append(result)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='*', default=['gymnastics.jpg'], help='the image to segment')
    parser.add_argument('-f', dest='folder', nargs='*', default=['img'], help='path list to the image folder path')
    parser.add_argument('--dataset', '-D', help='the dataset to run')
    parser.add_argument('--idx', nargs='+', type=int, default=[1,-1], help='the list of indices in the data set to use')
    parser.add_argument('--iters', '-i', type=int, default=5, help='the number of iterations to run grab-cut for')
    parser.add_argument('--run-parent', '-P', action='store_true', help='whether to run the OSVOS parent first')
    parser.add_argument('--save', '-s', action='store_true', help='whether to save results')
    args = parser.parse_args()
    # load image
    if not args.dataset:
        imgs = [get_image(i, args.folder) for i in args.images]
    else:
        test_frames, test_imgs = extras._get_frames(args.dataset, args.idx)
        imgs = [np.asarray(Image.open(img_path)) for img_path in test_imgs]
    # init masks, results, and outputs to None
    results = [None for _ in imgs]
    masks = None
    output_frames = None
    # run OSVOS parent (if applicable)
    if args.dataset and args.run_parent:
        parent_frames, output_frames = extras.load_parent(args.dataset, args.idx)
        print('parent frames:', parent_frames)
        masks = [load_mask(mask_path) for mask_path in parent_frames]
    # keep running until results are agreed upon
    if masks:
        results = evaluate(imgs, masks)
    while not all([r == 'accept' for r in results]):
        masks = annotate(imgs, masks)
        results = evaluate(imgs, masks)
    # save results
    if args.save and output_frames:
        print('saving results', output_frames)
        save_results(masks, output_frames)
