import os
import numpy as np
import argparse
from PIL import Image
from draw import DrawingInterface
from cuts import SuperPixelCut


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default='gymnastics.jpg', help='the image to segment')
    parser.add_argument('-f', dest='folder', nargs='*', default=['img'], help='path list to the image folder path')
    parser.add_argument('--save-figure', default=None, help='if specified the result figure will be saved to this path')
    parser.add_argument('--segments', '-s', default=500, type=int, dest='n_segments',
                        help='number of segments used in SLIC')
    parser.add_argument('--edges_per_node', '-epn', default=5, type=int,
                        help='average edges per superpixel node estimation')
    parser.add_argument('--compactness', '-c', default=20, type=int, help='the compactness param used in SLIC')
    args = parser.parse_args()
    # load image
    _path = args.folder or []
    image_path = os.path.join(*_path, args.image)
    print('SOURCE IMAGE:', image_path)
    img = np.asarray(Image.open(image_path))
    # user-defined seeds
    seed_drawer = DrawingInterface(img)
    seeds = seed_drawer.run()
    # specify output files
    _name = args.image.split('.')[0]
    _outfile = os.path.join(*_path, _name + '_segmentation.png')
    # cut method
    kwargs = vars(args)
    cut = SuperPixelCut(img, seeds, _outfile, **kwargs)
    segments, mask = cut.segment()
    # save/show results
    cut.save_segmentation(mask)
    cut.plot_results()
    print('segmentation complete')
