import os
import numpy as np
import tensorflow as tf
from . import osvos
from .dataset import Dataset


def _sanitize_kwargs(kwargs):
    keys_to_remove = []
    for k,v in kwargs.items():
        if v is None:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del kwargs[k]
    return kwargs


def _get_frames(seq_name, frames=None):
    if frames is None:
        frames = [1, -1]
    test_frames = np.array(sorted(os.listdir(os.path.join('img', 'DAVIS', 'lbls', seq_name))))
    test_frames = test_frames[frames]
    print('TEST FRAMES', test_frames)
    test_imgs = [os.path.join('img', 'DAVIS', 'images', seq_name, frame) for frame in test_frames]
    test_imgs = [img[:-3]+'jpg' for img in test_imgs]
    return test_frames, test_imgs


# Test the network
def load_parent(seq_name, frames=None, **kwargs):
    # result path
    result_path = os.path.join('img', 'DAVIS', 'parent', seq_name)
    # Define Dataset
    test_frames, test_imgs = _get_frames(seq_name, frames)
    print('images:', test_imgs, 'masks:', test_frames)
    output_frames = [os.path.join('img', 'DAVIS', 'parent', seq_name, frame) for frame in test_frames]
    train_frames = [os.path.join('img', 'DAVIS', 'train', seq_name, frame) for frame in test_frames]
    # if they don't exist, create them
    if not all([os.path.exists(frame) for frame in output_frames]):
        print('running parent')
        dataset = Dataset(None, test_imgs, './')
        with tf.Graph().as_default():
            checkpoint_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
            osvos.test(dataset, checkpoint_path, result_path)
    # return output frames
    return output_frames, train_frames
