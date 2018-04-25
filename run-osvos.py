"""
Taken from the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Sergi Caelles (scaelles@vision.ee.ethz.ch)
Please consider citing the paper if you use this code.
"""

import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
slim = tf.contrib.slim

# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
from osvos import osvos
from osvos.dataset import Dataset
os.chdir(root_folder)

# define acceptable sequences
_path = os.path.join('img', 'DAVIS', 'images')
known_sequences = next(os.walk(_path))[1]


def run_demo(seq_name, max_training_iters=200, **kwargs):
    # User Defined parameters
    side_supervision = kwargs.get('side_supervision', 1)

    # define paths
    result_path = os.path.join('img', 'DAVIS', 'results', seq_name)
    parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
    logs_path = os.path.join('models', seq_name)

    # Train parameters
    learning_rate = kwargs.get('learning_rate', 1e-8)
    save_step = kwargs.get('save_step', max_training_iters)
    display_step = kwargs.get('display_step', 10)

    # Define Dataset
    test_frames = sorted(os.listdir(os.path.join('img', 'DAVIS', 'images', seq_name)))
    test_imgs = [os.path.join('img', 'DAVIS', 'images', seq_name, frame) for frame in test_frames]
    train_frames = sorted(os.listdir(os.path.join('img', 'DAVIS', 'train', seq_name)))
    train_imgs = [os.path.join('img', 'DAVIS', 'train', seq_name, frame) for frame in train_frames]
    dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)

    # Train the network
    with tf.Graph().as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)
        osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                             save_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

    # Test the network
    with tf.Graph().as_default():
        checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        osvos.test(dataset, checkpoint_path, result_path)


def _sanitize_kwargs(kwargs):
    keys_to_remove = []
    for k,v in kwargs.items():
        if v is None:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del kwargs[k]
    return kwargs


if __name__ == "__main__":
    # define parser
    parser = argparse.ArgumentParser('parameters for running OSVOS')
    parser.add_argument('seq_name', choices=known_sequences, help='the name of the sequence to run')
    parser.add_argument('--epochs', '-t', dest='max_training_iters', default=200, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--supervision', dest='side_supervision', choices=[1, 2, 3], type=int,
                        help='Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision')
    parser.add_argument('--learning-rate', '-lr', type=float, help='the base learning rate')
    parser.add_argument('--save-step', '-ss', type=int, help='number of iterations to run before saving')
    # parse arguments
    args = parser.parse_args()
    seq = args.seq_name
    kwargs = vars(args)
    del kwargs['seq_name']
    kwargs = _sanitize_kwargs(kwargs)
    print('Parameters:', kwargs)
    # run learner
    run_demo(seq, **kwargs)
