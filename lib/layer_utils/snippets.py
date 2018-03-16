# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre(height, width, feat_stride=16,
                         anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """
        A wrapper function to generate anchors given different scales
        Also return the number of anchors in variable 'length'
    """

    shift_x = tf.range(width) * feat_stride
    shift_y = tf.range(height) * feat_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))

    anchors = generate_anchors(ratios=np.array(anchor_ratios),
                               scales=np.array(anchor_scales))

    A = anchors.shape[0]
    K = tf.multiply(width, height)
    length = K * A

    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

    anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)
    anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))

    return tf.cast(anchors_tf, dtype=tf.float32), length
