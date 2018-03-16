# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def generate_anchors(base_size=16, ratios=np.array([.5, 1, 2]),
                     scales=np.array([8., 16., 32.])):

    # Get the number of anchors & Initialize anchors matrix
    num_anchors = ratios.size * scales.size
    anchors = np.zeros((num_anchors, 4))

    # Scale base_size
    scales_base = np.tile(scales, (2, ratios.size)).T
    anchors[:, 2:] = base_size * scales_base

    # Compute areas of anchors w * h
    areas = anchors[:, 2] * anchors[:, 3]

    # Apply ratios on areas & correct height
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, scales.size))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, scales.size)

    # From center, w,h to corners
    anchors[:, 0::2] -= np.tile(anchors[:, 2] / 2, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] / 2, (2, 1)).T

    return np.round(anchors)


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)

    from IPython import embed
    embed()
