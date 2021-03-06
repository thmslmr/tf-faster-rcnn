# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes,
    bbox_transform_inv_tf, clip_boxes_tf
from model.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred,
    im_info, cfg_key, _feat_stride, anchors, num_anchors):

    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

    scores = rpn_cls_prob[:, :, :, num_anchors:]
    scores = tf.reshape(scores, shape=(-1,))
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, shape=(-1, 4))

    proposals = bbox_transform_inv_tf(anchors, rpn_bbox_pred)
    proposals = clip_boxes_tf(proposals, im_info[:2])

    indices = tf.image.non_max_suppression(proposals,
    scores, max_output_size=post_nms_topN, iou_threshold=nms_thresh)

    boxes = tf.gather(proposals, indices)
    boxes = tf.to_float(boxes)
    scores = tf.gather(scores, indices)
    scores = tf.reshape(scores, shape=(-1, 1))

    batch_inds = tf.zeros((tf.shape(indices)[0], 1), dtype=tf.float32)
    blob = tf.concat([batch_inds, boxes], 1)

    return blob, scores
