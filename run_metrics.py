import tensorflow as tf
from tensorflow.python.ops import array_ops
import keras.backend as K
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""

    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)

    # dice loss
    # intersection = tf.reduce_sum(labels*preds,axis=1)
    # union = tf.reduce_sum(labels,axis=1) + tf.reduce_sum(preds,axis=1) + 1e-5
    # dice = 1. - (2*intersection/union)
    # loss = 1-tf.reduce_mean(dice)
    # return loss

# GHM  https://github.com/GXYM/GHM_Loss
def ghm_class_loss(logits, targets, masks=None):
    """ Args:
    input [batch_num, class_num]:
        The direct prediction of classification fc layer.
    target [batch_num, class_num]:
        Binary target (0 or 1) for each sample each class. The value is -1
        when the sample is ignored.
    """
    train_mask = (1 - tf.cast(tf.equal(targets, -1), dtype=tf.float32))
    g_v = tf.abs(tf.sigmoid(logits) - targets)  # [batch_num, class_num]
    g = tf.expand_dims(g_v, axis=0)  # [1, batch_num, class_num]

    if masks is None:
        masks = tf.ones_like(targets)
    valid_mask = masks > 0
    weights, tot = calc(g, valid_mask)
    ghm_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets * train_mask,
                                                             logits=logits)
    ghm_class_loss = tf.reduce_sum(ghm_class_loss * weights) / tot

    return ghm_class_loss
def calc(g, valid_mask):
    bins=10
    momentum = 0
    edges_left = [float(x) / bins for x in range(bins)]
    edges_left = tf.constant(edges_left)  # [bins]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]
    edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1, 1]

    edges_right = [float(x) / bins for x in range(1, bins + 1)]
    edges_right[-1] += 1e-3
    edges_right = tf.constant(edges_right)  # [bins]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
    edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1, 1]
    alpha = momentum
    # valid_mask = tf.cast(valid_mask, dtype=tf.bool)
    tot = tf.maximum(tf.reduce_sum(tf.cast(valid_mask, dtype=tf.float32)), 1.0)
    inds_mask = tf.logical_and(tf.greater_equal(g, edges_left), tf.less(g, edges_right))
    zero_matrix = tf.cast(tf.zeros_like(inds_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

    inds = tf.cast(tf.logical_and(inds_mask, valid_mask), dtype=tf.float32)  # [bins, batch_num, class_num]

    num_in_bin = tf.reduce_sum(inds, axis=[1, 2, 3])  # [bins]
    valid_bins = tf.greater(num_in_bin, 0)  # [bins]

    num_valid_bin = tf.reduce_sum(tf.cast(valid_bins, dtype=tf.float32))

    if momentum > 0:
        acc_sum = [0.0 for _ in range(bins)]
        acc_sum = tf.Variable(acc_sum, trainable=False)

    if alpha > 0:
        update = tf.assign(acc_sum,
                           tf.where(valid_bins, alpha * acc_sum + (1 - alpha) * num_in_bin, acc_sum))
        with tf.control_dependencies([update]):
            acc_sum_tmp = tf.identity(acc_sum, name='updated_accsum')
            acc_sum = tf.expand_dims(acc_sum_tmp, -1)  # [bins, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
            acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1, 1]
            acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
    else:
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
        num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1, 1]
        num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
        weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
        weights = tf.reduce_sum(weights, axis=0)
    weights = weights / num_valid_bin
    return weights, tot

def dice_loss(preds, labels,smooth=1): #   DL
    inse = tf.reduce_sum(preds * labels,axis=1)
    l =  tf.reduce_sum(preds * preds,axis=1)
    r = tf.reduce_sum(labels * labels,axis=1)
    dice = (2. * inse + smooth) / (l + r + smooth)
    loss = tf.reduce_mean(1-dice)
    return loss

def xiaxin_weighted_cross_entropy(preds,labels,samples_per_cls): # 2020 7 24
    d_lossweight =samples_per_cls[0]/samples_per_cls[1]
    class_weights = tf.constant([1.0,d_lossweight])
    weights = tf.reduce_sum(class_weights * labels, axis=1)
    unweighted_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    weighted_losses = unweighted_loss * weights
    loss = tf.reduce_mean(weighted_losses)
    return loss

# RNN 模型loss
def weighted_cross_entropy(preds, labels,samples_per_cls):
    pos_weight = samples_per_cls[1]/samples_per_cls[0]
    loss = tf.nn.weighted_cross_entropy_with_logits(logits = preds,targets=labels,pos_weight=pos_weight)
    return tf.reduce_mean(loss)

def focal_loss(pred,y,samples_per_cls):
    gamma = 2
    alpha = samples_per_cls[1] / samples_per_cls[0]
    pred = tf.nn.softmax(pred)
    zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    pos_p_sub = array_ops.where(y > zeros, y - pred, zeros)  # positive sample
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(y > zeros, zeros, pred)  # negative sample
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
    return tf.reduce_mean(tf.reduce_sum(per_entry_cross_ent,axis=1))


def DSC_loss(y_pred, y_true): # https://www.cnblogs.com/hotsnow/p/10954624.html
    soomth = 0.5
    y_pred_rev = tf.subtract(1.0,y_pred)
    nominator = tf.multiply(tf.multiply(2.0,y_pred_rev),y_pred)*y_true
    denominator = tf.multiply(y_pred_rev,y_pred)+y_true
    dsc_coe = tf.subtract(1.0,tf.divide(nominator, denominator))
    return tf.reduce_mean(dsc_coe)

def sensitivity_loss(y_pred, y_true): # https://www.cnblogs.com/hotsnow/p/10954624.html
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)
