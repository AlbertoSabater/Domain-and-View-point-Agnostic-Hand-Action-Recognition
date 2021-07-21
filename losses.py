#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:50:33 2020

@author: asabater
"""

import tensorflow as tf


# https://github.com/wangz10/contrastive_loss/blob/master/losses.py
def supervised_nt_xent_loss(temperature=0.07, base_temperature=0.07):
# def supervised_nt_xent_loss(y, z, temperature=0.5, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss. 
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from: 
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    
    def loss(y,z):
        y = y[:,0]
    
        batch_size = tf.shape(z)[0]
        contrast_count = 1
        anchor_count = contrast_count
        y = tf.expand_dims(y, -1)
    
        # mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        #     has the same class as sample i. Can be asymmetric.
        mask = tf.cast(tf.equal(y, tf.transpose(y)), tf.float32)
        anchor_dot_contrast = tf.divide(
            tf.matmul(z, tf.transpose(z)),
            temperature
        )
        # # for numerical stability
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max
        # # tile mask
        logits_mask = tf.ones_like(mask) - tf.eye(batch_size)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - \
            tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))
    
        # compute mean of log-likelihood over positive
        # this may introduce NaNs due to zero division,
        # when a class only has one example in the batch
        mask_sum = tf.reduce_sum(mask, axis=1)
        mean_log_prob_pos = tf.reduce_sum(
            mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    
        # loss
        loss = -(temperature / base_temperature) * mean_log_prob_pos
        # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
        loss = tf.reduce_mean(loss)
        return loss
    
    return loss







