#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:01:46 2020

@author: asabater
"""

import os
import json
import tensorflow as tf
import numpy as np
import time
from joblib import Parallel, delayed

import pickle
from tqdm import tqdm
from scipy.spatial import distance

import sys
sys.path.append('../..')

import prediction_utils

# from data_generator_obj import DataGenerator_Hand
from data_generator import DataGenerator
# from data_generator_hand_contrastive import DataGenerator_HandContrastive
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


knn_neighbors = [1,3,5,7,9,11]


# def get_knn_classification(model, samples_dict, labels_dict, weights='uniform'):
def get_knn_classification(model, samples_dict, labels_dict, weights='distance'):
    
    # t = time.time()
    # embs_dict = { k:model.get_embedding(samples).numpy() for k,samples in samples_dict.items() }
    # Get embeddings from chunks of data
    # embs_dict = { k:np.concatenate([ model.get_embedding(s).numpy() for s in np.split(samples, max(1, len(samples)//1000)) ]) for k,samples in samples_dict.items() }
    embs_dict = { k:np.concatenate([ model.get_embedding(s).numpy() for s in np.array_split(samples, max(1, len(samples)//1000)) ]) for k,samples in samples_dict.items() }
    # print('** Preds_time:', time.time()-t)
    
    # print(embs_dict['train'].shape, embs_dict['val'].shape)
    
    # t = time.time()
    res = {}
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=8, weights=weights).fit(embs_dict['train'], labels_dict['train'])
    for n in knn_neighbors:
        knn = knn.set_params(**{'n_neighbors': n})
        preds = knn.predict(embs_dict['val'])
        acc = accuracy_score(labels_dict['val'], preds)
        res[n] = acc
        res['y_pred_{}'.format(n)] = preds
        res['y_true_{}'.format(n)] = labels_dict['val']
    # print('* KNN_time:', time.time()-t)
        
    return res


def get_knn_classification_v2(model, samples_dict, labels_dict):
    
    # t = time.time()
    embs_dict = { k:model.get_embedding(samples).numpy() for k,samples in samples_dict.items() }
    # print('** Preds_time:', time.time()-t)
    
    # t = time.time()
    res = {}
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=8, weights='distance').fit(embs_dict['train'], labels_dict['train'])
    for n in knn_neighbors:
        knn = knn.set_params(**{'n_neighbors': n})
        preds = knn.predict(embs_dict['val'])
        acc = accuracy_score(labels_dict['val'], preds)
        res[n] = acc
        res['y_pred_{}'.format(n)] = preds
        res['y_true_{}'.format(n)] = labels_dict['val']
    # print('* KNN_time:', time.time()-t)
        
    return res


def knn_mix_callback(mix_log_keys, writer):
    def eval_mix(epoch, logs):
        with writer.as_default():
            for new_key, (k1, k2) in mix_log_keys.items():    
                if new_key == 'mixknn_best':
                    # print(logs)
                    # print(logs.keys())
                    v1 = max([ v for k,v in logs.items() if k1 in k ])
                    v2 = max([ v for k,v in logs.items() if k2 in k ])
                else:
                    v1, v2 = logs[k1], logs[k2]
                new_value = v1 * v2
                logs[new_key] = new_value
                tf.summary.scalar(new_key, data=new_value, step=epoch)
        writer.flush()  
    return eval_mix


def knn_callback(model, model_params, dataset_name, knn_files, writer):
    
    np.random.seed(0)
    
    # # data_gen = DataGenerator_Hand(**model_params)
    # if 'use_rotations' in model_params: 
    #     print(' ** USING DataGenerator')
    #     data_gen = DataGenerator(**model_params)
    # else: 
    #     print(' ** USING DataGenerator')
    data_gen = DataGenerator(**model_params)
        
    annotations = { k:open(anns_file, 'r').read().splitlines() for k, anns_file in knn_files.items() }
    anns_files = { k:[ l.split()[0] for l in anns ] for k, anns in annotations.items() }
    labels_dict = { k:[ l.split()[1] for l in anns ] for k, anns in annotations.items() }
    
    samples_dict = { k:[ data_gen.get_pose_data_v2(data_gen.load_skel_coords(l), validation=True) for l in anns ] for k, anns in anns_files.items() }
    samples_dict = { k:pad_sequences(samples, abs(model_params['max_seq_len']), dtype='float32', padding='pre') for k,samples in samples_dict.items() }
        
    def eval_knn(epoch, logs):
        knn_res = get_knn_classification(model, samples_dict, labels_dict)

        with writer.as_default():
            for n in knn_neighbors:
                acc = knn_res[n]
                k = 'knn_{}_{}'.format(dataset_name, n).replace('-', '_')
                logs.update(**{k: acc})
                tf.summary.scalar(k, data=acc, step=epoch)
            writer.flush()  

    return eval_knn



