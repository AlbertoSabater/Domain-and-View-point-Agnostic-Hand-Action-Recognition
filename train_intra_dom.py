#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:35:50 2021

@author: asabater
"""

import os
import json

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import train_utils
from train_cross_dom import get_lr_metric
from data_generator import DataGenerator
from models.base_model import Base_CLF_Model
import numpy as np
import pickle


# %%

def main(model_params, override_backbone_params):

    # Initialize training
    model_params['path_model'] = train_utils.create_model_folder(model_params['path_results'], model_params['model_name'])
    backbone_params = model_params['backbone_params']

    # Create model
    clf_model = Base_CLF_Model(model_params['backbone_model_name'], backbone_params, 
                               model_params['clf_layers'], model_params['out_dim'], 
                               backbone_weights=model_params['backbone_weights'])


    clf_model.build((None, abs(backbone_params['max_seq_len']), backbone_params['num_feats']))

    # Initialize inputs and outputs
    dummy_inpt = np.random.rand(backbone_params['batch_size'], abs(backbone_params['max_seq_len']), backbone_params['num_feats'])
    print(' * dummy_shape:', dummy_inpt.shape)
    dummy_pred = clf_model(dummy_inpt);
    print(' * dummy_pred shape', [ p.shape for p in dummy_pred ])
    
    clf_model.save(model_params['path_model'] + 'model')

    
    callbacks = [ TensorBoard(log_dir = model_params['path_model'], profile_batch=0) ]
    callbacks += [ ModelCheckpoint(model_params['path_model'] + 'weights/' + \
                                    'ep{epoch:03d}-loss{loss:.5f}-' + mon + '{' + mon + ':.5f}.ckpt',
                                              monitor=mon, save_weights_only=True, 
                                              save_best_only=True, save_freq='epoch', mode=mon_mode) \
                      for mon, mon_mode in model_params['mon_ckpt'] ]
    callbacks += [ 
                    ReduceLROnPlateau(monitor=model_params['monitor'], min_delta=model_params['lr_min_delta'], 
                                      factor=model_params['factor'], patience=model_params['patience'], 
                                      verbose=1, min_lr=1e-7),
                ]
    
    print(callbacks)
    

    json.dump(model_params, open(model_params['path_model']+'model_params.json', 'w'))

    model_params['backbone_params'].update(**override_backbone_params)


    with open(model_params['train_annotations'], 'r') as f: num_train_files = len(f.read().splitlines())
    if model_params['val_annotations']  == '': num_val_files = 0
    else:
        with open(model_params['val_annotations'], 'r') as f: num_val_files = len(f.read().splitlines())
    print(' ** num_train_files: {} | num_val_files: {}'.format(num_train_files, num_val_files), 
           num_train_files//(model_params['backbone_params']['batch_size']//model_params['backbone_params']['K']),
           num_val_files//(model_params['backbone_params']['batch_size']//model_params['backbone_params']['K']))



    data_gen = DataGenerator(**backbone_params)



    init_epoch = 0
    train_resumes = []
    for fit_params in model_params['train_params']:
        clf_model.backbone.trainable = fit_params['backbone_trainable']
        print(clf_model.summary())
        
        optimizer = Adam(fit_params['init_lr'])
        clf_model.compile(optimizer=optimizer, 
                      loss = [tf.keras.losses.CategoricalCrossentropy()], 
                      metrics = ['accuracy', get_lr_metric(optimizer)])
        
        train_gen = data_gen.triplet_data_generator(model_params['train_annotations'], validation=False,
                           in_memory_generator=model_params['backbone_params']['in_memory_generator_train'], **model_params['backbone_params'])        
        val_gen = data_gen.triplet_data_generator(model_params['val_annotations'], validation=True,
                           in_memory_generator=model_params['backbone_params']['in_memory_generator_val'], **model_params['backbone_params'])        
        
        print(train_gen, val_gen)
        print('*'*60)
        print('*** TRAINING ***')
        print('*'*60)        
    
        hist = clf_model.fit(
                train_gen,
                validation_data = val_gen,
                steps_per_epoch = num_train_files//(model_params['backbone_params']['batch_size']//model_params['backbone_params']['K']),
                validation_steps = None if num_val_files == 0 else num_val_files//(model_params['backbone_params']['batch_size']//model_params['backbone_params']['K']),
                initial_epoch = init_epoch, 
                epochs = init_epoch + fit_params['num_epochs'],
                verbose = 2,
                callbacks = callbacks,
            )   
        train_resumes.append(hist)
        
        init_epoch += fit_params['num_epochs']
        
    
    train_resumes = [ tr.history for tr in train_resumes ]
    pickle.dump(train_resumes, open(model_params['path_model'] + 'train_resumes.pckl', 'wb'))
    
    
    fig, ax1 = plt.subplots(figsize=(8,4), dpi=200)
    ax2 = ax1.twinx()
    
    ax1.plot(sum([ hist['loss'] for hist in train_resumes ], []), 'b', label='Train loss')
    ax1.plot(sum([ hist['val_loss'] for hist in train_resumes ], []), 'g', label='Val loss')
    ax1.set_ylabel('Loss')
    ax2.plot(sum([ hist['lr'] for hist in train_resumes ], []), 'y', label='LR')
    
    ax1.set_xlabel('Epoch')
    ax2.set_ylabel('LR')
    ax2.set_yscale('log')
    plt.title(model_params['path_model'].split('/')[-3] + ' | {} | {}\n'.format(num_train_files, num_val_files) +  \
              'Model loss | val_loss: {:.3f} | val_acc: {:.3f}'.format(
               max(sum([ hist['val_loss'] for hist in train_resumes ], [])),
               max(sum([ hist['val_accuracy'] for hist in train_resumes ], []))))
    fig.legend()
    plt.show()    


    for mon, mon_mode in model_params['mon_ckpt']:
        train_utils.remove_path_weights(model_params['path_model'], mon, mon_mode=='min')
        
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    clf_model = None
    del clf_model
    
    
    
# %%

if __name__ == '__main__':
# %%    
    import socket
    import copy
    import prediction_utils
    
    
    """
    v0 -> no squeezing by max_len_len
    v1 -> no squeezing by max_len_len
    v2 -> squeezing by max_len_len
    v3 -> squeezing by max_len_len
    """
    
    
    
    
    pretrain_model = False
    
    # Initialize training and model params
    model_params = {}
    model_params['path_results'] = "./pretrained_models/"
    model_params['model_name'] = 'train_intradom_SHREC'
    model_params['mon_ckpt'] = [('val_loss', 'min'), ('val_accuracy', 'max')]
    model_params['monitor'] = 'val_loss'


    # ============= MODEL SELECTION ==============
    model_params['backbone_model_name'] = 'tcn_att'
    model_params['backbone_path'] = './pretrained_models/xdom_summarization'
    backbone_params = json.load(open(model_params['backbone_path'] + '/model_params.json'))
    model_params['backbone_params'] = backbone_params
    # ============================================


# =============================================================================
#     Choose the training dataset by commenting out the proper lines
# =============================================================================

    # SHREC training data
    # model_params['num_classes'] = 14; model_params['model_name'] = 'train_intradoc_SHREC_14'
    # model_params['num_classes'] = 28; model_params['model_name'] = 'train_intradoc_SHREC_28'
    # model_params['train_annotations'] = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn20.txt'.format(model_params['num_classes'])
    # model_params['val_annotations'] = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn20.txt'.format(model_params['num_classes'])

    # F-PHAB 1:1 training data
    # model_params['num_classes'] = 45; model_params['model_name'] = 'train_intradoc_FPHAB_1:1'
    # model_params['train_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/1:1_split0_annotations_train.txt'
    # model_params['val_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/1:1_split0_annotations_val.txt'
        
    # F-PHAB 1:3 splits
    # model_params['num_classes'] = 45; split_num = 0; model_params['model_name'] = 'train_intradoc_FPHAB_1:3'
    # model_params['train_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/1:3_split{}_annotations_train.txt'.format(split_num)
    # model_params['val_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/1:3_split{}_annotations_val.txt'.format(split_num)
   
    # F-PHAB 3:3 splits
    # model_params['num_classes'] = 45; split_num = 0; model_params['model_name'] = 'train_intradoc_FPHAB_3:1'
    # model_params['train_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/3:1_split{}_annotations_train.txt'.format(split_num)
    # model_params['val_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/3:1_split{}_annotations_val.txt'.format(split_num)
         
    # F-PHAB cross-subjects splits
    # model_params['num_classes'] = 45; split_num = 0; model_params['model_name'] = 'train_intradoc_FPHAB_xsubj'
    # model_params['train_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/cross_subj_split{}_annotations_train.txt'.format(split_num)
    # model_params['val_annotations'] = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/cross_subj_split{}_annotations_val.txt'.format(split_num)
    
    if 'train_annotations' not in model_params: raise ValueError('Select the data splits for training and evaluation')
    

    model_params['clf_layers'] = [-1]
    model_params['out_dim'] = model_params['num_classes']
    model_params['backbone_params']['num_classes'] = model_params['num_classes']
    
    
    # Define model base
    # Specific training params
    override_backbone_params = {'triplet': False, 'classification': True, 'decoder': False,
                                'in_memory_generator_train': False,
                                'in_memory_generator_val': True,
                                'in_memory_skels': True,
                                'batch_size': model_params['backbone_params']['num_classes'] * model_params['backbone_params']['K'],
                                # 'max_seq_len': abs(model_params['backbone_params']['max_seq_len'])  # TODO: .
                                }
    override_backbone_params['use_rotations'] = None
    
    
    
    train_params = [
            {'init_lr': 0.01, 'num_epochs': 600, 'backbone_trainable': None, 'skip_if_nopretrain': False},
            {'init_lr': 0.01, 'num_epochs': 600, 'backbone_trainable': True, 'skip_if_nopretrain': True},
        ]
    model_params['lr_min_delta'] = 0.0
    model_params['factor'] = 0.1
    model_params['patience'] = 30


    if pretrain_model:
        model_params['backbone_weights'] = prediction_utils.get_weights_filename(model_params['backbone_path'], 'mixknn_best')
    else:
        model_params['backbone_weights'] = None



    # for pretrain_model in [True, False]:
    for pretrain_model in [False]:
        model_params['train_params'] = copy.deepcopy(train_params)
        if not pretrain_model:
            model_params['train_params'] = [ tp for tp in model_params['train_params'] if not tp['skip_if_nopretrain'] ]
        for tp in model_params['train_params']: 
            if tp['backbone_trainable'] is None: 
                tp['backbone_trainable'] = not pretrain_model
        print(model_params['train_params'])
    
        # Train
        main(model_params.copy(), override_backbone_params.copy())






