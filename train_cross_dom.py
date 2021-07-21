#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 11:51:26 2021

@author: asabater
"""

"""
No EarlyStopping
Less LR drop
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
import json

from data_generator import DataGenerator
import train_utils
import argparse

from models.TCN_Att import TCN_Att


from losses import supervised_nt_xent_loss

from dataset_scripts.common_pose.n_shot_callback import knn_callback, knn_mix_callback



np.random.seed(123)
tf.random.set_seed(123)


def get_lr_metric(optimizer):
	def lr(y_true, y_pred, sample_weights=None):
		return optimizer.lr
	return lr


def main(model_params):
    train_verbose = 2

    model_params.update({
                'path_model': train_utils.create_model_folder(model_params['path_results'], model_params['model_name']),
            })
    
    
    data_gen = DataGenerator(**model_params)
        
    
    model_params['num_feats'] = data_gen.num_feats
    json.dump(model_params, open(model_params['path_model']+'model_params.json', 'w'))
    print(model_params)
    
    with open(model_params['train_annotations'], 'r') as f: num_train_files = len(f.read().splitlines())
    if model_params['val_annotations']  == '': num_val_files = 0
    else:
        with open(model_params['val_annotations'], 'r') as f: num_val_files = len(f.read().splitlines())
    print(num_train_files, num_val_files)
    
    
    
    model_name = model_params['model_name']
    print('Creating TCN_Att')
    model = TCN_Att(**model_params)
    
    
    if model_params.get('use_pretrained_weights', False):
        model.load_weights(model_params['weigths']).expect_partial()
        print(' ** Weights loaded:', model_params['weigths'])
    
    
    model.build((None, abs(model_params['max_seq_len']), model_params['num_feats']))

    
    dummy_inpt = np.random.rand(model_params['batch_size'], abs(model_params['max_seq_len']), model_params['num_feats']).astype('float32')
    print(' * dummy_shape:', dummy_inpt.shape)
    dummy_pred = model(dummy_inpt);
    print(' * dummy_pred shape', [ p.shape for p in dummy_pred ])
    dummy_emb = model.get_embedding(dummy_inpt);
    print(' * dummy_emb shape', dummy_emb.shape)
    
    
    optimizer = Adam(model_params['init_lr'], clipnorm=1.)
    losses, metrics, loss_weights, sample_weights_mode = {}, {}, {}, {}
    losses['output_1'] = supervised_nt_xent_loss(temperature=model_params['nt_xent_temp'])
    loss_weights['output_1'] = 1.0
    get_lr_metric(optimizer)
    

    print(' * losses:', losses)
    print(' * loss_weights:', loss_weights)
    if sample_weights_mode == {}: sample_weights_mode = None
    print(' * sample_weights_mode:', sample_weights_mode)

    model.summary(100)

    
    monitor = model_params.get('monitor', 'val_loss')
    monitor_mode = 'min' if model_params['min_monitor'] else 'max'
    print(' * Monitor:', monitor)
    callbacks = [ TensorBoard(log_dir = model_params['path_model'], profile_batch=0) ]
    callbacks += [ ModelCheckpoint(model_params['path_model'] + 'weights/' + \
                                    'ep{epoch:03d}-loss{loss:.5f}-' + mon + '{' + mon + ':.5f}.ckpt',
                                              monitor=mon, save_weights_only=True, 
                                              save_best_only=True, save_freq='epoch', mode=mon_mode) \
                      for mon, mon_mode in model_params['mon_ckpt'] ]
    if 'mix_log_keys' in model_params:
        callbacks += [ ModelCheckpoint(model_params['path_model'] + 'weights/' + \
                                    'ep{epoch:03d}-loss{loss:.5f}-' + mon + '{' + mon + ':.5f}.ckpt',
                                              monitor=mon, save_weights_only=True, 
                                              save_best_only=True, save_freq='epoch', mode='max') \
                      for mon in model_params['mix_log_keys'].keys() ]
    callbacks += [ 
                    ReduceLROnPlateau(monitor=monitor, mode=monitor_mode, min_delta=0.001, factor=0.5, patience=5, cooldown=3, verbose=1, min_lr=1e-7),
                ]
    print(callbacks)


    file_writer = tf.summary.create_file_writer(model_params['path_model'] + "/metrics")
    file_writer.set_as_default()
        

        
    if model_params['eval_knn']:
        new_callbacks = []
        for dataset_name, knn_files in model_params['knn_files'].items():
            new_callbacks.append(LambdaCallback(_supports_tf_logs = True, 
                                    on_epoch_end=knn_callback(model, model_params.copy(), dataset_name, knn_files.copy(), file_writer)))
            
        if 'mix_log_keys' in model_params:
            new_callbacks.append(LambdaCallback(_supports_tf_logs = True, on_epoch_end=knn_mix_callback(model_params['mix_log_keys'], file_writer)))
            
        callbacks = new_callbacks + callbacks
        

    print(callbacks)
    
    
    print(' * metrics:', metrics)
    print(' * sample_weights_mode:', sample_weights_mode)
    

    # Save model
    # model.save(model_params['path_model'] + 'model')
 
    
    
    for epoch_ind in range(len(model_params['warm_up']['epochs'])):
        optimizer.learning_rate = model_params['init_lr']
        model.compile(optimizer=optimizer, loss = losses, metrics = metrics, 
                      loss_weights = loss_weights, sample_weight_mode=sample_weights_mode)
        
        train_gen = data_gen.triplet_data_generator(model_params['train_annotations'], validation=False,
                           in_memory_generator=model_params['in_memory_generator_train'], **model_params)        
        val_gen = data_gen.triplet_data_generator(model_params['val_annotations'], validation=True,
                           in_memory_generator=model_params['in_memory_generator_val'], **model_params)   
        
        print(train_gen, val_gen)
        print('*'*60)
        print('*** TRAINING ***')
        print('*'*60)        
    
        model.fit(
                train_gen,
                validation_data = val_gen,
                steps_per_epoch = num_train_files//model_params['batch_size'],
                validation_steps = None if num_val_files == 0 else num_val_files//model_params['batch_size'],
                initial_epoch = sum(model_params['warm_up']['epochs'][:epoch_ind]), 
                epochs = sum(model_params['warm_up']['epochs'][:epoch_ind+1]),
                verbose = train_verbose,
                callbacks = callbacks,
                # workers=8,
                # use_multiprocessing=True,
            )    
        
        
        


# =============================================================================
#     Training finished
# =============================================================================


    del train_gen; del val_gen; del data_gen
    del callbacks; 
    file_writer.close(); del file_writer

    print(' * Training finished')

    model.summary(100)
    
    
    for mon, mon_mode in model_params['mon_ckpt']:
        train_utils.remove_path_weights(model_params['path_model'], mon, mon_mode=='min')

    if 'mix_log_keys' in model_params:
        for mon in model_params['mix_log_keys'].keys():
            train_utils.remove_path_weights(model_params['path_model'], mon, False)
            
        
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    model = None
    del model
    
    print(' * Suboptimal weights removed')
    
    



if __name__ == "__main__":

    model_params = {
            
            # Data details
            "path_results": "./pretrained_models/",
            # "path_results": "./pm/",
            "train_annotations": "./dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_28_jn20.txt",
            "val_annotations": "./dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_28_jn20.txt",
            "model_name": "train_tcn_att_fullcoords",
            # "model_name": "m",
            
            "joints_format": "common_minimal",
            "num_classes": 28,
            "dataset": "",              #SHREC2017_28_jn20
            "eval_knn": [True],
            "knn_files": {"shrec_28": {"train": "./dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_28_jn20.txt", 
                                       "val": "./dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_28_jn20.txt"}, 
                          "F-PHAB": {"train": "./dataset_scripts/common_pose/annotations/F_PHAB/annotations_train_jn20.txt", 
                                     "val": "./dataset_scripts/common_pose/annotations/F_PHAB/annotations_val_jn20.txt"}},
            "mix_log_keys": {"mixknn_train1_val1": ["knn_shrec_28_1", "knn_F_PHAB_1"],
                             "mixknn_train5_val1": ["knn_shrec_28_5", "knn_F_PHAB_1"],
                             "mixknn_best": ["knn_shrec_28", "knn_F_PHAB"]},
            "in_memory_generator_train": False,
            "in_memory_generator_val": True,
            'in_memory_skels': True,
            
            # Metris to report after each epoch and store weights
            # "min_monitor": True, "monitor": "val_loss",
            "min_monitor": False, "monitor": "mixknn_best",
            "mon_ckpt": [["val_loss", "min"], ["knn_F_PHAB_1", "max"], ["knn_shrec_28_5", "max"], ["mixknn_best", "max"]],
            
            # Kind of skeleton features to use
            "use_relative_coordinates": True,
            "use_jcd_features": False,
            "use_bone_angles": False,
            "use_coord_diff": False,
            "use_bone_angles_diff": False,

            "scale_by_torso": True,

            
            # Model hyperparameters
            "max_seq_len": -32,
            "conv_params": [256, 4, 2, True, "causal", [4]],
            "att_layers": ["conv64", -1],
            
            
            # Training details
            "triplet": True, "classification": False, "decoder": False, "reverse_decoder": None,
            "init_lr": 0.001,
            "use_rotations": "by_batch",
            "K": 2,
            "sample_repetitions": 4,
            "triplet_loss": "nt_xent",
            "nt_xent_temp": 0.07,
            "masking": True,
            "lstm_dropout": 0.0,
            # "is_tcn": None,
            "tcn_batch_norm": True,
            # "h_flip": False,
            "skip_frames": [3],
            "temporal_scale": [0.6, 1.4],
            "rotation_noise": 10,
            "average_wrong_skels": True,
            "noise": ["uniform", 0.04],
            "warm_up": {"epochs": [100,100]},
            # "warm_up": {"epochs": [1,2]},
            }
        
    model_params['batch_size'] = model_params['num_classes'] * model_params['K']
    
    
    main(model_params)


    