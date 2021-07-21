#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:46:16 2020

@author: asabater
"""

import json
import os
import pickle
# from data_generator import load_scaler


def get_weights_filename(path_model, loss_name, verbose=False, num_file=None):
    weights = sorted([ w for w in os.listdir(path_model + '/weights') if 'index' in w ])
    if verbose: print(weights)
    if loss_name is not None:
        weights = [ w for w in weights  if loss_name in w ][0]
    else:
        if num_file is not None:
            weights = weights[num_file]
        # weights = weights[0]
        elif 'mon' in weights[0]: 		# and False
            if verbose:  print('weights by monitor')
            weights = max(weights, key=lambda w: [ float(s[3:]) for s in w.replace('.ckpt.index', '').split('-') if s.startswith('mon') ][0])
        elif 'val_loss' not in weights[0]:
            if verbose: print('weights by last')
            weights = weights[-1]
        else:
            if verbose: print('weights by val_loss')
            losses = [ float(w.split('-')[2][8:15]) for w in weights ]
            weights = weights[losses.index(min(losses))]
    weights = weights[:-6]
    return path_model + '/weights/' + weights


def load_model(path_model, return_sequences=True, num_file=None, loss_name=None):
    model_params = json.load(open(path_model + '/model_params.json'))
    weights = get_weights_filename(path_model, loss_name, verbose=True)
    print(weights)
    
    if model_params.get('use_gru',False) == True and 'decoder_v2' not in path_model:
        model_params['use_gru'] = False
    
    if model_params['model_name'] == 'classifier_lstm':
        from models.classifier_lstm import ClassifierLSTM
        model = ClassifierLSTM(prediction_mode=return_sequences, **model_params)
    elif 'vae_lstm' in model_params['model_name']:
        from models.variational_autoencoder_lstm import VariationalAutoEncoderLSTM
        model = VariationalAutoEncoderLSTM(prediction_mode=return_sequences, **model_params)
    elif 'ae_lstm' in model_params['model_name']:
        from models.autoencoder_lstm import AutoEncoderLSTM
        model = AutoEncoderLSTM(prediction_mode=return_sequences, **model_params)
    elif 'ae_tcn' in model_params['model_name']:
        from models.autoencoder_tcn import AutoEncoderTCN
        model = AutoEncoderTCN(prediction_mode=return_sequences, **model_params)
    elif "tcn_contr" in model_params['model_name']:
        # from models.__OLD_autoencoder_tcn import TCN_Contrastive
        # print('Creating TCN Contrastive')
        # model = TCN_Contrastive(**model_params)
        from models.TCN_Att import TCN_Att
        print('Creating TCN Contrastive')
        model = TCN_Att(**model_params)
    elif "tcn_att" in model_params['model_name']:
        from models.TCN_Att import TCN_Att
        print('Creating TCN Contrastive')
        model = TCN_Att(**model_params)
    elif "base_tcb_clf" in model_params['model_name']:
        from models.base_model import Base_CLF_Model
        model = Base_CLF_Model(model_params['backbone_model_name'], model_params['backbone_params'], 
                               model_params['clf_layers'], model_params['out_dim'], 
                               backbone_weights=model_params['backbone_weights'])
        # model_params = {**model_params, **model_params['backbone_params']}
    else:
        raise ValueError('model_name not handled:', model_params['model_name'])
    
    print(' ** Model created')
    model.load_weights(weights).expect_partial()
    print(' ** Weights loaded:', weights)
    
    # model.build((None, None, model_params['num_feats']))
    # print(' ** Model bu ilt')
    
    
    scale_data = model_params.get('scale_data', False) or model_params.get('use_scaler', False)
    if scale_data: 
        print(' * Loading data scaler')
        model_params['scaler'] = pickle.load(open(path_model + 'scaler.pckl', 'rb'))
    else: model_params['scaler'] = None
    
    
    for data_key in ['use_jcd_features', 'use_speeds', 'use_coords_raw', 
                     'use_coords', 'use_jcd_diff', 'use_bone_angles',
                     'tcn_batch_norm', 'use_bone_angles_cent']:
        if data_key not in model_params: model_params[data_key] = False
        
    if 'average_wrong_skels' not in model_params: model_params['average_wrong_skels'] = True
    
    return model, model_params