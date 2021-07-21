#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:34:25 2021

@author: asabater
"""

import sys
sys.path.append('..')

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization




class Base_CLF_Model(Model):
    def __init__(self, backbone_model_name, backbone_params, clf_layers, out_dim, backbone_weights=None):
        super(Base_CLF_Model, self).__init__()
        
        print(' **', backbone_model_name, '**')
        if "tcn_contr" in backbone_model_name:
            print('Creating TCN Contrastive')
            from models.autoencoder_tcn import TCN_Contrastive
            self.backbone = TCN_Contrastive(**backbone_params)
        # elif "resnet_contr" in backbone_model_name:
        #     print('Creating ResNet Contrastive')
        #     backbone = ResNetEncoder(**backbone_params)
        elif "tcn_att" in backbone_model_name:
            print('Creating TCN_Att')
            from models.TCN_Att import TCN_Att
            self.backbone = TCN_Att(**backbone_params)
        else:
            raise ValueError("model name {} not recognized".format(backbone_model_name))
            
        self.backbone.set_encoder_return_sequences(False)
            
        if backbone_weights is not None:
            print(' ** Loading backcbone weights:', backbone_weights)
            self.backbone.load_weights(backbone_weights).expect_partial()
        
        
        self.backbone.build((None, abs(backbone_params['max_seq_len']), backbone_params['num_feats']))

        
        # self.clf_layers = [ Dense(layer_size if layer_size != -1 else out_dim, 
        #                           activation='relu' if layer_size != -1 else 'softmax') \
        #                               for layer_size in clf_layers ]
        self.clf_layers = []
        for layer_info in clf_layers:
            if str(layer_info) == 'BatchNorm':
                self.clf_layers.append(BatchNormalization())
            else:
                self.clf_layers.append(Dense(layer_info if layer_info != -1 else out_dim, 
                                  activation='relu' if layer_info != -1 else 'softmax'))
            
        self.clf_layers = Sequential(self.clf_layers, name='output_clf')
        # self.clf_layers.build((None, abs(backbone_params['max_seq_len']), backbone_params['conv_params'][0]))
        self.clf_layers.build((None, backbone_params['conv_params'][0]))


    def call(self, x, return_att_weights=False):
        att_weights, x = self.backbone.get_embedding(x, return_att_weights=True)
        x = self.clf_layers(x)
        if not return_att_weights: return [x]
        else: return [att_weights, x]





if __name__ == '__main__':
    
    import json
    import numpy as np
    import os
    import prediction_utils
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    
    # loss_name, backbone_model_name, path_model = 'mixknn_best', 'tcn_att', '/mnt/hdd/ml_results/core/SHREC2017_28_jn20_tcn_att_common_minimal_v1_256/0124_1916_model_21/'
    loss_name, backbone_model_name, path_model = 'mixknn_best', 'tcn_att', '/mnt/hdd/ml_results/core/SHREC2017_28_jn20_tcn_att_common_minimal_v1_256/0125_1438_model_71/'

    # clf_model, model_params = prediction_utils.load_model(path_model, False, loss_name = loss_name)

    backbone_params = json.load(open(path_model + '/model_params.json'))
    # backbone_weights = None
    backbone_weights = prediction_utils.get_weights_filename(path_model, loss_name)

    out_dim = backbone_params['num_classes']

    clf_layers = [256, 'BatchNorm', -1]

    clf_model = Base_CLF_Model(backbone_model_name, backbone_params, clf_layers, out_dim, backbone_weights=backbone_weights)


    # clf_model.build((None, abs(backbone_params['max_seq_len']), out_dim))
    # pred = clf_model(np.random.rand(1, abs(backbone_params['max_seq_len']), backbone_params['num_feats']))
    pred = clf_model.predict(np.random.rand(1, abs(backbone_params['max_seq_len']), backbone_params['num_feats']))

    clf_model.summary()
    clf_model.clf_layers.summary()






