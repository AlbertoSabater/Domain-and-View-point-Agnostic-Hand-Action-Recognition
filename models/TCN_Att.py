#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:17:45 2021

@author: asabater
"""

import os
# os.chdir('..')


from tcn import TCN, tcn_full_summary
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Lambda, Conv1D, BatchNormalization, Masking # BatchNormalization
import tensorflow as tf
import ast
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from models.autoencoder_tcn import EncoderTCN


import numpy as np




class TCN_Att(Model):
    def __init__(self, num_feats, conv_params, 
                 max_seq_len,
                 lstm_dropout,
                 masking=False, 
                 prediction_mode=False,
                 tcn_batch_norm = False,
                 att_layers = [],
                 representation_layers = [],     # len=0 -> no dense layers | -1 = nb_filters | 128/256...
                 # projection_dense = [],         # len=0 -> no projection | -1 = nb_filters | 128/256...
                 **kwargs
                 ):
        super(TCN_Att, self).__init__()
        
        if len(conv_params) == 6:
            nb_filters, kernel_size, nb_stacks, use_skip_connections, padding, dilations = conv_params
            if type(dilations) == int: 
                dilations = [dilations]
            if type(dilations) == str: dilations = ast.literal_eval(dilations)
            else:
                dilations = [ [ i for i in [1, 2, 4, 8, 16, 32] if i<= d ] for d in dilations ]

            print('dilations', dilations)
        else:
            raise ValueError('conv_params length not recognized', len(conv_params))
            
        self.encoder_net = EncoderTCN(
                                   num_feats = num_feats, 
                                   nb_filters=nb_filters, 
                                   kernel_size=kernel_size,
                                   nb_stacks=nb_stacks,
                                   use_skip_connections = use_skip_connections,
                                   padding = padding,
                                   dilations = dilations,
                                   lstm_dropout=lstm_dropout,
                                   masking=masking,
                                   prediction_mode = prediction_mode,
                                   tcn_batch_norm = tcn_batch_norm)            
        

        self.norm = Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))

        if representation_layers is not None and len(representation_layers) > 0:
            self.representation_block = []
            for layer_info in representation_layers:
                if str(layer_info) == 'BatchNorm':
                    self.representation_block.append(BatchNormalization())
                elif 'conv' in str(layer_info):
                    layer_info = int(layer_info.replace('conv', ''))
                    self.representation_block.append(Conv1D(layer_info, 1, padding=padding, activation='relu'))
                else:
                    raise ValueError('layer info error: {}'.format(layer_info))
                    # self.representation_block.append(Dense(max(layer_info, abs(max_seq_len)), activation='relu' if layer_info != -1 else 'sigmoid'))
            self.representation_block = Sequential(self.representation_block, name='representation_block')
        else: self.representation_block = None
        
        l = [ l for l in self.encoder_net.layers if type(l) == TCN ][-1]
        l.return_sequences = True
        self.use_attention = True
        
        if att_layers is not None and len(att_layers) > 0:
            # Attention block. TCN returns all embeddings. By default attention mis them to return just one
       
            self.att_dense = []
            flattened = False
            for layer_info in att_layers:
                if str(layer_info) == 'BatchNorm':
                    self.att_dense.append(BatchNormalization())
                elif 'conv' in str(layer_info):
                    layer_info = int(layer_info.replace('conv', ''))
                    self.att_dense.append(Conv1D(layer_info, 1, padding=padding, activation='relu'))
                else:
                    if not flattened: self.att_dense.append(tf.keras.layers.Flatten())
                    self.att_dense.append(Dense(max(layer_info, abs(max_seq_len)), activation='relu' if layer_info != -1 else 'sigmoid'))
            
            print(self.att_dense)
            self.att_dense = Sequential(self.att_dense, name='output_weighting')
        else: 
            self.att_dense = None
            # self.use_attention = False
        
       
        
    def compute_weighted_average(self, x):
        mean_weights = self.att_dense(x)
        
        mean_weights = mean_weights / tf.linalg.norm(mean_weights, ord=1, axis=1, keepdims=True)
        
        x = x*mean_weights[:,:,None]
        x = tf.reduce_mean(x,axis=1)
        return mean_weights, x

    def call(self, x, return_att_weights=False):
        x = self.encoder_net(x)
        
        if self.representation_block is not None:
            x = self.representation_block(x)
        
        mean_weights = None
        if self.use_attention:
            if self.att_dense is not None: 
                mean_weights, x = self.compute_weighted_average(x)  
            else: 
                x = x[:,-1]
            
        x = self.norm(x)
        if return_att_weights: return [mean_weights, x]
        else: return [x]
            
    def get_embedding(self, x, return_att_weights=False):
        res = self.call(x, return_att_weights)
        if return_att_weights: return res
        else: 
            return res[0]
            # if self.use_attention: return res[0]
            # else: return res[0]
        
    def set_encoder_return_sequences(self, return_sequences):
        if return_sequences: self.use_attention = False
        else: self.use_attention = True
        

class EncoderTCN(Model):
    def __init__(self, num_feats, nb_filters, kernel_size, nb_stacks, use_skip_connections, 
                 lstm_dropout, padding, dilations,
                 masking=False, 
                 prediction_mode=False,
                 tcn_batch_norm = False,
                 **kwargs
                 ):
        super(EncoderTCN, self).__init__()
        self.encoder_layers = []
        
        # Add masking layer
        if masking: 
            print('MASKING')
            self.encoder_layers.append(Masking())
        
        num_tcn = len(dilations)
        print('num_tcn:', num_tcn)
        for i in range(num_tcn-1):
            l = TCN(
                    nb_filters = nb_filters, 
                    kernel_size = kernel_size,
                    nb_stacks = nb_stacks,
                    use_skip_connections = use_skip_connections,
                    padding = padding,
                    dilations = dilations[i],
                    dropout_rate = lstm_dropout,
                    return_sequences=True,
                    use_batch_norm = tcn_batch_norm
                    )
            self.encoder_layers.append(l)        
            print('TCN', i, dilations[i], l.receptive_field)
        
        l = TCN(
                nb_filters = nb_filters, 
                kernel_size = kernel_size,
                nb_stacks = nb_stacks,
                use_skip_connections = use_skip_connections,
                padding = padding,
                dilations = dilations[-1],
                dropout_rate = lstm_dropout,
                return_sequences=prediction_mode,
                use_batch_norm = tcn_batch_norm
                )
        self.encoder_layers.append(l)
        print('TCN', -1, dilations[-1], l.receptive_field)
        
        for l in self.encoder_layers: print(l)
                
        self.encoder = Sequential(self.encoder_layers)

    def call(self, x):
        encoder = self.encoder(x)
        return encoder
    def get_embedding(self, x):
        emb = self.encoder(x)
        return emb

        
        
if __name__ == '__main__':
    
    # %%
    
    import numpy as np
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    
    num_feats = 87
    batch_size = 28
    max_seq_len = 64
    
    # conv_params = (256, 6, 2, True, 'causal', [2])             # 24
    # conv_params = (256, 6, 2, True, 'causal', [2])             # 24
    # conv_params = (256, 8, 2, True, 'causal', [4])
    conv_params = (256, 8, 2, True, 'causal', [4])
    dropout = 0.0
    masking = True
    prediction_mode=False
    tcn_batch_norm=True
    
        
    model = TCN_Att(num_feats, conv_params, 
                 max_seq_len,
                 dropout,
                 masking, 
                 prediction_mode,
                 tcn_batch_norm,
                 # att_layers = [64, -1]
                 # att_layers = [256, -1]
                  # att_layers = [128, -1]
                    # att_layers = [-1]
                    # att_layers = ['conv64', -1]
                    att_layers = ['conv64', 'BatchNorm',  -1],
                    # att_layers = [],
                 # representation_layers = []
                 # representation_layers = ['conv256']
                  representation_layers = ['conv67']
                 )    


    model.build((None, max_seq_len, num_feats))



    mw32, pred32 = model(np.random.rand(batch_size,max_seq_len,num_feats), return_att_weights=True)
    pred32 = pred32.numpy()
    if model.att_dense is not None: mw32 = mw32.numpy()
    # print([ p.shape for p in pred32 ])
    
    # pred32 = model(np.random.rand(14,max_seq_len,num_feats))
    # print([ p.shape for p in pred32 ])
    
    
    model.summary()
    if model.att_dense is not None: model.layers[-1].summary()

    # print([ (l, l.activation for l in model.att_dense.layers ])
    # print([ l for l in model.att_dense.layers ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    model.fit(x=np.random.rand(500, max_seq_len, num_feats).astype('float32'), y=np.random.rand(500,), 
              batch_size=batch_size, epochs=1)

    
    model.set_encoder_return_sequences(True)
    mw32_rs, pred32_rs = model(np.random.rand(batch_size,max_seq_len,num_feats), return_att_weights=True)
    pred32_rs = pred32_rs.numpy()
    # if model.att_dense is not None: mw32_rs = mw32_rs.numpy()
    
    

# %%

    if False:
        # %%
        
# =============================================================================
#         Check attention_weights
# =============================================================================

        import prediction_utils
        from data_generator_hand_contrastive import DataGenerator_HandContrastive
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import numpy as np
        import os 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    

        # loss_name, path_model = 'mixknn_train5_val1', '/mnt/hdd/ml_results/core/SHREC2017_28_jn20_tcn_att_common_minimal_v0/0120_0803_model_67/'
        # loss_name, path_model = 'mixknn_best', '/mnt/hdd/ml_results/core/SHREC2017_28_jn20_tcn_att_common_minimal_v1/0122_0920_model_0/'
        # loss_name, path_model = 'mixknn_best', '/mnt/hdd/ml_results/core/SHREC2017_28_jn20_tcn_att_common_minimal_v1_256/0124_1916_model_21/'
        # loss_name, path_model = 'mixknn_best', '/mnt/hdd/ml_results/core/SHREC2017_28_jn20_tcn_att_common_minimal_v1_256/0125_1438_model_71/'
        loss_name, path_model = 'mixknn_best', '/mnt/hdd/ml_results/core/SHREC2017_28_jn20_tcn_att_common_minimal_v1_256/0126_0723_model_119/'
        

        model, model_params = prediction_utils.load_model(path_model, False, loss_name = loss_name)
        model_params['use_rotations'] = None

        data_gen = DataGenerator_HandContrastive(**model_params)

        joints_num = 20
        
        # annotations_store_folder = './dataset_scripts/F_PHAB/paper_tables_annotations'
        # # Load all annotations
        # with open(os.path.join(annotations_store_folder, 'total_annotations_jn{}.txt'.format(joints_num)), 'r') as f: 
        #     total_annotations = f.read().splitlines()
        # total_labels = np.stack([ l.split()[-1] for l in total_annotations ])
        # total_annotations = [ l.split()[0] for l in total_annotations ]
        # action_sequences = [ data_gen.get_pose_data_v2(data_gen.load_skel_coords(ann), validation=True) for ann in total_annotations ]
        # action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')
        # print('* Data sequences loaded')
        
        
        shrec_train_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn{}.txt'.format('14', joints_num)
        shrec_val_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn{}.txt'.format('14', joints_num)
        with open(os.path.join(shrec_train_anns), 'r') as f: shrec_train_anns = f.read().splitlines()
        with open(os.path.join(shrec_val_anns), 'r') as f: shrec_val_anns = f.read().splitlines()
        total_shrec_anns = shrec_train_anns + shrec_val_anns
        total_shrec_anns = np.stack([ l.split()[0] for l in total_shrec_anns ])
        action_sequences = [ data_gen.get_pose_data_v2(data_gen.load_skel_coords(ann), validation=True) for ann in total_shrec_anns ]
        action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')

        
        # embs = np.concatenate([ model.get_embedding(s, return_att_weights=False).numpy() for s in np.array_split(action_sequences, max(1, len(action_sequences)//1000)) ])
        embs = [ model.get_embedding(s, return_att_weights=True) for s in np.array_split(action_sequences, max(1, len(action_sequences)//1000)) ]

        att_weights, embs = embs[0][0].numpy(), embs[0][1].numpy()
        
        
        import matplotlib.pyplot as plt
       
        # xvalues = range(1,33)
        xvalues = range(0,32)
        plt.figure(figsize=(8,4), dpi=250)
        for aw in att_weights:
            plt.plot(xvalues, aw, alpha=0.2, linestyle='-')
            pass
        
        plt.plot(xvalues, att_weights.mean(axis=0), alpha=1, linewidth=4, c='r')
        # plt.plot(xvalues, np.median(att_weights, axis=0), alpha=1, linewidth=4, c='r')
        plt.xlabel('Time-steps', fontsize=16)
        plt.ylabel('Descriptor relevance', fontsize=16)
        # plt.xticks(np.arange(0, 33, 8))


        # %%
        
# =============================================================================
#         Check attention_weights from CLF model
# =============================================================================

        import prediction_utils
        from data_generator_hand_contrastive import DataGenerator_HandContrastive
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        import numpy as np
        import os 
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    

        # loss_name, path_model = 'val_loss', '/mnt/hdd/ml_results/core/base_tcb_clf/0126_0723_model_119_FPHAB/v0/1:1_pretrainFalse_numrep0/0210_0115_model_0/'
        loss_name, path_model = 'val_loss', '/mnt/hdd/ml_results/core/base_tcb_clf/0126_0723_model_119_FPHAB/bench_v0/1:1_pretrainFalse_numrep0/0214_1816_model_0/'

        model, model_params = prediction_utils.load_model(path_model, False, loss_name = loss_name)
        model_params['use_rotations'] = None
        model_params = model_params['backbone_params']

        data_gen = DataGenerator_HandContrastive(**model_params)

        joints_num = 20
        
        
        
        fphab_train_anns = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/1:1_split0_annotations_train.txt'
        fphab_val_anns = './dataset_scripts/F_PHAB/paper_tables_annotations/label_perc_splits/1:1_split0_annotations_val.txt'

        with open(os.path.join(fphab_train_anns), 'r') as f: fphab_train_anns = f.read().splitlines()
        with open(os.path.join(fphab_val_anns), 'r') as f: fphab_val_anns = f.read().splitlines()
        total_shrec_anns = fphab_train_anns + fphab_val_anns
        total_shrec_anns = np.stack([ l.split()[0] for l in total_shrec_anns ])
        action_sequences = [ data_gen.get_pose_data_v2(data_gen.load_skel_coords(ann), validation=True) for ann in total_shrec_anns ]
        action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')

        
        # embs = np.concatenate([ model.get_embedding(s, return_att_weights=False).numpy() for s in np.array_split(action_sequences, max(1, len(action_sequences)//1000)) ])
        embs = [ model(s, return_att_weights=True) for s in np.array_split(action_sequences, max(1, len(action_sequences)//1000)) ]

        att_weights, embs = embs[0][0].numpy(), embs[0][1].numpy()
        
        
        import matplotlib.pyplot as plt
       
        plt.figure(figsize=(8,5), dpi=250)
        for aw in att_weights:
            plt.plot(aw, alpha=0.4, linestyle='-')
            pass
        
        plt.plot(att_weights.mean(axis=0), alpha=1, linewidth=4, c='r')
        

