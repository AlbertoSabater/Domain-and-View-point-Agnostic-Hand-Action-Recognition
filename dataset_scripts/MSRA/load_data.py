#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:37:22 2021

@author: asabater
"""

import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


path_dataset = './datasets/MSRA/cvpr15_MSRAHandGestureDB'
subjects = [ 'P{}'.format(i) for i in range(9)]  
actions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']
joints_inds = { j:i for i,j in enumerate(['wrist', 
                                          'index_mcp', 'index_pip', 'index_dip', 'index_tip', 
                                          'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip', 
                                          'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip', 
                                          'little_mcp', 'little_pip', 'little_dip', 'little_tip', 
                                          'thumb_mcp', 'thumb_pip', 'thumb_dip', 'thumb_tip'])}

joints_min_inds = [ joints_inds[j] for j in ['wrist', 'middle_mcp', 'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'little_tip']]
joints_cp_inds = [ joints_inds[j] for j in [ 'wrist' ] +\
                        ['thumb_pip', 'thumb_dip', 'thumb_tip'] +\
                        [ '{}_{}'.format(finger, part) for finger in  ['index', 'middle', 'ring', 'little' ] \
                         for part in ['mcp', 'pip', 'dip', 'tip']] ]



# Load skeletons and labels from original annotation files.
# Transform the joints to the specfied format
def load_data(data_format = 'common_minimal'):
    total_data = { sbj:{} for sbj in subjects }
    for sbj in subjects:
        for a in actions:
            with open(os.path.join(path_dataset, sbj, a, 'joint.txt')) as f: skels = f.read().splitlines()[1:]
            skels = np.array([ list(map(float, l.split())) for l in skels ])
            skels = skels.reshape((skels.shape[0], 21, 3))
    
            if data_format == 'common_minimal':
                skels = skels[:,joints_min_inds]
            elif data_format == 'common':
                skels = skels[:,joints_cp_inds]
            total_data[sbj][a] = skels

    return total_data
           

# Split skels into different action sequences if seq_len != -1
def actions_to_samples(total_data, seq_len):
    # if seq_len == -1: return total_data
    for sbj in subjects:
        for a in actions:
            
            if seq_len == -1: 
                total_data[sbj][a] = [total_data[sbj][a]]
            else:
                skels = total_data[sbj][a]
                # samples = np.array_split(skels, (len(skels)//seq_len)+1)
                samples = [ skels[i:i+seq_len] for i in np.arange(0, len(skels), seq_len) ]
                if len(samples[-1]) < seq_len//2: samples = samples[:-1]
            
                total_data[sbj][a] = samples
            
    return total_data




# Create data folds from action sequences stored in total_data
# cross-actions splits just by label
# cross-subject splits by label
# cross-subject-sequence split full videos by subject
def get_folds(total_data, n_splits=3):

    actions_list = np.array([ s      for sbj in subjects for act in actions for s in total_data[sbj][act] ])
    actions_labels = np.array([ act  for sbj in subjects for act in actions for s in total_data[sbj][act] ])
    actions_sbj = np.array([ sbj     for sbj in subjects for act in actions for s in total_data[sbj][act] ])
    actions_anns = np.array([ '{}_{}_{}'.format(sbj, act, i) for sbj in subjects for act in actions for i,s in enumerate(total_data[sbj][act]) ])
    actions_label_sbj = np.array([ sbj+'_'+act    for sbj in subjects for act in actions for s in total_data[sbj][act] ])
    
    shuff_inds = np.random.RandomState(seed=0).permutation(len(actions_list))
    actions_list = actions_list[shuff_inds]
    actions_labels = actions_labels[shuff_inds]
    actions_sbj = actions_sbj[shuff_inds]
    actions_anns = actions_anns[shuff_inds]
    actions_label_sbj = actions_label_sbj[shuff_inds]
    
    # cross-actions
    folds = {}
    # for num_fold, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=3).split(np.zeros(actions_label_sbj), actions_label_sbj)):
    # for num_fold, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=n_splits).split(actions_list, actions_label_sbj)):
    for num_fold, (train_index, test_index) in enumerate(StratifiedKFold(n_splits=n_splits).split(actions_list, actions_labels)):
        folds[num_fold] = {'indexes': test_index.tolist(), 
                           'annotations': actions_anns[test_index].tolist(),
                           'labels': actions_labels[test_index].tolist(),
                           }
 
        
    # cross-subject
    folds_subject = {}
    for num_fold, subject in enumerate(subjects):
        indexes = [ ind for ind, ann in enumerate(actions_anns) if subject in ann ]
        folds_subject[num_fold] = {'indexes': indexes, 
                                   'annotations': [ ann for ind, ann in enumerate(actions_anns) if subject in ann ],
                                   'labels': actions_labels[indexes].tolist(),
                                    }  
        
    # cross-subjects-folds
    folds_subject_splits = {}
    # total_subjects = [ 'P{}'.format(i) for i in range(9)]  
    for num_fold in range(3):
        sbjs = [ 'P{}'.format(i) for i in range(num_fold*3, num_fold*3+3) ]
        indexes = [ ind for sbj in sbjs for ind, ann in enumerate(actions_anns) if sbj in ann ]
        folds_subject_splits[num_fold] = {'indexes': indexes, 
                                   'annotations': [ str(ann) for sbj in sbjs for ind, ann in enumerate(actions_anns) if sbj in ann ],
                                   'labels': actions_labels[indexes].tolist(),
                                    }  

    # 3d PostureNet evaluation
    train_subjs = [ 'P{}'.format(i) for i in range(2, 9) ]
    test_subjs = ['P0', 'P1']
    train_indexes = [ ind for sbj in train_subjs for ind, ann in enumerate(actions_anns) if sbj in ann ]
    test_indexes = [ ind for sbj in test_subjs for ind, ann in enumerate(actions_anns) if sbj in ann ]
    folds_posturenet = {
            0: {'indexes': train_indexes, 
                'annotations': [ str(ann) for sbj in train_subjs for ind, ann in enumerate(actions_anns) if sbj in ann ],
                'labels': actions_labels[train_indexes].tolist(),
                 },
            1: {'indexes': test_indexes, 
                'annotations': [ str(ann) for sbj in test_subjs for ind, ann in enumerate(actions_anns) if sbj in ann ],
                'labels': actions_labels[test_indexes].tolist(),
                 }
        }
        
    return actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet


if __name__ == '__main__':
    # total_data = load_data()
    # total_data = actions_to_samples(total_data, 64)
    # actions_list, actions_labels, actions_label_sbj, folds, folds_subject = get_folds(total_data, n_splits=4)

    total_data = load_data('common_minimal')
    total_data_act = actions_to_samples(total_data, -1)
    actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet = get_folds(total_data_act, n_splits=4)


    # %%
    
    from collections import Counter

    for num_fold in range(len(folds_posturenet)):
        print(Counter(folds_posturenet[num_fold]['labels']).values())






