#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:19:18 2021

@author: asabater
"""

import os

import os
import pickle
import numpy as np
from tqdm import tqdm
import time
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from data_generator import DataGenerator
from tensorflow.keras.preprocessing.sequence import pad_sequences



knn_neighbors = [1,3,5,7,9,11]
# aug_loop = [0,10,20,40]
aug_loop = [0,40]
num_augmentations = max(aug_loop)
weights = 'distance'

crop_train = 80000
crop_test = np.inf

np.random.seed(0)



def evaluate_folds(folds_data, embs, total_labels, num_augmentations=0, embs_aug=None, 
                   evaluate_all_folds=True, leave_one_out=True, groupby=None, return_sequences=False):
    
    res = {}
    num_folds = len(folds_data) if evaluate_all_folds else 1
    for num_fold in range(num_folds):
        if leave_one_out:
            train_indexes =  np.concatenate([ f['indexes'] for i,f in folds_data.items() if i!= num_fold])
            test_indexes = folds_data[num_fold]['indexes']
        else:
            train_indexes = folds_data[num_fold]['indexes']
            test_indexes =  np.concatenate([ f['indexes'] for i,f in folds_data.items() if i!= num_fold])


        X_train = embs[train_indexes]
        X_test = embs[test_indexes]
        y_train = total_labels[train_indexes]
        y_test = total_labels[test_indexes]
        
        
        if groupby is not None:
            _, groups_test_true = groupby[train_indexes], groupby[test_indexes]
            
            
        if num_augmentations>0:
            X_train = np.concatenate([X_train] + [ embs_aug[i][train_indexes] for i in range(num_augmentations) ])
            y_train = np.concatenate([ y_train for i in range(num_augmentations+1) ])
 
            
        if return_sequences:
            if groupby is not None: groups_test_true = np.array([ y for y,seq in zip(groups_test_true, X_test) for _ in range(len(seq)) ])
            y_train = [ y for y,seq in zip(y_train, X_train) for _ in range(len(seq)) ]
            y_test = [ y for y,seq in zip(y_test, X_test) for _ in range(len(seq)) ]
            X_train = np.concatenate(X_train)
            X_test = np.concatenate(X_test)
            
            
        if groupby is not None and len(y_test) > crop_test:
            print('Cropping test results:', len(y_test))
            idx = np.random.choice(np.arange(len(y_test)), crop_test, replace=False)
            y_test = np.array(y_test)[idx].tolist()
            X_test = X_test[idx]
            groups_test_true = groups_test_true[idx]
 

        if len(y_train) > crop_train: 
            idx = np.random.choice(np.arange(len(y_train)), crop_train, replace=False)
            X_train = X_train[idx]
            y_train = np.array(y_train)[idx]

        res[num_fold] = {}
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=8, weights=weights).fit(X_train, y_train)
        for n in knn_neighbors:
            knn = knn.set_params(**{'n_neighbors': n})
            
            t = time.time()
            if groupby is not None: 
                preds_proba = knn.predict_proba(X_test)
                classes = sorted(list(set(y_train)))
                groups = list(set(groups_test_true))
                g_true, g_preds = [], []
                for g in groups:
                    g_true.append(g.split('_')[1])
                    g_inds = np.where(groups_test_true == g)
                    
                    g_pred = preds_proba[g_inds].mean(axis=0)
                    g_preds.append(classes[np.where(g_pred == g_pred.max())[0][0]])
                    
                acc = accuracy_score(g_true, g_preds)
            else: 
                preds = knn.predict(X_test)
                acc = accuracy_score(y_test, preds)
            res[num_fold][n] = acc
            tf = time.time()
            print(' ** Classification time ** num_fold [{}] | k [{}] | X_test [{}] | time [{:.3f}s] | ms per sequence [{:.3f}]'.format(num_fold, n, 
                                                                       len(X_test), tf-t, (tf-t)*1000/len(X_test)))

    res = { n:np.mean([ res[num_fold][n] for num_fold in range(num_folds) ]) for n in knn_neighbors }
    

    return res
        




# %%

# =============================================================================
# F-PHAB
# =============================================================================

def load_fphab_data():
    # Load all annotations
    annotations_store_folder = './dataset_scripts/F_PHAB/paper_tables_annotations'
    with open(os.path.join(annotations_store_folder, 'total_annotations_jn{}.txt'.format(20)), 'r') as f: 
        total_annotations = f.read().splitlines()
    total_labels = np.stack([ l.split()[-1] for l in total_annotations ])
    total_annotations = [ l.split()[0] for l in total_annotations ]
    
    
    # Load evaluation folds
    folds_1_1 = pickle.load(open(os.path.join(annotations_store_folder, 'annotations_table_2_cross-action_folds_1_1_jn{}.pckl'.format(20)), 'rb'))
    folds_base = pickle.load(open(os.path.join(annotations_store_folder, 'annotations_table_2_cross-action_folds_jn{}.pckl'.format(20)), 'rb'))
    folds_subject = pickle.load(open(os.path.join(annotations_store_folder, 'annotations_table_2_cross-person_folds_jn{}.pckl'.format(20)), 'rb'))
    return total_annotations, total_labels, folds_1_1, folds_base, folds_subject


from joblib import Parallel, delayed

def load_actions_sequences_data_gen(total_annotations, num_augmentations, model_params, load_from_files=True, return_sequences=False):
    np.random.seed(0)
    data_gen = DataGenerator(**model_params)
    if return_sequences: data_gen.max_seq_len = 0
    
    if load_from_files: skels_ann = [ data_gen.load_skel_coords(ann) for ann in total_annotations ]
    else: skels_ann = total_annotations

    def get_pose_features(validation=False):
        action_sequences = [ data_gen.get_pose_data_v2(skels, validation=validation) for skels in skels_ann ]
        if not return_sequences: action_sequences = pad_sequences(action_sequences, abs(model_params['max_seq_len']), dtype='float32', padding='pre')
        return action_sequences
    
    action_sequences = get_pose_features(validation=True)
    print('* Data sequences loaded')
    
    if num_augmentations > 0:
        action_sequences_augmented = Parallel(n_jobs=8)(delayed(get_pose_features)(validation=False) for i in tqdm(range(num_augmentations)))
        print('* Data sequences augmented')
    else: action_sequences_augmented = None
    
    return action_sequences, action_sequences_augmented



def get_tcn_embeddings(model, action_sequences, action_sequences_augmented, return_sequences=False):
    t = time.time()
    
    model.set_encoder_return_sequences(return_sequences)
    # Get embeddings from all annotations
    if return_sequences: 
        # embs = np.array([ model.get_embedding(s[None]).numpy()[0] for s in action_sequences ])
        embs = [ model.get_embedding(s[None]) for s in action_sequences ]
    else: 
        embs = [ model.get_embedding(s) for s in np.array_split(action_sequences, max(1, len(action_sequences)//1)) ]
    print('* Embeddings calculated')
    if num_augmentations > 0:
        if return_sequences: 
            embs_aug = [ [ model.get_embedding(s[None]) for s in samples ] for samples in tqdm(action_sequences_augmented) ]
        else: 
            embs_aug = [ [ model.get_embedding(s) for s in np.array_split(samples, max(1, len(samples)//1)) ] for samples in tqdm(action_sequences_augmented) ]
        print('* Augmented embeddings calculated')
    else: embs_aug = None
    
    tf = time.time()
    sys.stdout.flush
    
    if return_sequences: embs = np.array([ e[0] for e in embs ])
    else: embs = np.concatenate([ e for e in embs ])
    if num_augmentations > 0:
        if return_sequences: embs_aug = np.array([ [ s[0] for s in samples ] for samples in embs_aug ])
        else: embs_aug = [ np.concatenate([ s for s in samples ]) for samples in embs_aug ]
    
    num_sequences = len(embs)
    if num_augmentations > 0: num_sequences += sum([ len(e) for e in embs_aug ])
    print(' ** Prediction time **   Secuences evaluated [{}] | time [{:.3f}s] | ms per sequence [{:.3f}]'.format(num_sequences, tf-t, (tf-t)*1000/num_sequences))
 
    return embs, embs_aug


def evaluate_fphab(aug_loop, folds_base, folds_1_1, folds_subject, embs, embs_aug, total_labels, knn_neighbors):
    total_res = {}
    for n_aug in aug_loop:
        total_res[n_aug] = {}
        print(n_aug, '1:3')
        total_res[n_aug]['1:3'] = evaluate_folds(folds_base, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False)
        print(n_aug, '1:1')
        total_res[n_aug]['1:1'] = evaluate_folds(folds_1_1, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False, evaluate_all_folds=False)
        print(n_aug, '3:1')
        total_res[n_aug]['3:1'] = evaluate_folds(folds_base, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=True)
        print(n_aug, 'cross_sub')
        total_res[n_aug]['cross_sub'] = evaluate_folds(folds_subject, embs, total_labels, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=True)

    return total_res


def print_results(dataset_name, total_res, knn_neighbors, aug_loop, frame=True):
    if frame: print('-'*81)
    print('# | {} | {}'.format(dataset_name,
            ' | '.join([ '[{}] {:.1f} / {:.1f}'.format(k, max([ total_res[0][k][n] for n in knn_neighbors ])*100,
              max([ total_res[na][k][n] for n in knn_neighbors for na in aug_loop ])*100) for k in total_res[0].keys() ])
        ))
    if frame: print('-'*81)
    
    
# %%

# =============================================================================
# SHREC 14/28
# =============================================================================

def load_shrec_data():
    base_path = './'
    shrec_train_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn{}.txt'.format('14', 20)
    shrec_val_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn{}.txt'.format('14', 20)
    with open(os.path.join(base_path+shrec_train_anns), 'r') as f: shrec_train_anns = f.read().splitlines()
    with open(os.path.join(base_path+shrec_val_anns), 'r') as f: shrec_val_anns = f.read().splitlines()
    
    # Load data annotations
    total_shrec_anns = shrec_train_anns + shrec_val_anns
    shrec_train_indexes = [ total_shrec_anns.index(ann) for ann in shrec_train_anns ]
    shrec_val_indexes = [ total_shrec_anns.index(ann) for ann in shrec_val_anns ]
    total_shrec_anns = np.stack([ l.split()[0] for l in total_shrec_anns ])

    # Create data folds
    shrec_train_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn{}.txt'.format('14', 20)
    shrec_val_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn{}.txt'.format('14', 20)
    with open(os.path.join(base_path+shrec_train_anns), 'r') as f: shrec_train_anns = f.read().splitlines()
    with open(os.path.join(base_path+shrec_val_anns), 'r') as f: shrec_val_anns = f.read().splitlines()
    total_shrec_labels_14 = np.stack([ l.split()[-1] for l in shrec_train_anns+shrec_val_anns ])

    shrec_folds_14 = {
        0:{'indexes': shrec_train_indexes, 'annotations': total_shrec_anns[shrec_train_indexes], 'labels': total_shrec_labels_14[shrec_train_indexes]},
        1:{'indexes': shrec_val_indexes, 'annotations': total_shrec_anns[shrec_val_indexes], 'labels': total_shrec_labels_14[shrec_val_indexes]},
                   }
    
    # Create data folds
    shrec_train_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_train_{}_jn{}.txt'.format('28', 20)
    shrec_val_anns = './dataset_scripts/common_pose/annotations/SHREC2017/annotations_val_{}_jn{}.txt'.format('28', 20)
    with open(os.path.join(base_path+shrec_train_anns), 'r') as f: shrec_train_anns = f.read().splitlines()
    with open(os.path.join(base_path+shrec_val_anns), 'r') as f: shrec_val_anns = f.read().splitlines()
    total_shrec_labels_28 = np.stack([ l.split()[-1] for l in shrec_train_anns+shrec_val_anns ])

    shrec_folds_28 = {
        0:{'indexes': shrec_train_indexes, 'annotations': total_shrec_anns[shrec_train_indexes], 'labels': total_shrec_labels_28[shrec_train_indexes]},
        1:{'indexes': shrec_val_indexes, 'annotations': total_shrec_anns[shrec_val_indexes], 'labels': total_shrec_labels_28[shrec_val_indexes]},
                   }
    
    return total_shrec_anns, shrec_folds_14, shrec_folds_28, total_shrec_labels_14, total_shrec_labels_28
 

def evaluate_shrec(aug_loop, shrec_folds_14, shrec_folds_28, embs, embs_aug, total_shrec_labels_14, total_shrec_labels_28, knn_neighbors):
    total_res_shrec = {}
    for n_aug in aug_loop:
        print('***', n_aug, '***')
        total_res_shrec[n_aug] = {}
        print(n_aug, '14')
        total_res_shrec[n_aug]['14'] = evaluate_folds(shrec_folds_14, embs, total_shrec_labels_14, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False, evaluate_all_folds=False)
        print(n_aug, '28')
        total_res_shrec[n_aug]['28'] = evaluate_folds(shrec_folds_28, embs, total_shrec_labels_28, num_augmentations=n_aug, embs_aug=embs_aug, leave_one_out=False, evaluate_all_folds=False)
    return total_res_shrec


# %%

# =============================================================================
# MSRA functions
# =============================================================================

def load_MSRA_data(model_params, n_splits=4, seq_perc=1, data_format = 'common_minimal'):
    from dataset_scripts.MSRA import load_data
    np.random.seed(0)
    if seq_perc == -1: total_data = load_data.actions_to_samples(load_data.load_data(data_format), -1)
    else: total_data = load_data.actions_to_samples(load_data.load_data(data_format), int(abs(model_params['max_seq_len']*model_params['skip_frames'][0]*seq_perc)))
    actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet = \
        load_data.get_folds(total_data, n_splits=n_splits)
    return actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet

def evaluate_MSRA(aug_loop, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet, 
                  embs, embs_aug, actions_labels, knn_neighbors, return_sequences=False):
    total_res = {}
    for n_aug in aug_loop:
        print('***', n_aug, '***')
        total_res[n_aug] = {}

        print(n_aug, 'posturenet')
        total_res[n_aug]['posturenet'] = evaluate_folds(folds_posturenet, embs, actions_labels, num_augmentations=n_aug, \
                                                        embs_aug=embs_aug, leave_one_out=False, \
                                                        evaluate_all_folds = False,
                                                        groupby=actions_label_sbj, return_sequences=return_sequences)
        print(n_aug, 'posturenet_online')
        total_res[n_aug]['posturenet_online'] = evaluate_folds(folds_posturenet, embs, actions_labels, num_augmentations=n_aug, \
                                                        embs_aug=embs_aug, leave_one_out=False, \
                                                        evaluate_all_folds = False,
                                                        groupby=None, return_sequences=return_sequences)


    return total_res



# %%

if __name__ == '__main__':
    # %%

    # =============================================================================
    # Load model
    # =============================================================================
    
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import prediction_utils
    import time
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate action recognition in different datasets')
    parser.add_argument('--path_model', type=str, help='path to the trained model')
    parser.add_argument('--loss_name', type=str, help='key to load weights')
    parser.add_argument('--eval_fphab', action='store_true', help='evaluate on F-PHAB splits')
    parser.add_argument('--eval_shrec', action='store_true', help='evaluate on SHREC splits')
    parser.add_argument('--eval_msra', action='store_true', help='evaluate on MSRA dataset')
    args = parser.parse_args()

    model, model_params = prediction_utils.load_model(args.path_model, False, loss_name = args.loss_name)
    model_params['use_rotations'] = None
    
    print('* Model loaded')
    
    

    # %%
    
    # F-PHAB
    if args.eval_fphab:
        t = time.time()
        total_annotations, total_labels, folds_1_1, folds_base, folds_subject = load_fphab_data()
        action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(total_annotations, num_augmentations, model_params)
        embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented)
        total_res_fphab = evaluate_fphab(aug_loop, folds_base, folds_1_1, folds_subject, embs, embs_aug, total_labels, knn_neighbors)
        print_results('F-PHAB   ', total_res_fphab, knn_neighbors, aug_loop)
        print('Time elapsed: {:.2f}'.format((time.time()-t)/60))
        del embs; del embs_aug; del action_sequences; del action_sequences_augmented;
    
    # %%
    
    # SHREC
    if args.eval_shrec:
        t = time.time()
        total_shrec_annotations, shrec_folds_14, shrec_folds_28, total_shrec_labels_14, total_shrec_labels_28 = load_shrec_data()
        action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(total_shrec_annotations, num_augmentations, model_params)
        embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented)
        total_res_shrec = evaluate_shrec(aug_loop, shrec_folds_14, shrec_folds_28, embs, embs_aug, total_shrec_labels_14, total_shrec_labels_28, knn_neighbors)
        print_results('SHREC    ', total_res_shrec, knn_neighbors, aug_loop)
        print('Time elapsed: {:.2f}'.format((time.time()-t)/60))
        
        del embs; del embs_aug; del action_sequences; del action_sequences_augmented;
    
    
    
    # %%
    
    # MSRA full -> return_sequences == True
    if args.eval_msra:
        t = time.time()
        model_params['skip_frames'] = [1]
        actions_list, actions_labels, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet = \
                                    load_MSRA_data(model_params, n_splits=4, 
                                                               # seq_perc=0.2, data_format=model_params['joints_format'])
                                                                seq_perc=-1, data_format=model_params['joints_format'])
        action_sequences, action_sequences_augmented = load_actions_sequences_data_gen(actions_list, num_augmentations, model_params, 
                                                                           load_from_files=False, return_sequences=True)
        embs, embs_aug = get_tcn_embeddings(model, action_sequences, action_sequences_augmented, return_sequences=True)
        total_res_msra_full = evaluate_MSRA(aug_loop, actions_label_sbj, folds, folds_subject, folds_subject_splits, folds_posturenet,
                                            embs, embs_aug, actions_labels, knn_neighbors, return_sequences=True)
        print_results('MSRA     ', total_res_msra_full, knn_neighbors, aug_loop)
        print('Time elapsed: {:.2f}'.format((time.time()-t)/60))
        del embs; del embs_aug; del action_sequences; del action_sequences_augmented;


    # %%
    
    print('='*80)
    print('='*80)
    print('='*80)
    print('='*80)
    if args.eval_fphab: print_results('F-PHAB   ', total_res_fphab, knn_neighbors, aug_loop, frame=False)
    if args.eval_shrec: print_results('SHREC    ', total_res_shrec, knn_neighbors, aug_loop, frame=False)
    if args.eval_msra: print_results( 'MSRA     ', total_res_msra_full, knn_neighbors, aug_loop, frame=False)
    
    print()
    print(args.loss_name, args.path_model)
    
# %%





