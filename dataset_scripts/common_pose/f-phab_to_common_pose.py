# -*- coding: utf-8 -*-

import os
import numpy as np


joints_num = 20

fp_joints = { s:joint_num for joint_num,s in enumerate('Wrist, TMCP, IMCP, MMCP, RMCP, PMCP, TPIP, TDIP, TTIP, IPIP, IDIP, ITIP, MPIP, MDIP, MTIP, RPIP, RDIP, RTIP, PPIP, PDIP, PTIP'.split(', ')) }

if joints_num == 20:
    common_pose_joints = [ 'Wrist' ] +\
                        ['TPIP', 'TDIP', 'TTIP'] +\
                        [ '{}{}'.format(finger, part) for finger in  ['I', 'M', 'R', 'P' ] \
                         for part in ['MCP', 'PIP', 'DIP', 'TIP'] ]

common_pose_joint_inds = [ fp_joints[s] for s in common_pose_joints ]


# %%

common_pose_dataset_path = '/home/asabater/datasets/common_pose/'
common_pose_dataset_path_dest = './datasets/common_pose/'
base_dataset = '/home/asabater/datasets/F-PHAB/'

annotations_filename = base_dataset + 'data_split_action_recognition.txt'
with open(annotations_filename, 'r') as f: anns = f.read().splitlines()

anns_train = anns[1:601]
anns_val = anns[602::]


for split_mode, anns in [('train', anns_train), ('val', anns_val)]:
    # Create annotations filename
    store = open('./annotations/F_PHAB/' + 'annotations_{}_jn{}.txt'.format(split_mode, joints_num), 'w')
    for ann in anns:
        
        # Read and parse skel
        filename, label = ann.split()
        with open(os.path.join(base_dataset, 'Hand_pose_annotation_v1', filename, 'skeleton.txt'), 'r') as f: joints = f.read().splitlines()
        joints = np.array([ l.split() for l in joints ])[:,1:].reshape((len(joints), 21, 3))
        
        #
        new_skel_path = os.path.join(common_pose_dataset_path, 'F-PHAB', '{}_{}_jn{}.txt'.format(filename.replace('_', '').replace('/', '_'), split_mode, joints_num))
        new_skel_path_dest = os.path.join(common_pose_dataset_path_dest, 'F-PHAB', '{}_{}_jn{}.txt'.format(filename.replace('_', '').replace('/', '_'), split_mode, joints_num))
        joints_new = joints[:, common_pose_joint_inds]        
        
        with open(new_skel_path, 'w') as f:
            for frame_joints in joints_new:
                frame_joints = ' '.join(frame_joints.flatten().tolist())
                f.write(frame_joints + '\n')        

        store.write(new_skel_path_dest + ' ' + str(label) + '\n')
        

