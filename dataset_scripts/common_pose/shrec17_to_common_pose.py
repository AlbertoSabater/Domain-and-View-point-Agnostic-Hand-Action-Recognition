# -*- coding: utf-8 -*-

"""
Create annotations using just the joints from common_pose
"""

import os
import numpy as np

joints_num = 20
shrec_joints = { s.split('.')[1]:joint_num for joint_num, s in enumerate('1.Wrist, 2.Palm, 3.thumb_base, 4.thumb_first_joint, 5.thumb_second_joint, 6.thumb_tip, 7.index_base, 8.index_first_joint, 9.index_second_joint, 10.index_tip, 11.middle_base, 12.middle_first_joint, 13.middle_second_joint, 14.middle_tip, 15.ring_base, 16.ring_first_joint, 17.ring_second_joint, 18.ring_tip, 19.pinky_base, 20.pinky_first_joint, 21.pinky_second_joint, 22.pinky_tip'.split(', ')) }

if joints_num == 20:
    common_pose_joints = [ 'Wrist' ] +\
                        ['thumb_first_joint', 'thumb_second_joint', 'thumb_tip'] +\
                        [ '{}_{}'.format(finger, part) for finger in  ['index', 'middle', 'ring', 'pinky'] for part in ['base', 'first_joint', 'second_joint', 'tip'] ]

common_pose_joint_inds = [ shrec_joints[s] for s in common_pose_joints ]


# %%

common_pose_dataset_path = '/home/asabater/datasets/common_pose/'
common_pose_dataset_path_dest = './datasets/common_pose/'
base_dataset = '/home/asabater/datasets/HandGestureDataset_SHREC2017/'

# =============================================================================
# Translate and store skeleton coordinates
# =============================================================================

for split_mode in ['train', 'val']:
    path_split = base_dataset + '{}_gestures.txt'.format('test' if split_mode == 'val' else 'train')
    with open(path_split, 'r') as f: anns = f.read().splitlines()
    anns = [ list(map(int, l.split())) for l in anns ]
    
    
    store_14 = open('./annotations/SHREC2017/' + 'annotations_{}_{}_jn{}.txt'.format(split_mode, '14', joints_num), 'w')
    store_28 = open('./annotations/SHREC2017/' + 'annotations_{}_{}_jn{}.txt'.format(split_mode, '28', joints_num), 'w')

    for ann in anns:
        id_gesture, id_finger, id_subject, id_essai, label_14, label_28, size_sequence = ann
        skel_path = base_dataset + '/gesture_{}/finger_{}/subject_{}/essai_{}/skeletons_world.txt'.format(
            id_gesture, id_finger, id_subject, id_essai)
        
        with open(skel_path, 'r') as f: joints = f.read().splitlines()
        joints = np.array([ l.split() for l in joints ]).reshape((len(joints), 22, 3))
    
        new_skel_path = os.path.join(common_pose_dataset_path, 'SHREC2017', 'gesture{}_finger{}_subject{}_essai{}_{}_jn{}.txt'.format(id_gesture, id_finger, id_subject, id_essai, split_mode, joints_num))
        new_skel_path_dest = os.path.join(common_pose_dataset_path_dest, 'SHREC2017', 'gesture{}_finger{}_subject{}_essai{}_{}_jn{}.txt'.format(id_gesture, id_finger, id_subject, id_essai, split_mode, joints_num))
        joints_new = joints[:, common_pose_joint_inds]
        
        with open(new_skel_path, 'w') as f:
            for frame_joints in joints_new:
                frame_joints = ' '.join(frame_joints.flatten().tolist())
                f.write(frame_joints + '\n')
            
        store_14.write(new_skel_path_dest + ' ' + str(label_14) + '\n')
        store_28.write(new_skel_path_dest + ' ' + str(label_28) + '\n')        
         






