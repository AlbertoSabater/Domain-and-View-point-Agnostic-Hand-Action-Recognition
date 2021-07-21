# Domain and View-point Agnostic Hand Action Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-and-view-point-agnostic-hand-action/skeleton-based-action-recognition-on-first)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-first?p=domain-and-view-point-agnostic-hand-action)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/domain-and-view-point-agnostic-hand-action/skeleton-based-action-recognition-on-shrec)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-shrec?p=domain-and-view-point-agnostic-hand-action)


This repository contains the code to train and evaluate the work presented in the article [Domain and View-point Agnostic Hand Action Recognition](https://arxiv.org/abs/2103.02303).

![Motion representation model](https://github.com/AlbertoSabater/Domain-View-point-Agnostic-Hand-Action-Recognition/blob/main/TCN_pipeline.png)


## Download pre-trained models

Download the desired models used in the paper and store them under `./pretrained_models/`.

### Cross-domain
* [Last TCN descriptor](https://drive.google.com/file/d/1G8etWqt--gKG7P9NwEYIkSEw56JCdlBB/view?usp=sharing)
* [Summarization (Best)](https://drive.google.com/file/d/14SAHXg6TWNSc8pWjhuEDN2sQadbXTi_q/view?usp=sharing)

### Intra-domain
* [F-PHAB models](https://drive.google.com/file/d/1yF6fAxgabas3juLb6TDBSbAek2YhbYZ8/view?usp=sharing)
* [SHREC17 models](https://drive.google.com/file/d/19dliuo0MJv0seOcOVd-2L2kQHvsASo3q/view?usp=sharing)


## Data format

The present project uses skeleton representations based on the 20-joints that SHREC17 and F-PHAB have in common.

In F-PHAB Dataset: `Wrist, TPIP, TDIP, TTIP, IMCP, IPIP, IDIP, ITIP, MMCP, MPIP, MDIP, MTIP, RMCP, RPIP, RDIP, RTIP, PMCP, PPIP, PDIP, PTIP`

In SHREC-17 Dataset: `Wrist, thumb_first_joint, thumb_second_joint, thumb_tip, index_base, index_first_joint, index_second_joint, index_tip, middle_base, middle_first_joint, middle_second_joint, middle_tip, ring_base, ring_first_joint, ring_second_joint, ring_tip, pinky_base, pinky_first_joint, pinky_second_joint, pinky_tip`

The **7-joints minimal** skeleton representation proposed in the paper uses the skeleton joints indexed by `0,8,3,7,11,15,19`, which stands for `Wrist, middle_base, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip`.

![Skeleton representations](https://github.com/AlbertoSabater/Domain-View-point-Agnostic-Hand-Action-Recognition/blob/main/minimal_hand_v2.png)

Original skeletons files can be transformed to the 20-joints format with the scripts for [SHREC17](https://github.com/AlbertoSabater/Domain-View-point-Agnostic-Hand-Action-Recognition/blob/main/dataset_scripts/common_pose/shrec17_to_common_pose.py) and [F-PHAB](https://github.com/AlbertoSabater/Domain-View-point-Agnostic-Hand-Action-Recognition/blob/main/dataset_scripts/common_pose/f-phab_to_common_pose.py), and stored under `./datasets/`. Minimal 7-joints format is later obtained with the `DataGenerator`.

F-PHAB data splits used for the evaluation are located under `./dataset_scripts/F_PHAB/paper_tables_annotations/`.


## Python dependencies

Project tested with the following dependencies:

 * python 3.6
 * tensorflow 2.3.0
 * Keras 2.3.1
 * keras-tcn 3.1.0
 * scikit-learn 0.22.2
 * scipy 1.4.1
 * pandas 1.0.3


## Evaluate cross-domain action recognition

Execute the file `action_recognition_evaluation.py` to perform the cross-domain evaluation reported in the paper. The scripts loads skeleton actions sequences, augments them, generates sequence embeddings and performs a KNN classification. Final results show the accuracy calculated both with and without reference action data augmentation. Use the following flags to evaluate different datasets. `--eval_fphab`, `--eval_msra`

To reproduce the results given in the Table III from the paper, download the cross-domain models and execute the following commands:

`python action_recognition_evaluation.py --path_model ./pretrained_models/xdom_last_descriptor --loss_name mixknn_train5_val1 --eval_fphab`

`python action_recognition_evaluation.py --path_model ./pretrained_models/xdom_summarization --loss_name mixknn_best --eval_fphab --eval_msra`

Note that, since random operations are involved in the evaluation, final results can slightly differ from the results reported in the paper.
