#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/data/home/anonymous/Med-LISA/data
model_root=/data/home/anonymous/Med-LISA/model
path_save_log=/data/home/anonymous/Med-LISA/OPTIC/logs/

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS]
Source=RIM_ONE_r3

#Optimizer
optimizer=Adam
lr=0.05

#Hyperparameters
memory_size=40
neighbor=16
anchor_alpha=0.01
warm_n=5

#Command
cd OPTIC
CUDA_VISIBLE_DEVICES=0 python lisa.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--warm_n $warm_n