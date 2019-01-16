#!/usr/bin/python 

# Copyright (c) 2018 ELEMENT AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from common.gen_experiments import gen_experiments_dir, find_variables

import os
import time
import argparse

# os.system('pip install gitpython --user')
import git

os.environ['LANG'] = 'en_CA.UTF-8'

if __name__ == "__main__":

    exp_description = "scale_crossvalidation_MIN"

    params = dict(
        repeat=list(range(0, 1)),  # used to repeate the same experiment
        data_dir='/mnt/datasets/public/mini-imagenet',  # '/home/boris/cifar100',  # '/mnt/datasets/public/mini-imagenet',
        eval_interval_steps=5000,
        train_batch_size=32,  # [16, 32, 64]
        num_classes_train=5,  # [3, 5, 10, 15],
        num_classes_test=5,  # [3, 5, 10, 15],
        num_tasks_per_batch=2,
        pre_train_batch_size=64,
        num_shots_train=[5],  # [3, 5, 10],
        num_shots_test=5,
        optimizer='sgd',  # ['sgd', 'adam']
        num_units_in_block=3,
        num_blocks=4,
        num_max_pools=[3],
        num_filters=64,  # [32, 64, 128],
        block_size_growth=2.0,  # [1.0, 1.5, 2.0],
        embedding_pooled=True,
        weight_decay=0.0005,  # [0.0002, 0.0005, 0.00075, 0.001], # 0.0005 Seems to be optimal
        weight_decay_film=0.01,
        init_learning_rate=0.1,  # [0.1, 0.001],
        augment=False,
        number_of_steps=[30000, 60000],  # [5 * 5200, 3 * 5200],
        feature_extractor='simple_res_net',
        activation='swish-1',  # ['relu', 'selu', 'swish-1']
        metric=['film', 'polynomial'],  # ['polynomial','prototypical','cosine', 'film'],
        film_num_layers=[3],
        metric_multiplier_init=[0.5, 1.0, 5.0, 7.5, 10.0, 20.0],
        metric_multiplier_trainable=False,
        polynomial_metric_order=1,  # [1, 2, 3, 4, 5],  
        weights_initializer_factor=0.1,
        n_lr_decay=3,
        lr_decay_rate=10.0,
        lr_anneal='pwc',
        feat_extract_pretrain=['multitask', None],  # [None, 'finetune', 'freeze', 'multitask']
        feat_extract_pretrain_offset=15000,
        feat_extract_pretrain_decay_rate=0.9,  # TODO explore with more restarts [0.8, 0.9], seems like 0.8 better
        feat_extract_pretrain_decay_n=20,
        feat_extract_pretrain_lr_decay_rate=10.0,  # [5.0, 10.0, 100.0]
        feat_extract_pretrain_lr_decay_n=2.0,   # 1.6 [1.6, 2.6, 3.0],
        num_classes_pretrain=64,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aidata_home', type=str, default="/home/boris/", help='The path of your home in /mnt/.')

    aidata_home = parser.parse_known_args()[0].aidata_home
    exp_tag = '_'.join(find_variables(params))  # extract variable names
    exp_dir = os.path.join(aidata_home, "experiments_cleaning",
                           "%s_mini_imagenet_%s_%s" % (time.strftime("%y%m%d_%H%M%S"), exp_tag, exp_description))

    project_path = os.path.join(aidata_home, "dev/TADAM")
    
    # This is for the reproducibility purposes
    repo_path = '/mnt' + project_path
    repo = git.Repo(path=repo_path)
    params['commit'] = repo.head.object.hexsha
    
    borgy_args = [
        "--image=images.borgy.elementai.lan/tensorflow/tensorflow:1.4.1-devel-gpu-py3",
        "-e", "PYTHONPATH=%s" % repo_path,
        "-e", "DATA_PATH=/mnt/datasets/public/",
        "-v", "/mnt/datasets/public/:/mnt/datasets/public/",
        "-v", "/mnt/home/boris/:/mnt/home/boris/",
        "--cpu=2",
        "--gpu=1",
        "--mem=16",
        "--restartable"
    ]
    
    cmd = os.path.join(repo_path, "model/tadam.py")

    gen_experiments_dir(params, exp_dir, exp_description, cmd, blocking=True, borgy_args=borgy_args)
