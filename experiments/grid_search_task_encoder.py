#!/usr/bin/python 

from common.gen_experiments import gen_experiments_dir, find_variables

import os
import time
import argparse

# os.system('pip install gitpython --user')
import git

os.environ['LANG'] = 'en_CA.UTF-8'

if __name__ == "__main__":

    exp_description = "prototypical_5shot"

    params = dict(
        repeat=list(range(0, 10)),  # used to repeate the same experiment
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
        gradient_noise=1.0,
        sqrt_noise_anneal=False,
        num_units_in_block=3,
        num_blocks=4,
        num_max_pools=[3],
        num_filters=64,  # [32, 64, 128],
        block_size_growth=2.0,  # [1.0, 1.5, 2.0],
        embedding_pooled=True,
        weight_decay=0.0005,  # [0.0002, 0.0005, 0.00075, 0.001], # 0.0005 Seems to be optimal
        weight_decay_cbn=0.01,
        init_learning_rate=0.1,  # [0.1, 0.001],
        augment=False,
        number_of_steps=30000,  # [5 * 5200, 3 * 5200],
        feature_extractor='simple_res_net',
        activation='swish-1',  # ['relu', 'selu', 'swish-1']
        encoder_sharing='shared',  # ['shared', 'siamese'],
        encoder_classifier_link=['cbn'],  # ['polynomial','prototypical','cosine', 'cbn'], 
        cbn_premultiplier='var',
        cbn_num_layers=[3],
        cbn_per_block=False,
        cbn_per_network=False,
        metric_multiplier_init=1.0,  # [0.5, 1.0, 5.0, 7.5, 10.0, 20.0], 
        metric_multiplier_trainable=False,
        polynomial_metric_order=1,  # [1, 2, 3, 4, 5],  
        attention_fusion='sum',  # ['sum', 'highway', 'weighted'],
        attention_no_original_embedding=True,
        dropout=1.0,
        class_embed_size=None,  # 32
        attention_num_filters=128,  # [64, 128]
        task_encoder='class_mean',  # ['talkthrough', 'class_mean', 'label_embed', 'self_attention']
        num_self_attention_splits=4,
        weights_initializer_factor=0.1,
        num_attention_models=16,  # [8, 16]
        num_attention_layers=1,
        n_lr_decay=3,
        lr_decay_rate=10.0,
        lr_anneal='pwc',
        fc_dropout=0.0,  # [None, 0.1, 0.2, 0.5],
        feat_extract_pretrain=['multitask'],  # [None, 'finetune', 'freeze', 'multitask']
        feat_extract_pretrain_offset=15000,
        feat_extract_pretrain_decay_rate=0.9,  # TODO explore with more restarts [0.8, 0.9], seems like 0.8 better
        feat_extract_pretrain_decay_n=20,
        feat_extract_pretrain_lr_decay_rate=10.0,  # [5.0, 10.0, 100.0]
        feat_extract_pretrain_lr_decay_n=2.0,   # 1.6 [1.6, 2.6, 3.0],
        feature_dropout_p=None,
        feature_bottleneck_size=None,
        feature_expansion_size=None,
        num_classes_pretrain=64,
        task_encoder_sharing=None,
        aux_num_classes_test=10,
        aux_num_shots=1,
        aux_decay_rate=None,  # 0.9
        aux_decay_n=20,
        aux_lr_decay_rate=5.0,
        aux_lr_decay_n=1.6,  # [1.6, 2.6, 3.0],
        conv_dropout=None,
        num_batches_neg_mining=0,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aidata_home', type=str, default="/home/boris/", help='The path of your home in /mnt/.')

    aidata_home = parser.parse_known_args()[0].aidata_home
    exp_tag = '_'.join(find_variables(params))  # extract variable names
    exp_dir = os.path.join(aidata_home, "experiments_task_encoder",
                           "%s_mini_imagenet_%s_%s" % (time.strftime("%y%m%d_%H%M%S"), exp_tag, exp_description))

    project_path = os.path.join(aidata_home, "dev/TADAM")
    borgy_args = [
        "--image=images.borgy.elementai.lan/tensorflow/tensorflow:1.4.1-devel-gpu-py3",
        "-e", "PYTHONPATH=%s" % project_path,
        "-e", "DATA_PATH=/mnt/datasets/public/",
        "-v", "/mnt/datasets/public/:/mnt/datasets/public/",
        "--req-cores=2",
        "--req-gpus=1",
        "--req-ram-gbytes=16",
        "--restartable"
    ]

    # This is for the reproducibility purposes
    repo = git.Repo(path='/mnt' + project_path)
    params['commit'] = repo.head.object.hexsha

    cmd = os.path.join(project_path, "model/tadam.py")

    gen_experiments_dir(params, exp_dir, exp_description, cmd, blocking=True, borgy_args=borgy_args)
