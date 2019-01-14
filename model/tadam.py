#!/usr/bin/env python3

"""Training and evaluation entry point."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes
from scipy.spatial import KDTree
from datasets.data import _load_mini_imagenet
from common.util import Dataset
from common.util import ACTIVATION_MAP
# from tqdm import trange
import pathlib
import logging
from common.util import summary_writer
from common.gen_experiments import load_and_save_params
import time

'''
Execute as a script:
go to deep-prior root and run
export PYTHONPATH=`pwd`
export CUDA_VISIBLE_DEVICES=0
go to deep-prior/deep_prior/experiements/mini-imagenet and run
python nearest_neighbor.py
'''

# docker pull gcr.io/tensorflow/tensorflow:1.4.1-devel-gpu-py3
# docker tag gcr.io/tensorflow/tensorflow:1.4.1-devel-gpu-py3 images.borgy.elementai.lan/tensorflow/tensorflow:1.4.1-devel-gpu-py3
# docker push images.borgy.elementai.lan/tensorflow/tensorflow:1.4.1-devel-gpu-py3

tf.logging.set_verbosity(tf.logging.INFO)
logging.basicConfig(level=logging.INFO)

# --data_dir=/home/boris/Downloads/cifar-100-python
# --data_dir=../../../data/mini-imagenet
DATA_DIR = os.path.join(os.environ['DATA_PATH'], 'mini-imagenet')


def get_image_size(data_dir):
    if 'mini-imagenet' in data_dir:
        image_size = 84
    elif 'cifar' in data_dir:
        image_size = 32
    else:
        raise Exception('Unknown dataset: %s' %data_dir)
    return image_size


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test', 'train_classifier'])
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='Path to the data.')
    parser.add_argument('--data_split', type=str, default='sources', choices=['sources', 'target_val', 'target_tst'], help='Split of the data to be used to perform operation.')

    # Training parameters
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--number_of_steps', type=int, default=int(30000),
                        help="Number of training steps")
    parser.add_argument('--number_of_steps_to_early_stop', type=int, default=int(1000000),
                        help="Number of training steps after half way to early stop the training")
    parser.add_argument('--log_dir', type=str, default='', help='Base log dir')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiment directory for Borgy')
    parser.add_argument('--num_classes_train', type=int, default=5, help='Number of classes in the train phase, this is coming from the prototypical networks')
    parser.add_argument('--num_shots_train', type=int, default=5, help='Number of shots in a few shot meta-train scenario')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--pre_train_batch_size', type=int, default=64, help='Batch size to pretrain feature extractor.')
    parser.add_argument('--num_tasks_per_batch', type=int, default=2,
                        help='Number of few shot tasks per batch, so the task encoding batch is num_tasks_per_batch x num_classes_test x num_shots_train .')
    parser.add_argument('--init_learning_rate', type=float, default=0.1, help='Initial learning rate.')
    parser.add_argument('--save_summaries_secs', type=int, default=60, help='Time between saving summaries')
    parser.add_argument('--save_interval_secs', type=int, default=60, help='Time between saving model?')
    parser.add_argument('--num_classes_pretrain', type=int, default=64,
                        help='number of classes when jointly training on the entire train dataset')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--augment', type=bool, default=False)
    # Learning rate paramteres
    parser.add_argument('--lr_anneal', type=str, default='pwc', choices=['const', 'pwc', 'exp'])
    parser.add_argument('--n_lr_decay', type=int, default=3)
    parser.add_argument('--lr_decay_rate', type=float, default=10.0)
    parser.add_argument('--num_steps_decay_pwc', type=int, default=2500, help='Decay learning rate every num_steps_decay_pwc')

    parser.add_argument('--clip_gradient_norm', type=float, default=1.0, help='gradient clip norm.')
    parser.add_argument('--weights_initializer_factor', type=float, default=0.1, help='multiplier in the variance of the initialization noise.')
    # Evaluation parameters
    parser.add_argument('--max_number_of_evaluations', type=float, default=float('inf'))
    parser.add_argument('--eval_interval_secs', type=int, default=120, help='Time between evaluating model?')
    parser.add_argument('--eval_interval_steps', type=int, default=1000, help='Number of train steps between evaluating model in the training loop')
    parser.add_argument('--eval_interval_fine_steps', type=int, default=250, help='Number of train steps between evaluating model in the training loop in the final phase')
    parser.add_argument('--num_samples_eval', type=int, default=12000, help='Number of evaluation samples?')
    parser.add_argument('--eval_batch_size', type=int, default=100, help='Evaluation batch size?')
    parser.add_argument('--num_evals', type=int, default=100, help='Number of evaluations in the evaluation phase')
    # Test parameters
    parser.add_argument('--num_classes_test', type=int, default=5, help='Number of classes in the test phase')
    parser.add_argument('--num_shots_test', type=int, default=5, help='Number of shots in a few shot meta-test scenario')
    parser.add_argument('--num_cases_test', type=int, default=50000,
                        help='Number of few-shot cases to compute test accuracy')
    parser.add_argument('--pretrained_model_dir', type=str,
                        default='./logs/batch_size-32-num_tasks_per_batch-1-lr-0.122-lr_anneal-pwc-epochs-100.0-optimizer-sgd-weight_decay-0.0005-augment-False-num_filters-64-feature_extractor-simple_res_net/train',
                        help='Path to the pretrained model to run the nearest neigbor baseline test.')
    # Architecture parameters
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--weight_decay_cbn', type=float, default=0.01)
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--num_units_in_block', type=int, default=3)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--num_max_pools', type=int, default=3)
    parser.add_argument('--block_size_growth', type=float, default=2.0)
    parser.add_argument('--activation', type=str, default='swish-1', choices=['relu', 'selu', 'swish-1'])
    parser.add_argument('--feature_extractor', type=str, default='simple_res_net',
                        choices=['simple_conv_net', 'simple_res_net'], help='Which feature extractor to use')
    # Feature extractor pretraining parameters (auxiliary 64-classification task)
    parser.add_argument('--feat_extract_pretrain', type=str, default=None,
                        choices=[None, 'finetune', 'freeze', 'multitask'], help='Whether or not pretrain the feature extractor')
    parser.add_argument('--feat_extract_pretrain_offset', type=int, default=15000)
    parser.add_argument('--feat_extract_pretrain_decay_rate', type=float, default=0.9, help='rate at which 64 way task selection probability decays in multitask mode')
    parser.add_argument('--feat_extract_pretrain_decay_n', type=int, default=20, help='number of times 64 way task selection probability decays in multitask mode')
    parser.add_argument('--feat_extract_pretrain_lr_decay_rate', type=float, default=10.0,
                        help='rate at which 64 way task learning rate decays')
    parser.add_argument('--feat_extract_pretrain_lr_decay_n', type=float, default=2.0,
                        help='number of times 64 way task learning rate decays')
    parser.add_argument('--encoder_classifier_link', type=str, default='polynomial',
                        choices=['cbn', 'prototypical', 'std_normalized_euc_head',
                                 'cosine', 'polynomial', 'cbn_cos'],
                        help='How to link fetaure extractors in task encoder and classifier')
    parser.add_argument('--embedding_pooled', type=bool, default=True,
                        help='Whether to use avg pooling to create embedding')
    parser.add_argument('--metric_multiplier_init', type=float, default=1.0, help='multiplier of cosine metric')
    parser.add_argument('--metric_multiplier_trainable', type=bool, default=False, help='multiplier of cosine metric trainability')
    parser.add_argument('--polynomial_metric_order', type=int, default=1)

    parser.add_argument('--cbn_num_layers', type=int, default=3)
    parser.add_argument('--cbn_per_block', type=bool, default=False)
    parser.add_argument('--cbn_per_network', type=bool, default=False)
    
    args = parser.parse_args()
    if args.num_evals == 0:
        args.num_evals = args.num_samples_eval / args.eval_batch_size
    
    print(args)
    return args


def get_logdir_name(flags):
    """Generates the name of the log directory from the values of flags
    Parameters
    ----------
        flags: neural net architecture generated by get_arguments()
    Outputs
    -------
        the name of the directory to store the training and evaluation results
    """
    
    param_list = ['batch_size', str(flags.train_batch_size), 'num_tasks', str(flags.num_tasks_per_batch), 'lr', str(flags.init_learning_rate), 'lr_anneal', flags.lr_anneal,
                  'opt', flags.optimizer,
                  'weight_decay', str(flags.weight_decay),
                  'nfilt', str(flags.num_filters), 'feature_extractor', str(flags.feature_extractor),
                  'enc_cl_link', flags.encoder_classifier_link]

    if flags.log_dir == '':
        logdir = './logs1/' + '-'.join(param_list)
    else:
        logdir = os.path.join(flags.log_dir, '-'.join(param_list))

    if flags.exp_dir is not None:
        # Running a Borgy experiment
        logdir = flags.exp_dir

    return logdir


class ScaledVarianceRandomNormal(init_ops.Initializer):
    """Initializer that generates tensors with a normal distribution scaled as per https://arxiv.org/pdf/1502.01852.pdf.
    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      dtype: The data type. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, factor=1.0, seed=None, dtype=dtypes.float32):
        self.mean = mean
        self.factor = factor
        self.seed = seed
        self.dtype = dtypes.as_dtype(dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype

        if shape:
            n = float(shape[-1])
        else:
            n = 1.0
        for dim in shape[:-2]:
            n *= float(dim)

        self.stddev = np.sqrt(self.factor * 2.0 / n)
        return random_ops.random_normal(shape, self.mean, self.stddev,
                                        dtype, seed=self.seed)


def _get_scope(is_training, flags):
    normalizer_params = {
        'epsilon': 0.001,
        'momentum': .95,
        'trainable': is_training,
        'training': is_training,
    }
    conv2d_arg_scope = slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=ACTIVATION_MAP[flags.activation],
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params=normalizer_params,
        # padding='SAME',
        trainable=is_training,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay),
        weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
        biases_initializer=tf.constant_initializer(0.0)
    )
    return conv2d_arg_scope


def _get_scope_cbn(is_training, flags):
    normalizer_params = {
        'epsilon': 0.001,
        'momentum': .95,
        'trainable': is_training,
        'training': is_training,
        'center' : False,
        'scale' : False
    }
    conv2d_arg_scope = slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=ACTIVATION_MAP[flags.activation],
        normalizer_fn=tf.layers.batch_normalization,
        normalizer_params=normalizer_params,
        # padding='SAME',
        trainable=is_training,
        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay),
        weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor),
        biases_initializer=tf.constant_initializer(0.0)
    )
    return conv2d_arg_scope


def build_simple_conv_net(images, flags, is_training, reuse=None, scope=None):
    conv2d_arg_scope = _get_scope(is_training, flags)
    with conv2d_arg_scope:
        with tf.variable_scope(scope or 'feature_extractor', reuse=reuse):
            h=images
            for i in range(4):
                h = slim.conv2d(h, num_outputs=flags.num_filters, kernel_size=3, stride=1,
                                scope='conv' + str(i), padding='SAME',
                                weights_initializer=ScaledVarianceRandomNormal(factor=flags.weights_initializer_factor))
                h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='VALID', scope='max_pool' + str(i))

            if flags.embedding_pooled == True:
                kernel_size = h.shape.as_list()[-2]
                h = slim.avg_pool2d(h, kernel_size=kernel_size, scope='avg_pool')
            h = slim.flatten(h)
    return h


def leaky_relu(x, alpha=0.1, name=None):
    return tf.maximum(x, alpha * x, name=name)


def get_cbn_layer(h, beta, gamma):
    """

    :param h: input layer
    :param beta: additive conditional batch norm paraemeter
    :param gamma: multiplicative conditional batch norm parameter in the delta form
    :return: conditional batch norm in the form (gamma + 1.0) * h + beta
    """
    if beta is None or gamma is None:
        return h

    beta = tf.expand_dims(beta, axis=1)
    gamma = tf.expand_dims(gamma, axis=1)

    beta = tf.tile(beta, multiples=[1, h.shape.as_list()[0] // beta.shape.as_list()[0], 1])
    beta = tf.reshape(beta, [-1, beta.shape.as_list()[-1]])
    beta = tf.reshape(beta, [-1, 1, 1, beta.shape.as_list()[-1]])

    gamma = tf.tile(gamma, multiples=[1, h.shape.as_list()[0] // gamma.shape.as_list()[0], 1])
    gamma = tf.reshape(gamma, [-1, beta.shape.as_list()[-1]])
    gamma = tf.reshape(gamma, [-1, 1, 1, beta.shape.as_list()[-1]])

    h = (gamma + 1.0) * h + beta
    return h


def build_simple_res_net(images, flags, num_filters, beta=None, gamma=None, is_training=False, reuse=None, scope=None):
    conv2d_arg_scope = _get_scope(is_training, flags)
    activation_fn = ACTIVATION_MAP[flags.activation]
    with conv2d_arg_scope:
        with tf.variable_scope(scope or 'feature_extractor', reuse=reuse):
            h = images
            for i in range(len(num_filters)):
                # make shortcut
                shortcut = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=1, stride=1,
                                       activation_fn=None,
                                       scope='shortcut' + str(i), padding='SAME')

                for j in range(flags.num_units_in_block):
                    h = slim.conv2d(h, num_outputs=num_filters[i], kernel_size=3, stride=1,
                                    scope='conv' + str(i) + '_' + str(j), padding='SAME', activation_fn=None)
                    if beta is not None and gamma is not None:
                        with tf.variable_scope('conditional_batch_norm'+ str(i) + '_' + str(j), reuse=reuse):
                            h = get_cbn_layer(h, beta=beta[i,j], gamma=gamma[i,j])

                    if j < (flags.num_units_in_block - 1) :
                        h = activation_fn(h, name='activation_' + str(i) + '_' + str(j))
                h = h + shortcut

                h = activation_fn(h, name='activation_' + str(i) + '_' + str(flags.num_units_in_block - 1))
                if i < flags.num_max_pools:
                    h = slim.max_pool2d(h, kernel_size=2, stride=2, padding='SAME', scope='max_pool' + str(i))

            if flags.embedding_pooled == True:
                kernel_size = h.shape.as_list()[-2]
                h = slim.avg_pool2d(h, kernel_size=kernel_size, scope='avg_pool')
            h = slim.flatten(h)
    return h


def build_feature_extractor_graph(images, flags, num_filters, beta=None, gamma=None, is_training=False, scope='feature_extractor_task_encoder', reuse=None, is_64way=False):
    if flags.feature_extractor == 'simple_conv_net':
        h = build_simple_conv_net(images, flags=flags, is_training=is_training, reuse=reuse, scope=scope)
    elif flags.feature_extractor == 'simple_res_net':
        h = build_simple_res_net(images, flags=flags, num_filters=num_filters, beta=beta, gamma=gamma, is_training=is_training, reuse=reuse, scope=scope)
    else:
        h = None

    embedding_shape = h.get_shape().as_list()
    if is_training and is_64way is False:
        h = tf.reshape(h, shape=(flags.num_tasks_per_batch, embedding_shape[0] // flags.num_tasks_per_batch, -1),
                                      name='reshape_to_separate_tasks_generic_features')
    else:
        h = tf.reshape(h, shape=(1, embedding_shape[0], -1),
                                      name='reshape_to_separate_tasks_generic_features')
    return h


def build_task_encoder(embeddings, labels, flags, is_training, reuse=None, scope='class_encoder'):
    conv2d_arg_scope = _get_scope(is_training, flags)

    with conv2d_arg_scope:
        with tf.variable_scope(scope, reuse=reuse):
            embedding_shape = embeddings.get_shape().as_list()
            task_encoding = embeddings
            if is_training:
                task_encoding = tf.reshape(task_encoding, 
                                           shape=(flags.num_tasks_per_batch, flags.num_classes_train, flags.num_shots_train, -1),
                                           name='reshape_to_separate_tasks_task_encoding')
            else:
                task_encoding = tf.reshape(task_encoding, shape=(1, flags.num_classes_test, flags.num_shots_test, -1),
                                           name='reshape_to_separate_tasks_task_encoding')
            task_encoding = tf.reduce_mean(task_encoding, axis=2, keep_dims=False)
            return task_encoding


def build_std_normalized_head(features_generic, task_encoding, flags, is_training, scope='std_normalized_euc_head'):
    """
    Implements the head using feature normalization by std before the Euclidian distance
    :param features_generic:
    :param task_encoding:
    :param flags:
    :param is_training:
    :param reuse:
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):
        _, features_generic_var = tf.nn.moments(features_generic, axes=-1, keep_dims=True)
        _, task_encoding_var = tf.nn.moments(task_encoding, axes=-1, keep_dims=True)

        features_generic = tf.div(features_generic, 1e-6 + tf.sqrt(features_generic_var, name='feature_generic_std'))
        task_encoding = tf.div(task_encoding, 1e-6 + tf.sqrt(task_encoding_var, name='feature_generic_std'))

        if len(features_generic.get_shape().as_list()) == 2:
            features_generic = tf.expand_dims(features_generic, axis=0)
        if len(task_encoding.get_shape().as_list()) == 2:
            task_encoding = tf.expand_dims(task_encoding, axis=0)

        # i is the number of steps in the task_encoding sequence
        # j is the number of steps in the features_generic sequence
        j = task_encoding.get_shape().as_list()[1]
        i = features_generic.get_shape().as_list()[1]

        # tile to be able to produce weight matrix alpha in (i,j) space
        features_generic = tf.expand_dims(features_generic, axis=2)
        task_encoding = tf.expand_dims(task_encoding, axis=1)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        task_encoding_tile = tf.tile(task_encoding, (1, i, 1, 1))
        features_generic_tile = tf.tile(features_generic, (1, 1, j, 1))
        # implement equation (4)
        euclidian = -tf.norm(task_encoding_tile - features_generic_tile, name='neg_euclidian_distance', axis=-1)

        if is_training:
            euclidian = tf.reshape(euclidian, shape=(flags.num_tasks_per_batch * flags.train_batch_size, -1))
        else:
            euclidian_shape = euclidian.get_shape().as_list()
            euclidian = tf.reshape(euclidian, shape=(euclidian_shape[1], -1))

        return euclidian


def get_polynomial(input, flags, is_training, scope='polynomial_metric', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        output = 0.0
        for p in range(1, flags.polynomial_metric_order+1):
            if p == 1:
                init_val=-flags.metric_multiplier_init
            else:
                init_val=0.0
            weight = tf.Variable(init_val, trainable=(is_training and flags.metric_multiplier_trainable), name='power_weight'+str(p), dtype=tf.float32)
            tf.summary.scalar('power_weight'+str(p), weight)
            output = output + tf.multiply(weight, tf.pow(input, p))
    return output


def build_polynomial_head(features_generic, task_encoding, flags, is_training, scope='polynomial_head'):
    """
    Implements the head using feature normalization by std before the Euclidian distance
    :param features_generic:
    :param task_encoding:
    :param flags:
    :param is_training:
    :param reuse:
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):
        # features_generic_norm = tf.norm(features_generic, axis=-1, keep_dims=True)
        # task_encoding_norm = tf.norm(task_encoding, axis=-1, keep_dims=True)
        #
        # features_generic = tf.div(features_generic, 1e-6 + features_generic_norm, name='feature_generic_normalized')
        # task_encoding = tf.div(task_encoding, 1e-6 + task_encoding_norm, name='feature_generic_normalized')

        if len(features_generic.get_shape().as_list()) == 2:
            features_generic = tf.expand_dims(features_generic, axis=0)
        if len(task_encoding.get_shape().as_list()) == 2:
            task_encoding = tf.expand_dims(task_encoding, axis=0)

        # i is the number of steps in the task_encoding sequence
        # j is the number of steps in the features_generic sequence
        j = task_encoding.get_shape().as_list()[1]
        i = features_generic.get_shape().as_list()[1]

        # tile to be able to produce weight matrix alpha in (i,j) space
        features_generic = tf.expand_dims(features_generic, axis=2)
        task_encoding = tf.expand_dims(task_encoding, axis=1)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        task_encoding_tile = tf.tile(task_encoding, (1, i, 1, 1))
        features_generic_tile = tf.tile(features_generic, (1, 1, j, 1))
        # implement equation (4)
        euclidian = tf.norm(task_encoding_tile - features_generic_tile, name='euclidian_distance', axis=-1)

        polynomial_metric = get_polynomial(euclidian, flags, is_training=is_training)

        if is_training:
            polynomial_metric = tf.reshape(polynomial_metric, shape=(flags.num_tasks_per_batch * flags.train_batch_size, -1))
        else:
            polynomial_metric_shape = polynomial_metric.get_shape().as_list()
            polynomial_metric = tf.reshape(polynomial_metric, shape=(polynomial_metric_shape[1], -1))

        return polynomial_metric


def build_cosine_head(features_generic, task_encoding, flags, is_training, scope='cosine_head'):
    """
    Implements the head using feature normalization by std before the Euclidian distance
    :param features_generic:
    :param task_encoding:
    :param flags:
    :param is_training:
    :param reuse:
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):
        features_generic_norm = tf.norm(features_generic, axis=-1, keep_dims=True)
        task_encoding_norm = tf.norm(task_encoding, axis=-1, keep_dims=True)

        features_generic = tf.div(features_generic, 1e-6 + features_generic_norm, name='feature_generic_normalized')
        task_encoding = tf.div(task_encoding, 1e-6 + task_encoding_norm, name='feature_generic_normalized')

        if len(features_generic.get_shape().as_list()) == 2:
            features_generic = tf.expand_dims(features_generic, axis=0)
        if len(task_encoding.get_shape().as_list()) == 2:
            task_encoding = tf.expand_dims(task_encoding, axis=0)

        # i is the number of steps in the task_encoding sequence
        # j is the number of steps in the features_generic sequence
        j = task_encoding.get_shape().as_list()[1]
        i = features_generic.get_shape().as_list()[1]

        # tile to be able to produce weight matrix alpha in (i,j) space
        features_generic = tf.expand_dims(features_generic, axis=2)
        task_encoding = tf.expand_dims(task_encoding, axis=1)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        task_encoding_tile = tf.tile(task_encoding, (1, i, 1, 1))
        features_generic_tile = tf.tile(features_generic, (1, 1, j, 1))
        # implement equation (4)
        euclidian = tf.norm(task_encoding_tile - features_generic_tile, name='euclidian_distance', axis=-1)

        cosine_mult = tf.Variable(flags.metric_multiplier_init, trainable=(is_training and flags.metric_multiplier_trainable), name='cosine_mult', dtype=tf.float32)
        tf.summary.scalar('cosine_mult', cosine_mult)
        cosine = 0.0 - cosine_mult*tf.square(euclidian)


        if is_training:
            cosine = tf.reshape(cosine, shape=(flags.num_tasks_per_batch * flags.train_batch_size, -1))
        else:
            cosine_shape = cosine.get_shape().as_list()
            cosine = tf.reshape(cosine, shape=(cosine_shape[1], -1))

        return cosine


def build_prototypical_head(features_generic, task_encoding, flags, is_training, scope='prototypical_head'):
    """
    Implements the prototypical networks few-shot head
    :param features_generic:
    :param task_encoding:
    :param flags:
    :param is_training:
    :param reuse:
    :param scope:
    :return:
    """

    with tf.variable_scope(scope):

        if len(features_generic.get_shape().as_list()) == 2:
            features_generic = tf.expand_dims(features_generic, axis=0)
        if len(task_encoding.get_shape().as_list()) == 2:
            task_encoding = tf.expand_dims(task_encoding, axis=0)

        # i is the number of steps in the task_encoding sequence
        # j is the number of steps in the features_generic sequence
        j = task_encoding.get_shape().as_list()[1]
        i = features_generic.get_shape().as_list()[1]

        # tile to be able to produce weight matrix alpha in (i,j) space
        features_generic = tf.expand_dims(features_generic, axis=2)
        task_encoding = tf.expand_dims(task_encoding, axis=1)
        # features_generic changes over i and is constant over j
        # task_encoding changes over j and is constant over i
        task_encoding_tile = tf.tile(task_encoding, (1, i, 1, 1))
        features_generic_tile = tf.tile(features_generic, (1, 1, j, 1))
        # implement equation (4)
        euclidian = -tf.norm(task_encoding_tile-features_generic_tile, name='neg_euclidian_distance', axis=-1)

        if is_training:
            euclidian = tf.reshape(euclidian, shape=(flags.num_tasks_per_batch * flags.train_batch_size, -1))
        else:
            euclidian_shape = euclidian.get_shape().as_list()
            euclidian = tf.reshape(euclidian, shape=(euclidian_shape[1], -1))

        return euclidian


def placeholder_inputs(batch_size, image_size, scope):
    """
    :param batch_size:
    :return: placeholders for images and
    """
    with tf.variable_scope(scope):
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3), name='images')
        labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size), name='labels')
        return images_placeholder, labels_placeholder


def get_batch(data_set, images_placeholder, labels_placeholder, batch_size):
    """
    :param data_set:
    :param images_placeholder:
    :param labels_placeholder:
    :return:
    """
    images_feed, labels_feed = data_set.next_batch(batch_size)

    feed_dict = {
        images_placeholder: images_feed.astype(dtype=np.float32),
        labels_placeholder: labels_feed,
    }
    return feed_dict


def preprocess(images):
    # mean = tf.constant(np.asarray([127.5, 127.5, 127.5]).reshape([1, 1, 3]), dtype=tf.float32, name='image_mean')
    # std = tf.constant(np.asarray([127.5, 127.5, 127.5]).reshape([1, 1, 3]), dtype=tf.float32, name='image_std')
    # return tf.div(tf.subtract(images, mean), std)

    std = tf.constant(np.asarray([0.5, 0.5, 0.5]).reshape([1, 1, 3]), dtype=tf.float32, name='image_std')
    return tf.div(images, std)


def get_nearest_neighbour_acc(flags, embeddings, labels):
    num_correct = 0
    num_tot = 0
    for i in range(flags.num_cases_test):
        test_classes = np.random.choice(np.unique(labels), size=flags.num_classes_test, replace=False)
        train_idxs, test_idxs = get_few_shot_idxs(labels=labels, classes=test_classes, num_shots=flags.num_shots_test)
        # TODO: this is to fix the OOM error, this can be removed when embed() supports batch processing
        test_idxs = np.random.choice(test_idxs, size=100, replace=False)

        np_embedding_train = embeddings[train_idxs]
        # Using the np.std instead of np.linalg.norm improves results by around 1-1.5%
        np_embedding_train = np_embedding_train / np.std(np_embedding_train, axis=1, keepdims=True)
        # np_embedding_train = np_embedding_train / np.linalg.norm(np_embedding_train, axis=1, keepdims=True)
        labels_train = labels[train_idxs]

        np_embedding_test = embeddings[test_idxs]
        np_embedding_test = np_embedding_test / np.std(np_embedding_test, axis=1, keepdims=True)
        # np_embedding_test = np_embedding_test / np.linalg.norm(np_embedding_test, axis=1, keepdims=True)
        labels_test = labels[test_idxs]

        kdtree = KDTree(np_embedding_train)
        nns, nn_idxs = kdtree.query(np_embedding_test, k=1)
        labels_predicted = labels_train[nn_idxs]

        num_matches = sum(labels_predicted == labels_test)

        num_correct += num_matches
        num_tot += len(labels_predicted)

    # print("Accuracy: ", (100.0 * num_correct) / num_tot)
    return (100.0 * num_correct) / num_tot


def get_cbn_premultiplier(task_encoding, i, j, flags, is_training, reuse):
    """

    :param task_encoding:
    :param i:
    :param j:
    :param flags:
    :param is_training:
    :param reuse:
    :return:
    """
    beta_weight = tf.get_variable(name='beta_weight' + str(i) + str(j), dtype=tf.float32, initializer=0.0,
                                  trainable=is_training,
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay_cbn,
                                                                               scope='penalize_beta' + str(i) + str(j)))
    gamma_weight = tf.get_variable(name='gamma_weight' + str(i) + str(j), dtype=tf.float32, initializer=0.0,
                                   trainable=is_training,
                                   regularizer=tf.contrib.layers.l2_regularizer(scale=flags.weight_decay_cbn,
                                                                                scope='penalize_gamma' + str(i) + str(
                                                                                    j)))
    tf.summary.scalar('beta_weight' + str(i) + str(j), beta_weight)
    tf.summary.scalar('gamma_weight' + str(i) + str(j), gamma_weight)
    return beta_weight, gamma_weight


def get_cbn_gamma_beta_net(h, i, j, num_filters, flags, is_training, reuse):
    """

    :param h:
    :param i:
    :param j:
    :param flags:
    :param is_training:
    :param reuse:
    :return:
    """

    activation_fn = ACTIVATION_MAP[flags.activation]
    beta, gamma = h, h
    for l in range(flags.cbn_num_layers):
        beta_old, gamma_old = beta, gamma
        beta = slim.fully_connected(beta, num_outputs=num_filters,
                                    activation_fn=None, normalizer_fn=None, reuse=reuse,
                                    weights_initializer=ScaledVarianceRandomNormal(factor=0.1),
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(
                                        scale=flags.weight_decay),
                                    scope='projection_beta' + str(i) + str(j) + str(l), trainable=is_training)
        gamma = slim.fully_connected(gamma, num_outputs=num_filters,
                                     activation_fn=None, normalizer_fn=None, reuse=reuse,
                                     weights_initializer=ScaledVarianceRandomNormal(factor=0.1),
                                     weights_regularizer=tf.contrib.layers.l2_regularizer(
                                         scale=flags.weight_decay),
                                     scope='projection_gamma' + str(i) + str(j) + str(l), trainable=is_training)
        if l > 0:
            beta = tf.add(beta, beta_old, name='shortcut_beta' + str(i) + str(j) + str(l))
            gamma = tf.add(gamma, gamma_old, name='shortcut_gamma' + str(i) + str(j) + str(l))
        if l < flags.cbn_num_layers - 1:
            beta = activation_fn(beta, name='activate_beta' + str(i) + str(j) + str(l))
            gamma = activation_fn(gamma, name='activate_gamma' + str(i) + str(j) + str(l))

    beta_weight, gamma_weight = get_cbn_premultiplier(h, i, j, flags, is_training, reuse)

    beta = tf.multiply(beta, beta_weight, name='premultiply_cbn_beta' + str(i) + str(j))
    gamma = tf.multiply(gamma, gamma_weight, name='premultiply_cbn_gamma' + str(i) + str(j))
    return beta, gamma


def get_cbn_params(features_task_encode, num_filters_list, flags, reuse=False, is_training=False, scope='cbn_params_raw'):
    """

    :param features_task_encode:
    :param num_filters_list:
    :param flags:
    :param reuse:
    :param is_training:
    :param scope:
    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):
        h = tf.reduce_mean(features_task_encode, axis=1, keep_dims=False)
        beta_reshape = [[None] * flags.num_units_in_block for _ in range(len(num_filters_list))]
        gamma_reshape = [[None] * flags.num_units_in_block for _ in range(len(num_filters_list))]
        for i, num_filters in enumerate(num_filters_list):
            for j in range(flags.num_units_in_block):
                if flags.cbn_per_block and j < (flags.num_units_in_block-1):
                    beta, gamma = None, None
                elif flags.cbn_per_network and (j < (flags.num_units_in_block-1) or i < (len(num_filters_list)-1)):
                    beta, gamma = None, None
                else:
                    beta, gamma = get_cbn_gamma_beta_net(h, i, j, num_filters=num_filters, flags=flags, is_training=is_training, reuse=reuse)
                beta_reshape[i][j] = beta
                gamma_reshape[i][j] = gamma
    return np.asarray(gamma_reshape), np.asarray(beta_reshape)


def build_inference_graph(images_deploy_pl, images_task_encode_pl, labels_task_encode_pl, flags, is_training):
    num_filters = [round(flags.num_filters * pow(flags.block_size_growth, i)) for i in range(flags.num_blocks)]
    reuse = False

    with tf.variable_scope('Model'):
        feature_extractor_encoding_scope = 'feature_extractor_encoder'

        features_task_encode = build_feature_extractor_graph(images=images_task_encode_pl, flags=flags,
                                                             is_training=is_training,
                                                             num_filters=num_filters,
                                                             scope=feature_extractor_encoding_scope,
                                                             reuse=flags.feat_extract_pretrain is not None)
        
        if flags.encoder_classifier_link=='prototypical':
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                               flags=flags, is_training=is_training, reuse=reuse)
            features_generic = build_feature_extractor_graph(images=images_deploy_pl, flags=flags,
                                                             is_training=is_training,
                                                             scope=feature_extractor_encoding_scope,
                                                             num_filters=num_filters,
                                                             reuse=True)
            logits = build_prototypical_head(features_generic, task_encoding, flags, is_training=is_training)
        elif flags.encoder_classifier_link=='cosine':
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                               flags=flags, is_training=is_training, reuse=reuse)
            features_generic = build_feature_extractor_graph(images=images_deploy_pl, flags=flags,
                                                             is_training=is_training,
                                                             scope=feature_extractor_encoding_scope,
                                                             num_filters=num_filters,
                                                             reuse=True)
            logits = build_cosine_head(features_generic, task_encoding, flags, is_training=is_training)
        elif flags.encoder_classifier_link=='polynomial':
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                               flags=flags, is_training=is_training, reuse=reuse)
            features_generic = build_feature_extractor_graph(images=images_deploy_pl, flags=flags,
                                                             is_training=is_training,
                                                             scope=feature_extractor_encoding_scope,
                                                             num_filters=num_filters,
                                                             reuse=True)
            logits = build_polynomial_head(features_generic, task_encoding, flags, is_training=is_training)
        elif flags.encoder_classifier_link=='std_normalized_euc_head':
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                               flags=flags, is_training=is_training, reuse=reuse)
            features_generic = build_feature_extractor_graph(images=images_deploy_pl, flags=flags,
                                                             is_training=is_training,
                                                             scope=feature_extractor_encoding_scope,
                                                             num_filters=num_filters,
                                                             reuse=True)
            logits = build_std_normalized_head(features_generic, task_encoding, flags, is_training=is_training)
        elif flags.encoder_classifier_link == 'cbn':
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                               flags=flags,
                                               is_training=is_training, reuse=reuse)
            # gamma, beta = None,None
            gamma, beta = get_cbn_params(features_task_encode=task_encoding, num_filters_list=num_filters,
                                         is_training=is_training, flags=flags, reuse=reuse)

            features_task_encode = build_feature_extractor_graph(images=images_task_encode_pl, flags=flags,
                                                                 is_training=is_training,
                                                                 num_filters=num_filters,
                                                                 gamma=gamma, beta=beta,
                                                                 scope=feature_extractor_encoding_scope,
                                                                 reuse=True)
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                               flags=flags, is_training=is_training, reuse=reuse)

            features_generic = build_feature_extractor_graph(images=images_deploy_pl, flags=flags,
                                                             is_training=is_training,
                                                             num_filters=num_filters,
                                                             gamma=gamma, beta=beta,
                                                             scope=feature_extractor_encoding_scope,
                                                             reuse=True)
            logits = build_polynomial_head(features_generic, task_encoding, flags, is_training=is_training)
        elif flags.encoder_classifier_link == 'cbn_cos':
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                               flags=flags,
                                               is_training=is_training, reuse=reuse)
            # gamma, beta = None,None
            gamma, beta = get_cbn_params(features_task_encode=task_encoding, num_filters_list=num_filters,
                                         is_training=is_training, flags=flags, reuse=reuse)

            features_task_encode = build_feature_extractor_graph(images=images_task_encode_pl, flags=flags,
                                                                 is_training=is_training,
                                                                 num_filters=num_filters,
                                                                 gamma=gamma, beta=beta,
                                                                 scope=feature_extractor_encoding_scope,
                                                                 reuse=True)
            task_encoding = build_task_encoder(embeddings=features_task_encode, labels=labels_task_encode_pl,
                                                      flags=flags, is_training=is_training, reuse=reuse)

            features_generic = build_feature_extractor_graph(images=images_deploy_pl, flags=flags,
                                                             is_training=is_training,
                                                             num_filters=num_filters,
                                                             gamma=gamma, beta=beta,
                                                             scope=feature_extractor_encoding_scope,
                                                             reuse=True)
            logits = build_cosine_head(features_generic, task_encoding, flags, is_training=is_training)
        else:
            raise Exception('Option not implemented')

    return logits, features_task_encode, features_generic


def build_feat_extract_pretrain_graph(images, flags, is_training):
    num_filters = [round(flags.num_filters * pow(flags.block_size_growth, i)) for i in range(flags.num_blocks)]

    with tf.variable_scope('Model'):
        feature_extractor_encoding_scope = 'feature_extractor_encoder'
        features = build_feature_extractor_graph(images=images, flags=flags,
                                                             is_training=is_training,
                                                             num_filters=num_filters,
                                                             scope=feature_extractor_encoding_scope,
                                                             reuse=False, is_64way=True)
        embedding_shape = features.get_shape().as_list()
        features = tf.reshape(features, shape=(embedding_shape[0]*embedding_shape[1],-1))
        # Classification loss
        logits = slim.fully_connected(features, flags.num_classes_pretrain,
                                      activation_fn=None, normalizer_fn=None, reuse=False,
                                      scope='pretrain_logits', trainable=is_training)
    return logits


def get_train_datasets(flags):
    mini_imagenet = _load_mini_imagenet(data_dir=flags.data_dir, split='sources')
    few_shot_data_train = Dataset(mini_imagenet)
    pretrain_data_train, pretrain_data_test = None, None
    if flags.feat_extract_pretrain:
        train_idx = np.random.choice(range(len(mini_imagenet[0])), size=int(0.9 * len(mini_imagenet[0])),
                                     replace=False)
        test_idx = np.setxor1d(range(len(mini_imagenet[0])), train_idx)
        new_labels = mini_imagenet[1]
        for i, old_class in enumerate(set(mini_imagenet[1])):
            new_labels[mini_imagenet[1] == old_class] = i
        pretrain_data_train = Dataset((mini_imagenet[0][train_idx], new_labels[train_idx]))
        pretrain_data_test = Dataset((mini_imagenet[0][test_idx], new_labels[test_idx]))
    return few_shot_data_train, pretrain_data_train, pretrain_data_test


def get_pwc_learning_rate(global_step, flags):
    learning_rate = tf.train.piecewise_constant(global_step, [np.int64(flags.number_of_steps / 2),
                                                              np.int64(flags.number_of_steps / 2 + flags.num_steps_decay_pwc),
                                                              np.int64(flags.number_of_steps / 2 + 2*flags.num_steps_decay_pwc)],
                                                [flags.init_learning_rate, flags.init_learning_rate * 0.1,
                                                 flags.init_learning_rate * 0.01,
                                                 flags.init_learning_rate * 0.001])
    return learning_rate


def train(flags):
    log_dir = get_logdir_name(flags)
    flags.pretrained_model_dir = log_dir
    log_dir = os.path.join(log_dir, 'train')
    # This is setting to run evaluation loop only once
    flags.max_number_of_evaluations = 1
    flags.eval_interval_secs = 0
    image_size=get_image_size(flags.data_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        global_step_pretrain = tf.Variable(0, trainable=False, name='global_step_pretrain', dtype=tf.int64)
        
        num_pretrain_steps = flags.feat_extract_pretrain_offset
        pretrain_lr_decay_step = int(float(flags.feat_extract_pretrain_offset) / flags.feat_extract_pretrain_lr_decay_n)
        if flags.feat_extract_pretrain == 'multitask' or flags.feat_extract_pretrain is None:
            num_pretrain_steps = 0
            pretrain_lr_decay_step = int(float(flags.number_of_steps) / flags.feat_extract_pretrain_lr_decay_n)

        images_deploy_pl, labels_deploy_pl = placeholder_inputs(batch_size=flags.num_tasks_per_batch*flags.train_batch_size,
                                                                image_size=image_size, scope='inputs/deploy')
        images_task_encode_pl, labels_task_encode_pl = placeholder_inputs(batch_size=flags.num_tasks_per_batch*flags.num_classes_train*flags.num_shots_train,
                                                                          image_size=image_size, scope='inputs/task_encode')

        # Auxiliary 64-way classification task oprations
        if flags.feat_extract_pretrain:
            pretrain_images_pl, pretrain_labels_pl = placeholder_inputs(batch_size=flags.pre_train_batch_size,
                                                                        image_size=image_size, scope='inputs/pretrain')
            pretrain_logits = build_feat_extract_pretrain_graph(pretrain_images_pl, flags, is_training=True)
            pretrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                           logits=pretrain_logits,
                                           labels=tf.one_hot(pretrain_labels_pl, flags.num_classes_pretrain)))
            pretrain_regu_losses = slim.losses.get_regularization_losses(scope='.*feature_extractor_encoder.*')

            pretrain_loss = tf.add_n([pretrain_loss] + pretrain_regu_losses)
            pretrain_misclass = 1.0 - slim.metrics.accuracy(tf.argmax(pretrain_logits, 1), pretrain_labels_pl)

            if flags.feat_extract_pretrain == 'multitask':
                pretrain_learning_rate = get_pwc_learning_rate(global_step, flags)
            else:
                pretrain_learning_rate = get_pwc_learning_rate(global_step_pretrain, flags)

            pretrain_optimizer = tf.train.MomentumOptimizer(learning_rate=pretrain_learning_rate, momentum=0.9,
                                                            name='PretrainOptimizer')

            pretrain_train_op = slim.learning.create_train_op(total_loss=pretrain_loss,
                                                              optimizer=pretrain_optimizer, global_step=global_step_pretrain,
                                                              clip_gradient_norm=flags.clip_gradient_norm)

            tf.summary.scalar('pretrain/loss', pretrain_loss)
            tf.summary.scalar('pretrain/misclassification', pretrain_misclass)
            tf.summary.scalar('pretrain/learning_rate', pretrain_learning_rate)
            # pretrain_summary = tf.summary.merge_all(scope='pretrain')
            # summaries = tf.get_collection('summaries', scope='(?!pretrain).*')
            # Merge only pretrain summaries
            pretrain_summary = tf.summary.merge(tf.get_collection('summaries', scope='pretrain'))

        # Primary task operations
        logits, _, _ = build_inference_graph(images_deploy_pl=images_deploy_pl, images_task_encode_pl=images_task_encode_pl,
                                       labels_task_encode_pl=labels_task_encode_pl,
                                       flags=flags, is_training=True)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(labels_deploy_pl, flags.num_classes_train)))

        # Losses and optimizer
        regu_losses = slim.losses.get_regularization_losses()
        loss = tf.add_n([loss] + regu_losses)
        misclass = 1.0 - slim.metrics.accuracy(tf.argmax(logits, 1), labels_deploy_pl)

        # Learning rate
        if flags.lr_anneal=='const':
            learning_rate = flags.init_learning_rate
        elif flags.lr_anneal=='pwc':
            learning_rate = get_pwc_learning_rate(global_step, flags)
        elif flags.lr_anneal=='exp':
            lr_decay_step = flags.number_of_steps // flags.n_lr_decay
            learning_rate = tf.train.exponential_decay(flags.init_learning_rate, global_step, lr_decay_step,
                                                       1.0/flags.lr_decay_rate, staircase=True)
        else:
            raise Exception('Not implemented')

        # Optimizer
        if flags.optimizer == 'sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        train_op = slim.learning.create_train_op(total_loss=loss, optimizer=optimizer, global_step=global_step,
                                                 clip_gradient_norm=flags.clip_gradient_norm)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('misclassification', misclass)
        tf.summary.scalar('learning_rate', learning_rate)
        # Merge all summaries except for pretrain
        summary = tf.summary.merge(tf.get_collection('summaries', scope='(?!pretrain).*'))
        
        # Get datasets
        few_shot_data_train, pretrain_data_train, pretrain_data_test = get_train_datasets(flags)
        # Define session and logging
        summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        supervisor = tf.train.Supervisor(logdir=log_dir, init_feed_dict=None,
                                         summary_op=None,
                                         init_op=tf.global_variables_initializer(),
                                         summary_writer=summary_writer,
                                         saver=saver,
                                         global_step=global_step, save_summaries_secs=flags.save_summaries_secs,
                                         save_model_secs=0) # flags.save_interval_secs

        with supervisor.managed_session() as sess:
            checkpoint_step = sess.run(global_step)
            if checkpoint_step > 0:
                checkpoint_step += 1
            if not flags.feat_extract_pretrain=='multitask':
                checkpoint_step = checkpoint_step + sess.run(global_step_pretrain)

            eval_interval_steps = flags.eval_interval_steps
            for step in range(checkpoint_step, flags.number_of_steps + num_pretrain_steps):
                # get batch of data to compute classification loss
                images_deploy, labels_deploy, images_task_encode, labels_task_encode = \
                    few_shot_data_train.next_few_shot_batch(deploy_batch_size=flags.train_batch_size,
                                                            num_classes_test=flags.num_classes_train,
                                                            num_shots=flags.num_shots_train,
                                                            num_tasks=flags.num_tasks_per_batch)
                if flags.augment:
                    images_deploy = image_augment(images_deploy)
                    images_task_encode = image_augment(images_task_encode)

                feed_dict = {images_deploy_pl: images_deploy.astype(dtype=np.float32), labels_deploy_pl: labels_deploy,
                             images_task_encode_pl: images_task_encode.astype(dtype=np.float32),
                             labels_task_encode_pl: labels_task_encode}

                # This is feature extractor 64-way classification task pretrain loop
                if flags.feat_extract_pretrain:
                    pretrain_images, pretrain_labels = pretrain_data_train.next_batch(flags.pre_train_batch_size)
                    if flags.augment:
                        pretrain_images = image_augment(pretrain_images)
                    pretrain_feed_dict = {pretrain_images_pl: pretrain_images.astype(dtype=np.float32),
                                          pretrain_labels_pl: pretrain_labels}
                    # TODO: this should not be necessary, but tf still says that I have to feed pretrain related placeholders
                    feed_dict.update(pretrain_feed_dict)

                if flags.feat_extract_pretrain and step < num_pretrain_steps or flags.feat_extract_pretrain=='multitask':
                    # Compute probability to select 64-way classification task
                    multitask_decay_steps = flags.number_of_steps // flags.feat_extract_pretrain_decay_n
                    multitask_proba = pow(flags.feat_extract_pretrain_decay_rate, step // multitask_decay_steps + 1)

                    t_train = time.time()
                    if np.random.uniform() < multitask_proba  + float(not flags.feat_extract_pretrain=='multitask'):
                        pretrain_loss = sess.run(pretrain_train_op, feed_dict=pretrain_feed_dict)
                    else:
                        pretrain_loss = np.nan
                    dt_train = time.time() - t_train

                    if step % 100 == 0:
                        pretrain_summary_str = sess.run(pretrain_summary, feed_dict=pretrain_feed_dict)
                        summary_writer.add_summary(pretrain_summary_str, step)
                        summary_writer.flush()
                        logging.info("step %d, pretrain loss : %.4g, dt: %.3gs" % (step, pretrain_loss, dt_train))

                    if step % 1000 == 0:
                        saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
                        eval_pretrain(flags, data_set_train=pretrain_data_train, data_set_test=pretrain_data_test)

                    if not flags.feat_extract_pretrain=='multitask':
                        continue
                        
                t_train = time.time()
                loss = sess.run(train_op, feed_dict=feed_dict)
                dt_train = time.time() - t_train

                if step % 100 == 0:
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    logging.info("step %d, loss : %.4g, dt: %.3gs" % (step, loss, dt_train))

                if float(step-num_pretrain_steps) / flags.number_of_steps > 0.5:
                    eval_interval_steps = flags.eval_interval_fine_steps

                if eval_interval_steps > 0 and step % eval_interval_steps == 0:
                    saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
                    eval(flags=flags)

                if float(step-num_pretrain_steps) > 0.5*flags.number_of_steps + flags.number_of_steps_to_early_stop:
                    break


def train_classifier(flags):
    log_dir = get_logdir_name(flags)
    flags.pretrained_model_dir = log_dir
    log_dir = os.path.join(log_dir, 'train')
    # This is setting to run evaluation loop only once
    flags.max_number_of_evaluations = 1
    flags.eval_interval_secs = 0
    image_size = get_image_size(flags.data_dir)

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)

        # 64-way classification task oprations
        pretrain_images_pl, pretrain_labels_pl = placeholder_inputs(batch_size=flags.pre_train_batch_size,
                                                                    image_size=image_size, scope='inputs/pretrain')
        pretrain_logits = build_feat_extract_pretrain_graph(pretrain_images_pl, flags, is_training=True)
        pretrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pretrain_logits,
            labels=tf.one_hot(pretrain_labels_pl, flags.num_classes_pretrain)))
        pretrain_regu_losses = slim.losses.get_regularization_losses(scope='.*feature_extractor_encoder.*')

        pretrain_loss = tf.add_n([pretrain_loss] + pretrain_regu_losses)
        pretrain_misclass = 1.0 - slim.metrics.accuracy(tf.argmax(pretrain_logits, 1), pretrain_labels_pl)

        pretrain_learning_rate = tf.train.piecewise_constant(global_step, [np.int64(flags.number_of_steps / 2),
                                                              np.int64(3*flags.number_of_steps / 4),
                                                              np.int64(7*flags.number_of_steps / 8)],
                                                [flags.init_learning_rate, flags.init_learning_rate * 0.1,
                                                 flags.init_learning_rate * 0.01,
                                                 flags.init_learning_rate * 0.001])

        pretrain_optimizer = tf.train.MomentumOptimizer(learning_rate=pretrain_learning_rate, momentum=0.9,
                                                        name='PretrainOptimizer')

        pretrain_train_op = slim.learning.create_train_op(total_loss=pretrain_loss,
                                                          optimizer=pretrain_optimizer,
                                                          global_step=global_step,
                                                          clip_gradient_norm=flags.clip_gradient_norm)

        tf.summary.scalar('pretrain/loss', pretrain_loss)
        tf.summary.scalar('pretrain/misclassification', pretrain_misclass)
        tf.summary.scalar('pretrain/learning_rate', pretrain_learning_rate)
        pretrain_summary = tf.summary.merge(tf.get_collection('summaries', scope='pretrain'))

        # Get datasets
        few_shot_data_train, pretrain_data_train, pretrain_data_test = get_train_datasets(flags)
        # Define session and logging
        summary_writer = tf.summary.FileWriter(log_dir, flush_secs=1)
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        supervisor = tf.train.Supervisor(logdir=log_dir, init_feed_dict=None,
                                         summary_op=None,
                                         init_op=tf.global_variables_initializer(),
                                         summary_writer=summary_writer,
                                         saver=saver,
                                         global_step=global_step, save_summaries_secs=flags.save_summaries_secs,
                                         save_model_secs=0)  # flags.save_interval_secs

        with supervisor.managed_session() as sess:
            checkpoint_step = sess.run(global_step)
            for step in range(checkpoint_step, flags.number_of_steps):
                pretrain_images, pretrain_labels = pretrain_data_train.next_batch(flags.pre_train_batch_size)
                feed_dict = {pretrain_images_pl: pretrain_images.astype(dtype=np.float32),
                                      pretrain_labels_pl: pretrain_labels}

                t_train = time.time()
                pretrain_loss = sess.run(pretrain_train_op, feed_dict=feed_dict)
                dt_train = time.time() - t_train

                if step % 100 == 0:
                    pretrain_summary_str = sess.run(pretrain_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(pretrain_summary_str, step)
                    summary_writer.flush()
                    logging.info("step %d, pretrain loss : %.4g, dt: %.3gs" % (step, pretrain_loss, dt_train))

                if step % 500 == 0:
                    saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
                    eval_pretrain(flags, data_set_train=pretrain_data_train, data_set_test=pretrain_data_test)


class PretrainModelLoader:
    def __init__(self, model_path, batch_size):
        self.batch_size = batch_size

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        step = int(os.path.basename(latest_checkpoint).split('-')[1])

        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        image_size = get_image_size(flags.data_dir)

        with tf.Graph().as_default():
            pretrain_images_pl, pretrain_labels_pl = placeholder_inputs(
                batch_size=batch_size, image_size=image_size, scope='inputs/pretrain')
            logits = build_feat_extract_pretrain_graph(pretrain_images_pl, flags, is_training=False)

            self.pretrain_images_pl = pretrain_images_pl
            self.pretrain_labels_pl = pretrain_labels_pl

            init_fn = slim.assign_from_checkpoint_fn(
                latest_checkpoint,
                slim.get_model_variables('Model'))

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # Run init before loading the weights
            self.sess.run(tf.global_variables_initializer())
            # Load weights
            init_fn(self.sess)

            self.flags = flags
            self.logits = logits
            self.logits_size = self.logits.get_shape().as_list()[-1]
            self.step = step

    def eval(self, data_set=None):

        num_batches = data_set.n_samples // self.batch_size
        num_correct = 0.0
        num_tot = 0.0

        for pretrain_images, pretrain_labels in data_set.sequential_batches(batch_size=self.batch_size, n_batches=num_batches):

            feed_dict = {self.pretrain_images_pl: pretrain_images.astype(dtype=np.float32),
                         self.pretrain_labels_pl: pretrain_labels}
            logits = self.sess.run(self.logits, feed_dict)
            labels_pred = np.argmax(logits, axis=-1)

            num_matches = sum(labels_pred == pretrain_labels)
            num_correct += num_matches
            num_tot += len(labels_pred)

        return num_correct / num_tot


class ModelLoader:
    def __init__(self, model_path, batch_size):
        self.batch_size = batch_size

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=os.path.join(model_path, 'train'))
        step = int(os.path.basename(latest_checkpoint).split('-')[1])
        
        flags = Namespace(load_and_save_params(default_params=dict(), exp_dir=model_path))
        image_size = get_image_size(flags.data_dir)
        
        with tf.Graph().as_default():
            images_deploy_pl, labels_deploy_pl = placeholder_inputs(batch_size=batch_size,
                                                                    image_size=image_size, scope='inputs/deploy')
            task_encode_batch_size = flags.num_classes_test * flags.num_shots_test
            images_task_encode_pl, labels_task_encode_pl = placeholder_inputs(batch_size=task_encode_batch_size,
                                                                              image_size=image_size, scope='inputs/task_encode')

            self.tensor_images_deploy = images_deploy_pl
            self.tensor_labels_task_encode = labels_task_encode_pl
            self.tensor_images_task_encode = images_task_encode_pl
            self.tensor_labels_deploy_pl = labels_deploy_pl

            # TODO: This is just used to create the variables of feature extractor in the case of pretrain
            # there might be a more elegant way to do it.
            if flags.feat_extract_pretrain:
                pretrain_images_pl, pretrain_labels_pl = placeholder_inputs(
                    batch_size=flags.pre_train_batch_size, image_size=image_size, scope='inputs/pretrain')
                pretrain_logits = build_feat_extract_pretrain_graph(pretrain_images_pl, flags, is_training=False)

            logits, features_sample, features_query = build_inference_graph(
                        images_deploy_pl=images_deploy_pl, images_task_encode_pl=images_task_encode_pl,
                        labels_task_encode_pl=labels_task_encode_pl, flags=flags, is_training=False)

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=tf.one_hot(labels_deploy_pl, flags.num_classes_test)))
            # Losses and optimizer
            regu_losses = slim.losses.get_regularization_losses()
            loss = tf.add_n([loss] + regu_losses)

            init_fn = slim.assign_from_checkpoint_fn(
                latest_checkpoint,
                slim.get_model_variables('Model'))

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            # Run init before loading the weights
            self.sess.run(tf.global_variables_initializer())
            # Load weights
            init_fn(self.sess)

            self.flags=flags
            self.logits = logits
            self.loss = loss
            self.features_sample = features_sample
            self.features_query = features_query
            self.logits_size = self.logits.get_shape().as_list()[-1]
            self.step = step

            log_dir = get_logdir_name(flags)
            graphpb_txt = str(tf.get_default_graph().as_graph_def())
            pathlib.Path(os.path.join(log_dir, 'eval')).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(log_dir, 'eval', 'graph.pbtxt'), 'w') as f:
                f.write(graphpb_txt)

    def eval(self, data_dir, num_cases_test, split='target_val'):
        data_set = Dataset(_load_mini_imagenet(data_dir=data_dir, split=split))

        num_batches = num_cases_test // self.batch_size
        num_correct = 0.0
        num_tot = 0.0
        loss_tot = 0.0
        for i in range(num_batches): # trange
            num_classes, num_shots = self.flags.num_classes_test, self.flags.num_shots_test
            
            images_deploy, labels_deploy, images_task_encode, labels_task_encode = \
                data_set.next_few_shot_batch(deploy_batch_size=self.batch_size,
                                             num_classes_test=num_classes, num_shots=num_shots,
                                             num_tasks=1)

            feed_dict = {self.tensor_images_deploy: images_deploy.astype(dtype=np.float32),
                         self.tensor_labels_deploy_pl: labels_deploy,
                         self.tensor_labels_task_encode: labels_task_encode,
                         self.tensor_images_task_encode: images_task_encode.astype(dtype=np.float32)}
            [logits, loss] = self.sess.run([self.logits, self.loss], feed_dict)
            labels_deploy_pred = np.argmax(logits, axis=-1)

            num_matches = sum(labels_deploy_pred == labels_deploy)
            num_correct += num_matches
            num_tot += len(labels_deploy_pred)
            loss_tot += loss

        return num_correct/num_tot, loss_tot/num_batches


def get_few_shot_idxs(labels, classes, num_shots):
    train_idxs, test_idxs = [], []
    idxs = np.arange(len(labels))
    for cl in classes:
        class_idxs = idxs[labels == cl]
        class_idxs_train = np.random.choice(class_idxs, size=num_shots, replace=False)
        class_idxs_test = np.setxor1d(class_idxs, class_idxs_train)

        train_idxs.extend(class_idxs_train)
        test_idxs.extend(class_idxs_test)

    assert set(class_idxs_train).isdisjoint(test_idxs)

    return np.array(train_idxs), np.array(test_idxs)


def test(flags):
    test_dataset = _load_mini_imagenet(data_dir=flags.data_dir, split='target_val')

    # test_dataset = _load_mini_imagenet(data_dir=flags.data_dir, split='sources')
    images = test_dataset[0]
    labels = test_dataset[1]
    
    embedding_model = ModelLoader(flags.pretrained_model_dir, batch_size=100)
    embeddings = embedding_model.embed(images=test_dataset[0])
    embedding_model = None
    print("Accuracy test raw embedding: ", get_nearest_neighbour_acc(flags, embeddings, labels))


def get_agg_misclassification(logits_dict, labels_dict):
    summary_ops = []
    update_ops = {}
    for key, logits in logits_dict.items():
        accuracy, update = slim.metrics.streaming_accuracy(tf.argmax(logits, 1), labels_dict[key])

        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(
            {'misclassification_' + key: (1.0 - accuracy, update)})

        for metric_name, metric_value in names_to_values.items():
            op = tf.summary.scalar(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        for update_name, update_op in names_to_updates.items():
            update_ops[update_name] = update_op
    return summary_ops, update_ops


def eval(flags):
    log_dir = get_logdir_name(flags)

    eval_writer = summary_writer(log_dir + '/eval')
    i=0
    last_step=-1
    while i < flags.max_number_of_evaluations:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir=flags.pretrained_model_dir)
        model_step = int(os.path.basename(latest_checkpoint or '0-0').split('-')[1])
        if last_step < model_step:
            results = {}
            model = ModelLoader(model_path=flags.pretrained_model_dir, batch_size=flags.eval_batch_size)
            
            acc_tst, loss_tst = model.eval(data_dir=flags.data_dir, num_cases_test=flags.num_cases_test, split='target_tst')
            acc_val, loss_val = model.eval(data_dir=flags.data_dir, num_cases_test=flags.num_cases_test, split='target_val')
            acc_trn, loss_trn = model.eval(data_dir=flags.data_dir, num_cases_test=flags.num_cases_test, split='sources')

            results["accuracy_target_tst"] = acc_tst
            results["accuracy_target_val"] = acc_val
            results["accuracy_sources"] = acc_trn

            results["loss_target_tst"] = loss_tst
            results["loss_target_val"] = loss_val
            results["loss_sources"] = loss_trn
            
            last_step = model.step
            eval_writer(model.step, **results)
            logging.info("accuracy_%s: %.3g, accuracy_%s: %.3g, accuracy_%s: %.3g."
                         %("target_tst", acc_tst, "target_val", acc_val, "sources", acc_trn))
        if flags.eval_interval_secs > 0:
            time.sleep(flags.eval_interval_secs)
        i=i+1


def eval_pretrain(flags, data_set_train, data_set_test):
    log_dir = get_logdir_name(flags)
    eval_writer = summary_writer(log_dir + '/eval')

    results = {}
    model = PretrainModelLoader(model_path=flags.pretrained_model_dir, batch_size=flags.eval_batch_size)

    acc_tst = model.eval(data_set=data_set_test)
    acc_trn = model.eval(data_set=data_set_train)

    results["pretrain/accuracy_test"] = acc_tst
    results["pretrain/accuracy_train"] = acc_trn

    eval_writer(model.step, **results)
    logging.info("pretrain_accuracy_%s: %.3g, pretrain_accuracy_%s: %.3g." % ("test", acc_tst, "train", acc_trn))


def image_augment(images):
    """

    :param images:
    :return:
    """
    pad_percent = 0.125
    flip_proba=0.5
    image_size = images.shape[1]
    pad_size=int(pad_percent*image_size)
    max_crop = 2*pad_size

    images_aug = np.pad(images, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0,0)), mode='constant')
    output = []
    for image in images_aug:
        if np.random.rand() < flip_proba:
            image = np.flip(image, axis=1)
        crop_val = np.random.randint(0, max_crop)
        image = image[crop_val:crop_val+image_size,crop_val:crop_val+image_size,:]
        output.append(image)
    return np.asarray(output)

    
def main(argv=None):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    print(os.getcwd())

    default_params = get_arguments()
    log_dir=get_logdir_name(flags=default_params)
    
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    # This makes sure that we can store a json and recove a namespace back
    flags = Namespace(load_and_save_params(vars(default_params), log_dir))
    
    if flags.mode == 'train':
        train(flags=flags)
    elif flags.mode == 'train_classifier':
        train_classifier(flags=flags)
    elif flags.mode == 'eval':
        eval(flags=flags)
    elif flags.mode == 'test':
        test(flags=flags)
        

if __name__ == '__main__':
    tf.app.run()