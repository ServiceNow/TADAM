# Copyright (c) 2018, ELEMENT AI. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
from common.util import Dataset
from collections import defaultdict


def _load_mini_imagenet(data_dir, split):
    """Load mini-imagenet from numpy's npz file format."""
    _split_tag = {'sources': 'train', 'target_val': 'val', 'target_tst': 'test'}[split]
    dataset_path = os.path.join(data_dir, 'few-shot-{}.npz'.format(_split_tag))
    logging.info("Loading mini-imagenet...")
    data = np.load(dataset_path)
    fields = data['features'], data['targets']
    logging.info("Done loading.")
    return fields


def make_all_datasets(data_dir, n_tasks_dict, n_samples_per_classes, n_classes_per_task=5):
    """High level function creating a dictionary of datasest for (sources, targets) x (train, test).

    Args: 
        data_dir: The directory containing the mini-imagnet dataset in npz format.
        n_tasks_dict: A dictionary with keys in ('sources', 'target_val', 'target_tst'), mapping to the number of 
          tasks required for that partition.
        n_samples_per_class: determines the number of samples each class will have in each task i.e. this imposes
          that each class have the same number of samples and the size of each tasks is:
          n_classes_per_task * n_samples_per_class
        n_classes_per_task: numer of classes per tasks.    

    Returns:
        A dictionary of multitask dataset, where keys are tuples of strings generated from the cartesian product of keys
        in n_tasks_dict and ('trn', 'tst'). e.g.: ('sources', 'trn'), ('target_val', 'tst') ...
    """
    dataset_dict = {}
    task_id_start = 0
    for split_name, task_size in n_tasks_dict.items():
        images, labels = _load_mini_imagenet(data_dir, split_name)
        task_ids = range(task_id_start, task_id_start + task_size)
        task_id_start += task_size
        trn_dataset, tst_dataset = make_multitask_dataset(
            images, labels, task_ids, n_samples_per_classes, n_classes_per_task)
        dataset_dict[(split_name, 'trn')] = trn_dataset
        dataset_dict[(split_name, 'tst')] = tst_dataset

    return dataset_dict


def make_multitask_dataset(images_db, labels, task_ids, n_samples_per_class, n_classes_per_task):
    """Make train and test datasets containing multiple tasks merged into a single dataset.

    Args:
        images_db: array of shape (n_total_samples, width, height, depth) containing the images in mini-imagenet
        labels: array of shape (n_total_samples,) representing original labels
        task_ids: array of pre-specified task_ids. The lenght determines the number of tasks created.
          the actual values don't matter as long as they are unique across all sources and target datasets.
        n_samples_per_class: determines the number of samples each class will have in each task i.e. this imposes
          that each class have the same number of samples and the size of each tasks is:
          n_classes_per_task * n_samples_per_class
        n_classes_per_task: numer of classes per tasks.
    """
    trn_fields = []
    tst_fields = []
    class_maps = {}

    for task_id in task_ids:
        trn_indices, tst_indices, class_map = make_task(labels, n_samples_per_class, n_classes_per_task)
        class_maps[task_id] = class_map
        trn_fields.append((trn_indices, class_map.to_new_class(labels[trn_indices]), [task_id] * len(trn_indices)))
        tst_fields.append((tst_indices, class_map.to_new_class(labels[tst_indices]), [task_id] * len(tst_indices)))

    def make_dataset(fields):
        return ImagesDataset(tuple(np.hstack(fields).astype(np.int32)), images_db, class_maps)

    return make_dataset(trn_fields), make_dataset(tst_fields)


def make_task(labels, n_samples_per_class, n_classes_per_task, rng=np.random):
    """Create a new task and make its train and test partition.
    
    Args:
        labels: 1d array of labels from the original classes of mini-imagenet
        n_samples_per_class: integer
        n_classes_per_task: integer
        rng: Random number generator
    
    Returns:
        trn_indices: 1d array of pointers in the mini-imagenet dataset, reserved for training
        tst_indices: 1d array of pointers in the mini-imagenet dataset, reserved for tests
    """
    unique_labels = np.unique(labels)
    classes = rng.choice(unique_labels, n_classes_per_task, replace=False)

    trn_indices = []
    tst_indices = []
    for c in classes:
        indices = np.random.permutation(np.flatnonzero(labels == c)).astype(np.int32)
        trn_indices.append(indices[:n_samples_per_class])
        tst_indices.append(indices[n_samples_per_class:])
    trn_indices = np.hstack(trn_indices)
    tst_indices = np.hstack(tst_indices)

    return trn_indices, tst_indices, ClassMap(classes)


def grouped_sampler(task_ids, n_task_per_batch):
    groups = defaultdict(list)

    for idx, task_id in enumerate(task_ids):
        groups[task_id].append(idx)

    for key, val in groups.items():
        groups[key] = np.array(val)

    unique_tasks = np.unique(task_ids)

    def sampler(n_samples, rng=np.random):
        n_samples_per_task = n_samples // n_task_per_batch
        assert n_samples_per_task * n_task_per_batch == n_samples
        active_task_ids = rng.choice(unique_tasks, n_task_per_batch)
        return np.hstack([rng.choice(groups[task_id], n_samples_per_task) for task_id in active_task_ids])

    return sampler


class ClassMap:

    def __init__(self, classes):
        """Simple class to help map back and forth the classes ids.
        
        Args:
            classes: array of classes such that classes[new_class] == original_class
        """
        self.classes = np.asarray(classes, dtype=np.int32)
        self.reverse_map = np.zeros(np.max(classes) + 1, dtype=np.int32) - 1  # not the most memory efficient but it doesn't matter
        for i, c in enumerate(classes):
            self.reverse_map[c] = i
        
    def to_new_class(self, original_classes):
        return self.reverse_map[original_classes]

    def to_original_class(self, new_classes):
        return self.classes[new_classes]


class ImagesDataset(Dataset):

    def __init__(self, fields, images_db, class_map):
        super().__init__(fields)
        self.images_db = images_db
        self.class_map = class_map

    def _from_ptr_to_images(self, indices, new_labels, task_ids):
        images = self.images_db[indices]
        return images, new_labels, task_ids

    def next_batch(self, n, rng=np.random):
        return self._from_ptr_to_images(*super().next_batch(n, rng))

    def sequential_batches(self, batch_size, n_batches, rng=np.random):
        for fields in super().sequential_batches(batch_size, n_batches, rng):
            yield self._from_ptr_to_images(*fields)

