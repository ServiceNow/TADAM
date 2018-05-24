"""Creates the cifar100 few-shot learning dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import pathlib
import pickle

# Make train, validation and test splits deterministic from one run to another
np.random.seed(2017 + 5 + 17)

# Dataset split
# 00 'aquatic_mammals'
# 01 'fish'
# 02 'flowers'
# 03 'food_containers'
# 04 'fruit_and_vegetables'
# 05 'household_electrical_devices'
# 06 'household_furniture'
# 07 'insects'
# 08 'large_carnivores'
# 09 'large_man-made_outdoor_things'
# 10 'large_natural_outdoor_scenes'
# 11 'large_omnivores_and_herbivores'
# 12 'medium_mammals'
# 13 'non-insect_invertebrates'
# 14 'people'
# 15 'reptiles'
# 16 'small_mammals'
# 17 'trees'
# 18 'vehicles_1'
# 19 'vehicles_2'

# CIFAR100_PATH = '/mnt/datasets/public/cifar100'
# CIFAR100_PATH = '/home/boris/Downloads/cifar-100-python'
class_split = {'train': {1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19}, 'val': {8, 11, 13, 16}, 'test': {0, 7, 12, 14}}

def main(data_dir, output_dir):
    # load the full CFAR100 dataset, including train and test
    with open(os.path.join(data_dir, 'train'), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        images = dict[b'data']
        fine_labels = dict[b'fine_labels']
        coarse_labels = dict[b'coarse_labels']

    with open(os.path.join(data_dir, 'test'), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        images = np.concatenate((images, dict[b'data']))
        fine_labels = np.concatenate((fine_labels,dict[b'fine_labels']))
        coarse_labels = np.concatenate((coarse_labels,dict[b'coarse_labels']))

    images = images.reshape((-1, 3, 32, 32))
    images = images.transpose((0, 2, 3, 1))

    for split_name, split_coarse_classes in class_split.items():
        split_images=[]
        split_fine_labels=[]
        split_coarse_labels=[]
        for current_coarse_label in split_coarse_classes:
            idxs = coarse_labels == current_coarse_label
            split_images.append(images[idxs])
            split_fine_labels.append(fine_labels[idxs])
            split_coarse_labels.append(coarse_labels[idxs])

        split_images = np.concatenate(split_images)
        split_fine_labels = np.concatenate(split_fine_labels)
        split_coarse_labels = np.concatenate(split_coarse_labels)

        # Save dataset to disk
        permutation = np.random.permutation(len(split_images))
        features = split_images[permutation]
        targets = split_fine_labels[permutation]
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez(
            os.path.join(output_dir, 'mini-imagenet-{}.npz'.format(split_name)),
            features=features, targets=targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', type=str,
        default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'cifar100', 'raw-data'),
        help='Path to the raw data')
    parser.add_argument(
        '--output-dir', type=str, default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'cifar100', 'few-shot'),
        help='Output directory')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
