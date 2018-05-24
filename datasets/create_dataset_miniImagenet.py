"""Creates the mini-ImageNet dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import sys

import zipfile
import numpy as np
import scipy.misc

# Make train, validation and test splits deterministic from one run to another
np.random.seed(2017 + 5 + 17)


def main(data_dir, output_dir):
    for split in ('val', 'test', 'train'):
        # List of selected image files for the current split
        file_paths = []

        with open('{}.csv'.format(split), 'r') as csv_file:
            # Read the CSV file for that split, and get all classes present in
            # that split.
            reader = csv.DictReader(csv_file, delimiter=',')
            file_paths, labels = zip(
                *((os.path.join('images', row['filename']), row['label'])
                  for row in reader))
            all_labels = sorted(list(set(labels)))

        archive = zipfile.ZipFile(os.path.join(data_dir, 'images.zip'), 'r')

        # Processing loop over examples
        features, targets = [], []
        for i, (file_path, label) in enumerate(zip(file_paths, labels)):
            # Write progress to stdout
            sys.stdout.write(
                '\r>> Processing {} image {}/{}'.format(
                    split, i + 1, len(file_paths)))
            sys.stdout.flush()

            # Load image in RGB mode to ensure image.ndim == 3
            file_path = archive.open(file_path)
            image = scipy.misc.imread(file_path, mode='RGB')

            # Infer class from filename.
            label = all_labels.index(label)

            # Central square crop of size equal to the image's smallest side.
            height, width, channels = image.shape
            crop_size = min(height, width)
            start_height = (height // 2) - (crop_size // 2)
            start_width = (width // 2) - (crop_size // 2)
            image = image[
                start_height: start_height + crop_size,
                start_width: start_width + crop_size, :]

            # Resize image to 84 x 84.
            image = scipy.misc.imresize(image, (84, 84), interp='bilinear')

            features.append(image)
            targets.append(label)

        sys.stdout.write('\n')
        sys.stdout.flush()

        # Save dataset to disk
        features = np.stack(features, axis=0)
        targets = np.stack(targets, axis=0)
        permutation = np.random.permutation(len(features))
        features = features[permutation]
        targets = targets[permutation]
        np.savez(
            os.path.join(output_dir, 'few-shot-{}.npz'.format(split)),
            features=features, targets=targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', type=str,
        default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'mini-imagenet', 'raw-data'),
        help='Path to the raw data')
    parser.add_argument(
        '--output-dir', type=str, default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'mini-imagenet'),
        help='Output directory')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
