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

"""Creates the clevr-shapenet few-shot learning dataset."""
import argparse
import os

import numpy as np
import pathlib
import pickle
from tqdm import tqdm

import json
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ClevrShapenet(Dataset):
    def __init__(self, path: Path, setname: str, size=224):
        self.setname = setname
        self.size = size

        path = Path(path)
        metadata_path = path / '1' / setname / 'scene'
        metadata = {}
        label2idx = {}
        for filename in metadata_path.iterdir():
            if not filename.suffix == '.json':
                print(".. skipping", filename)
                continue
            with open(filename, 'rt', encoding='utf8') as f:
                m = json.load(f)
                label = m['objects'][0]['shape'].split('/')[0]
                if label not in label2idx:
                    label2idx[label] = len(label2idx)
                metadata[m['image_index']] = dict(
                    filename=m['image_filename'],
                    label=label2idx[label])

        image_path = path / '1' / setname / 'images'
        self.data = [image_path / metadata[i]['filename'] for i in range(len(metadata))]
        self.labels = [metadata[i]['label'] for i in range(len(metadata))]

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.labels[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


def main(data_dir, output_dir):
    setnames = ["test", "train", "valid"]
    setname_map = {"test": "test", "train": "train", "valid": "val"}
    for setname in setnames:
        print("Loading original CLEVR-SHAPENET dataset, split %s, from %s" %(setname, data_dir))
        clevr_dataset = ClevrShapenet(path=data_dir, setname=setname, size=84)

        print("Generating image and label data")
        features=[]
        targets=[]
        for image, label in tqdm(clevr_dataset):
            features.append(np.expand_dims(np.array(image), 0))
            targets.append(label)

        features = np.concatenate(features)
        targets = np.array(targets)

        print("Saving few-shot data to disc", output_dir)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez(os.path.join(output_dir, 'few-shot-{}.npz'.format(setname_map[setname])), features=features, targets=targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="/mnt/scratch/serdyuk/data/clevr-shapenet/",
                        help='Path to the raw data')
    parser.add_argument('--output-dir', type=str, default="/mnt/scratch/serdyuk/data/clevr-shapenet/1/",
                        help='Output directory')

    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
    
