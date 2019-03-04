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
    exp_description = "clevr_proto_vs_tadam_5shot"

    params = dict(
        repeat=list(range(0, 3)),  # used to repeate the same experiment
        data_dir='/mnt/scratch/serdyuk/data/clevr-shapenet/1',
        metric=['film', 'prototypical'],
        feat_extract_pretrain=['multitask', None],
        number_of_steps=[120000],
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
        "-w", "/",
        "-e", "PYTHONPATH=%s" % repo_path,
        "-e", "DATA_PATH=/mnt/datasets/public/",
        "-v", "/mnt/datasets/public/:/mnt/datasets/public/",
        "-v", "/mnt/home/boris/:/mnt/home/boris/",
        "-v", "/mnt/scratch/:/mnt/scratch/",
        "--cpu=2",
        "--gpu=1",
        "--mem=16",
        "--restartable"
    ]

    cmd = os.path.join(repo_path, "model/tadam.py")

    gen_experiments_dir(params, exp_dir, exp_description, cmd, blocking=True, borgy_args=borgy_args)
