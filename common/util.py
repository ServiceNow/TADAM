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

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import fnmatch


def variable_report(report_non_trainable=True):
    """Create a small report, showing the shapes of all trainable variables."""
    total_params = 0
    lines = ['Trainable Variables Report',
             '--------------------------']
    
    trainable_variables = tf.trainable_variables()

    for var in trainable_variables:
        shape = var.get_shape().as_list()
        num_params = np.prod(shape)
        total_params += num_params
        lines.append("shape: %15s, %5d, %s, %s"%(shape, num_params, var.name, var.dtype))
    lines.append("Total number of trainable parameters: %d"%total_params)

    if report_non_trainable:
        lines.extend(['','Non-Trainable Variables', '---------------------'])
        for var in tf.global_variables():
            if var in trainable_variables:
                continue
            shape = var.get_shape().as_list()
            num_params = np.prod(shape)
            lines.append("shape: %15s, %5d, %s, %s"%(shape, num_params, var.name, var.dtype))

    return '\n'.join(lines)


def variables_by_name(pattern, variable_list=None):
    if variable_list is None:
        variable_list = tf.global_variables()
    return [var for var in variable_list if fnmatch.fnmatch(var.name, pattern)]


def unique_variable_by_name(pattern, variable_list=None):
    var_list = variables_by_name(pattern, variable_list)
    if len(var_list) != 0:
        raise ValueError("Non unique variable. list = %s" % str(var_list) )
    return var_list[0]


def profiled_run(sess, ops, feed_dict, is_profiling=False, log_dir=None):
    if not is_profiling:
        return sess.run(ops, feed_dict=feed_dict)
    else:
        if log_dir is None:
            raise ValueError("You need to provide a log_dir for profiling.")
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        outputs = sess.run(ops, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(os.path.join(log_dir, 'timeline.json'), 'w') as f:
            f.write(ctf)

        return outputs


def summary_writer(log_dir):
    """Convenient wrapper for writing summaries."""
    writer = tf.summary.FileWriter(log_dir)

    def call(step, **value_dict):
        summary = tf.Summary()
        for tag, value in value_dict.items():
            summary.value.add(tag=tag, simple_value=value)
        writer.add_summary(summary, step)
        writer.flush()
    return call


def uniform(n):
    def sampler(n_samples, rng=np.random):
        return rng.choice(n, n_samples)
    return sampler


def categorical(probs):
    probs = np.asarray(probs)
    np.testing.assert_array_less(0, probs)
    cumsum = np.cumsum(probs)

    def sampler(n_samples, rng=np.random):
        return cumsum.searchsorted(rng.uniform(0, cumsum[-1], size=n_samples))
    return sampler


class Dataset(object):
    """Basic dataset interface."""

    def __init__(self, fields, fn=None, sampler=None):
        """Store a tuple of fields and access it through next_batch interface.

        By default, field[0] and field[1] are considered to be x and y. More fields can be 
        stored but they are unnamed.
        """
        self.fn = fn
        self.n_samples = len(fields[0])
        self.fields = fields
        if sampler is None:
            self.sampler = uniform(self.n_samples)
        else:
            self.sampler = sampler

    @property
    def x(self):
        return self.fields[0]

    @property
    def y(self):
        return self.fields[1]

    def next_batch(self, n, rng=np.random):
        idx = self.sampler(n, rng)
        return tuple(field[idx] for field in self.fields)

    def get_few_shot_idxs(self, labels, classes, num_shots):
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

    def next_few_shot_batch(self, deploy_batch_size, num_classes_test, num_shots, num_tasks):
        labels = self.y
        classes = np.unique(labels)

        deploy_images=[]
        deploy_labels=[]
        task_encode_images=[]
        task_encode_labels=[]
        for task in range(num_tasks):
            test_classes = np.random.choice(classes, size=num_classes_test, replace=False)

            task_encode_idxs, deploy_idxs = self.get_few_shot_idxs(labels, classes=test_classes, num_shots=num_shots)
            deploy_idxs = np.random.choice(deploy_idxs, size=deploy_batch_size, replace=False)

            labels_deploy = labels[deploy_idxs]
            labels_task_encode = labels[task_encode_idxs]

            class_map = {c:i for i,c in enumerate(test_classes)}
            class_map_fn = np.vectorize(lambda t: class_map[t])

            deploy_images.append(self.x[deploy_idxs])
            deploy_labels.append(class_map_fn(labels_deploy))
            task_encode_images.append(self.x[task_encode_idxs])
            task_encode_labels.append(class_map_fn(labels_task_encode))

        return np.concatenate(deploy_images, axis=0), np.concatenate(deploy_labels, axis=0), \
               np.concatenate(task_encode_images, axis=0), np.concatenate(task_encode_labels, axis=0)


    def next_triplet_batch_with_hard_negative(self, sess, triplet_logits_neg_mine, anchor_features_placeholder,
                                              positive_features_placeholder, negative_features_placeholder, batch_size, num_negatives_mining):
        """Generator for the triplet batches (anchor, positive, negative) based on the facenet paper"""
        labels = self.y
        classes = np.unique(labels)
        all_idxs = np.arange(self.n_samples)

        anchor_idxs = np.zeros(shape=(batch_size,), dtype=np.int32)
        positive_idxs = np.zeros(shape=(batch_size,), dtype=np.int32)
        negative_idxs_hard = np.zeros(shape=(batch_size,), dtype=np.int32)

        anchor_classes = np.random.choice(classes, size=batch_size, replace=True)
        for i in range(batch_size):
            pos_and_anchor_idxs = np.random.choice(all_idxs[labels == anchor_classes[i]], size=2, replace=False)

            anchor_idxs[i] = pos_and_anchor_idxs[0]
            positive_idxs[i] = pos_and_anchor_idxs[1]
            negative_idxs = np.random.choice(all_idxs[labels != anchor_classes[i]], size=num_negatives_mining, replace=False)

            anchor_features, positive_features, negative_features = self.x[anchor_idxs[i]], self.x[positive_idxs[i]], self.x[negative_idxs]

            feed_dict = {}
            feed_dict[anchor_features_placeholder] = np.expand_dims(anchor_features, axis=0)
            feed_dict[positive_features_placeholder] = np.expand_dims(positive_features, axis=0)
            feed_dict[negative_features_placeholder] = negative_features
            triplet_logits_neg_mine_np = sess.run(triplet_logits_neg_mine, feed_dict=feed_dict)

            negative_idxs_hard[i]=np.argmin(triplet_logits_neg_mine_np)

        return self.x[anchor_idxs], self.x[positive_idxs], self.x[negative_idxs_hard]

    def next_triplet_batch(self, batch_size):
        """Generator for the triplet batches (anchor, positive, negative) based on the facenet paper"""
        labels = self.y
        classes = np.unique(labels)
        all_idxs = np.arange(self.n_samples)
        anchor_idxs = np.zeros(shape=(batch_size,), dtype=np.int32)
        positive_idxs = np.zeros(shape=(batch_size,), dtype=np.int32)
        negative_idxs = np.zeros(shape=(batch_size,), dtype=np.int32)

        chosen_classes = np.random.choice(classes, size=2, replace=False)
        pos_and_anchor_idxs = np.random.choice(all_idxs[labels == chosen_classes[0]], size=2, replace=False)
        for i in range(batch_size):
            # chosen_classes = np.random.choice(classes, size=2, replace=False)
            # pos_and_anchor_idxs = np.random.choice(all_idxs[labels == chosen_classes[0]], size=2, replace=False)

            anchor_idxs[i] = pos_and_anchor_idxs[0]
            positive_idxs[i] = pos_and_anchor_idxs[1]
            negative_idxs[i] = np.random.choice(all_idxs[labels != chosen_classes[0]], size=1, replace=False)

        images = self.x
        return images[anchor_idxs], images[positive_idxs], images[negative_idxs]

    def sequential_batches(self, batch_size, n_batches, rng=np.random):
        """Generator for a random sequence of minibatches with no overlap."""
        permutation = rng.permutation(self.n_samples)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = np.minimum((start + batch_size), self.n_samples)
            idx = permutation[start:end]
            yield tuple(field[idx] for field in self.fields)
            if end == self.n_samples:
                break


class Bunch:
    def __init__(self, **kwargs):
        self.__init__ = kwargs


ACTIVATION_MAP = {"relu": tf.nn.relu,
                 "selu": tf.nn.selu,
                 "swish-1": lambda x, name='swish-1': tf.multiply(x, tf.nn.sigmoid(x), name=name),
                 }