# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

"""
    Modified version to the original file to reduce RAM consumption.

    Here we make a simplification for the computation of shard size
        wiki_plus_bookcorpus_train_approx_sentences = 212000000
        wiki_plus_bookcorpus_test_approx_sentences = 1000000

    --- Jia Cheng
"""

import random
import math
from collections import defaultdict
from itertools import islice

from tqdm import tqdm
import multiprocessing
import statistics


class Sharding:
    def __init__(
        self, input_files, output_name_prefix, n_training_shards,
            n_test_shards, fraction_test_set, segmenter=None
    ):
        assert len(input_files) > 0, "The input file list must contain at least one file."
        assert n_training_shards > 0, "There must be at least one output shard."
        assert n_test_shards > 0, "There must be at least one output shard."

        self.n_training_shards = n_training_shards
        self.n_test_shards = n_test_shards
        self.fraction_test_set = fraction_test_set

        self.input_files = input_files

        self.output_name_prefix = output_name_prefix
        self.output_training_identifier = "training"
        self.output_test_identifier = "test"
        self.output_file_extension = ".txt"

        assert (segmenter is not None), " add segmenter in the modified TextSharing file "
        self.segmenter = segmenter

    # Remember, the input files contain one article per line (the whitespace check is to skip extraneous blank lines)
    def write_shards(self):
        print("Start: Loading Articles")

        # init working variables
        single_shard_sentences = []
        num_sentences_single_shard = 0
        single_shard = []  # not used.

        num_shard_train_idx = 0
        num_shard_test_idx = 0

        # hard-coded optimization for smaller RAM consumption
        # wiki+corpus estimate num sentences
        wiki_plus_bookcorpus_train_approx_sentences = 212000000
        wiki_plus_bookcorpus_test_approx_sentences = 1000000
        sentences_per_train_shard = math.ceil((wiki_plus_bookcorpus_train_approx_sentences - wiki_plus_bookcorpus_test_approx_sentences) / self.n_training_shards)
        sentences_per_test_shard = math.ceil(wiki_plus_bookcorpus_test_approx_sentences / self.n_test_shards)

        # flip a coin, and decide whether it's a test article or a train article,
        # add to test until it's max num sequences are filled up, and the remaining articles
        # pass them to the training
        where_to_put_next_shard = 'train'
        enough_test = False
        enough_train = False
        how_many_test = 0
        how_many_train = 0

        for input_file in self.input_files:
            print("input file:", input_file)
            with open(input_file, mode="r", newline="\n") as f:
                for line in f:
                    if line.strip():

                        if enough_test and enough_train:
                            print("enough test and enough train! Return!")
                            return

                        if where_to_put_next_shard is 'train':
                            num_sentences_per_shard = sentences_per_train_shard
                        else:
                            num_sentences_per_shard = sentences_per_test_shard

                        if num_sentences_single_shard >= num_sentences_per_shard:
                            # create new shard
                            num_sentences_single_shard = 0

                            # flip a coin, and decide whether it's a test article or a train article,
                            # add to test until it's max num sequences are filled up, and the remaining articles
                            # pass them to the training
                            if where_to_put_next_shard is 'train' and not enough_train:
                                num_shard_train_idx += 1
                                next_shard_train_file_path = self.output_name_prefix \
                                                             + self.output_training_identifier \
                                                             + str(num_shard_train_idx) + self.output_file_extension

                                self.write_single_shard(next_shard_train_file_path, single_shard_sentences)
                                print("train shard num: " + str(num_shard_train_idx))
                                how_many_train += 1
                                if how_many_train >= self.n_training_shards:
                                    enough_train = True
                                    where_to_put_next_shard = 'train'
                            if where_to_put_next_shard is 'test' and not enough_test:
                                num_shard_test_idx += 1
                                next_shard_test_file_path = self.output_name_prefix \
                                    + self.output_test_identifier \
                                    + str(num_shard_test_idx) + self.output_file_extension
                                self.write_single_shard(next_shard_test_file_path, single_shard_sentences)
                                print("test shard num: " + str(num_shard_test_idx))
                                how_many_test += 1
                                if how_many_test >= self.n_test_shards:
                                    enough_test = True
                                    where_to_put_next_shard = 'test'

                            if (not enough_test) or (not enough_train):
                                if (random.random() <= 0.4):
                                    where_to_put_next_shard = 'test'
                                else:
                                    where_to_put_next_shard = 'train'

                            # reset working variables
                            single_shard_sentences = []

                        else:
                            single_article_sentences = self.segment_one_article_into_sentences(line.rstrip())
                            single_shard_sentences.append(single_article_sentences)
                            num_sentences_single_shard += len(single_article_sentences)

                    del line

    def segment_one_article_into_sentences(self,
                                           single_article):
        sentences = None
        si = self.segmenter.segment_string(single_article)
        if si is not None:
            sentences = si
        return sentences

    def write_single_shard(self, shard_name, shard):
        with open(shard_name, mode="w", newline="\n") as f:
            # for article_id in shard: <- not used in this implementation
            for article in shard:
                for line in article:
                    f.write(line + "\n")
                f.write("\n")  # Line break between articles

try:
    import nltk

    nltk.download("punkt")
except ModuleNotFoundError or ImportError as e:
    print("nltk is required for sharding. please install before running.")


class NLTKSegmenter:
    def segment_string(self, article):
        try:
            t = nltk.tokenize.sent_tokenize(article)
            return t
        except IndexError:
            print(article[:100])
            return None

