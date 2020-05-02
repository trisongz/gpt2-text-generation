# -*- coding: utf-8 -*-
import pandas as pd
import json
from test_tube import HyperOptArgumentParser
from torchnlp.datasets.dataset import Dataset


def collate_lists(text: list) -> dict:
    """ Converts each line into a dictionary. """
    collated_dataset = []
    for i in range(len(text)):
        collated_dataset.append({"text": str(text[i])})
    return collated_dataset

def collate_dicts(text: list, label_text: list) -> dict:
    """ Converts a pair of label + text into a dictionary. """
    collated_dataset = []
    for i in range(len(text)):
        joined_text = "<LABEL> " + str(label_text[i]) + "<TARGET> " + str(text[i])
        collated_dataset.append({"text": joined_text})
    return collated_dataset


def collate_jsonl(items, labels):
    """ Converts a dict of items + list of labels into a dictionary. """
    collated_dataset = []
    for i, item in enumerate(items):
        if len(labels) == 1:
            joined_text = item[labels[0]]
        
        elif len(labels) == 2:
            joined_text = "<LABEL> " + item[labels[1]] + "<TARGET> " + item[labels[0]]
        
        collated_dataset.append({"text": joined_text})
    return collated_dataset

def text_dataset(
    hparams: HyperOptArgumentParser, train=True, val=True, test=True
):
    """
    Loads the Dataset from the csv files passed to the parser.
    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.

    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """

    def load_dataset(path):
        if path.endswith('.jsonl'):
            labels = ['text']
            if hparams.label is not None:
                label_name = hparams.label
                labels.append(label_name)
            
            with open(path, 'r+') as f:
                items = [json.loads(l) for i, l in enumerate(f)]
            
            return Dataset(collate_jsonl(items, labels))

        
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            text = list(df.text)
            if hparams.label is not None:
                label_name = hparams.label
                label_text = list(df[label_name])
                return Dataset(collate_dicts(text, label_text))

            else:
                return Dataset(collate_lists(text))

    func_out = []
    if train:
        func_out.append(load_dataset(hparams.train_fn))
    if val:
        func_out.append(load_dataset(hparams.dev_fn))
    if test:
        func_out.append(load_dataset(hparams.test_fn))

    return tuple(func_out)
