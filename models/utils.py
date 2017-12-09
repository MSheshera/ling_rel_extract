"""
General utilities; reading files and such.
"""
import os, sys
import json

import numpy as np
import torch

import sentslstm as slstm


def read_json(json_file):
    """
    Read per line JSON and yield.
    :param json_file: Just a open file. file-like with a next method.
    :return: yield one json object.
    """
    for json_line in json_file:
        # Try to manually skip bad chars.
        # https://stackoverflow.com/a/9295597/3262406
        try:
            f_dict = json.loads(json_line.replace('\r\n', '\\r\\n'),
                                encoding='utf-8')
            yield f_dict
        # Skip case which crazy escape characters.
        except ValueError:
            yield {}


def load_intmapped_data(int_mapped_path, use_toy):
    """
    Load the int mapped train, dev and test data from disk. Load a smaller
    dataset use_toy is set.
    :param int_mapped_path:
    :param use_toy:
    :return:
    """
    if use_toy:
        set_str = 'small'
    else:
        set_str = 'full'
    train_path = os.path.join(int_mapped_path,
                              'train-im-{:s}.json'.format(set_str))
    with open(train_path) as fp:
        X_train, y_train = json.load(fp)  # l of l of l, l
    dev_path = os.path.join(int_mapped_path, 'dev-im-{:s}.json'.format(set_str))
    with open(dev_path) as fp:
        X_dev, y_dev = json.load(fp)
    test_path = os.path.join(int_mapped_path, 'test-im-{:s}.json'.format(set_str))
    with open(test_path) as fp:
        X_test, y_test = json.load(fp)
    map_path = os.path.join(int_mapped_path,
                            'word2idx-{:s}.json'.format(set_str))
    with open(map_path) as fp:
        word2idx = json.load(fp)

    return X_train, y_train, X_dev, y_dev, X_test, y_test, word2idx


def restore_model(embedding_path, model_path, model_name, word2idx):
    # TODO: Store and retrieve hyperparameters. --high-pri.
    # Load model.
    model_file = os.path.join(model_path, model_name)
    model = slstm.SentsLSTM(word2idx, embedding_path, num_classes=6,
                            num_layers=1, embedding_dim=200, hidden_dim=50,
                            dropout=0.3)
    model.load_state_dict(torch.load(model_file))
    return model


def write_predictions(pred_file, y_true, y_pred):
    with open(pred_file, 'w') as fp:
        both = np.vstack([y_true, y_pred]).T  # write np.array(samples, 2)
        np.save(fp, both)
    print('Wrote: {}'.format(pred_file))