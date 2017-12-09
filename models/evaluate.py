"""
Code to make predictions and evaluate.
"""
import argparse, os, sys
import math, time, json, codecs, pickle

from sklearn import metrics
import numpy as np
import torch

import utils
import model_utils as mu
import sentslstm as slstm


def evaluate_preds(true_y, pred_y):
    """
    :return:
    """
    wf1 = metrics.f1_score(true_y, pred_y, average='weighted')
    wp = metrics.precision_score(true_y, pred_y, average='weighted')
    wr = metrics.recall_score(true_y, pred_y, average='weighted')
    ac = metrics.accuracy_score(true_y, pred_y)
    print('Weighted F1: {:.4f}'.format(wf1))
    print('Weighted precision: {:.4f}'.format(wp))
    print('Weighted recall: {:.4f}'.format(wr))
    print('Accuracy: {:.4f}'.format(ac))
    print(metrics.classification_report(y_true=true_y, y_pred=pred_y))


def make_predictions(data, batcher, model, model_name, result_path,
                     batch_size=None, write_predictions=False):
    """

    :param model: pytorch model.
    :param data: dict{'X_train':, 'y_train':, 'X_dev':, 'y_dev':}
    :param batcher: a model_utils.Batcher class.
    :param model_name: the name of the model from the trainer class.
    :param result_path: the path to which predictions should get written.
    :param batch_size: int; number of docs to consider in a batch.
    :param write_predictions: if predictions should be written.
    :return:
    """
    # Unpack data.
    X_train, y_train, X_dev, y_dev, X_test, y_test = \
        data['X_train'], data['y_train'], data['X_dev'], data['y_dev'], \
        data['X_test'], data['y_test']

    # Make predictions on the test and dev sets.
    test_batcher = batcher(full_X=X_test, full_y=y_test, shuffle=False)
    X_test_tt, y_test_true = test_batcher.full_batch()
    y_test_true = y_test_true.numpy()
    y_test_preds = model.predict(batch_X=X_test_tt)

    dev_batcher = batcher(full_X=X_dev, full_y=y_dev, shuffle=False)
    X_dev_tt, y_dev_true = dev_batcher.full_batch()
    y_dev_true = y_dev_true.numpy()
    y_dev_preds = model.predict(batch_X=X_dev_tt)

    # Make predictions on the train set in batches because its much bigger.
    train_batcher = batcher(full_X=X_train, full_y=y_train,
                            batch_size=batch_size, shuffle=False)
    y_train_true = np.array(y_train)
    y_train_preds = []
    for iter, (batch_X, batch_y) in enumerate(train_batcher.next_batch()):
        preds = model.predict(batch_X=batch_X)
        y_train_preds.append(preds)
    y_train_preds = np.hstack(y_train_preds)

    if write_predictions:
        test_preds_file = os.path.join(result_path, model_name + '_test_preds.npy')
        utils.write_predictions(test_preds_file, y_test_true, y_test_preds)

        dev_preds_file = os.path.join(result_path, model_name + '_dev_preds.npy')
        utils.write_predictions(dev_preds_file, y_dev_true, y_dev_preds)

        train_preds_file = os.path.join(result_path, model_name + '_train_preds.npy')
        utils.write_predictions(train_preds_file, y_train_true, y_train_preds)

    return y_test_true, y_test_preds, y_dev_true, y_dev_preds, y_train_true, \
           y_train_preds



