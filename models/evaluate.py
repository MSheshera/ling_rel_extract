"""
Code to make predictions and evaluate.
"""
from __future__ import print_function
import argparse, os, sys
import math, time, json, codecs, pickle

from sklearn import metrics
import numpy as np

import utils
import model_utils as mu


def evaluate_preds(true_y, pred_y):
    """
    Helper to compute eval metrics.
    :param true_y: numpy array; (num_samples, )
    :param pred_y: numpy array; (num_samples, )
    :return: None.
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
    print()


def make_predictions(data, batcher, model, result_path,
                     batch_size, write_preds=False):
    """
    Make predictions on passed data in batches with the model and save if asked.
    :param model: pytorch model.
    :param data: dict{'X_train':, 'y_train':, 'X_dev':, 'y_dev':,
        'X_test':, 'y_test':}
    :param batcher: reference to model_utils.Batcher class.
    :param result_path: the path to which predictions should get written.
    :param batch_size: int; number of docs to consider in a batch.
    :param write: if predictions should be written.
    :return: Return in order truth and prediction.
        y_test_true, y_test_preds: numpy array; (num_samples, )
        y_dev_true, y_dev_preds: numpy array; (num_samples, )
        y_train_true, y_train_preds: numpy array; (num_samples, )
    """
    # Unpack data.
    X_train, y_train, X_dev, y_dev, X_test, y_test = \
        data['X_train'], data['y_train'], data['X_dev'], data['y_dev'], \
        data['X_test'], data['y_test']

    # Make predictions on the test, dev and train sets.
    y_test_true = np.array(y_test)
    start = time.time()
    y_test_preds = mu.batched_predict(model, batcher, batch_size, X_test,
                                      y_test)
    print('Test prediction time: {:.4f}s'.format(time.time()-start))

    start = time.time()
    y_dev_true = np.array(y_dev)
    y_dev_preds = mu.batched_predict(model, batcher, batch_size, X_dev, y_dev)
    print('Dev prediction time: {:.4f}s'.format(time.time() - start))

    start = time.time()
    y_train_true = np.array(y_train)
    y_train_preds = mu.batched_predict(model, batcher, batch_size, X_train,
                                       y_train)
    print('Train prediction time: {:.4f}s'.format(time.time() - start))
    if write_preds:
        test_preds_file = os.path.join(result_path, 'test_preds.npy')
        write_predictions(test_preds_file, y_test_true, y_test_preds)

        dev_preds_file = os.path.join(result_path, 'dev_preds.npy')
        write_predictions(dev_preds_file, y_dev_true, y_dev_preds)

        train_preds_file = os.path.join(result_path, 'train_preds.npy')
        write_predictions(train_preds_file, y_train_true, y_train_preds)

    return y_test_true, y_test_preds, y_dev_true, y_dev_preds, y_train_true, \
           y_train_preds


def write_predictions(pred_file, y_true, y_pred):
    with open(pred_file, 'w') as fp:
        both = np.vstack([y_true, y_pred]).T  # write np.array(samples, 2)
        np.save(fp, both)
    print('Wrote: {}'.format(pred_file))


def read_plot_cm(prediction_path):
    """
    Read the predictions and make a training and dev set confusion matrix.
    :param prediction_path: str.
    :return: None.
    """
    with open(os.path.join(prediction_path, 'dev_preds.npy')) as fp:
        dev_preds = np.load(fp)
    with open(os.path.join(prediction_path, 'train_preds.npy')) as fp:
        train_preds = np.load(fp)
    class_labels= ['institution', 'place_of_birth', 'place_of_death',
                   'date_of_birth', 'education-degree', 'no_relation']
    dev_cm = metrics.confusion_matrix(dev_preds[:, 0], dev_preds[:, 1])
    train_cm = metrics.confusion_matrix(train_preds[:, 0], train_preds[:, 1])
    utils.plot_confusion_matrix(dev_cm, class_labels, prediction_path,
                                'Dev confusion matrix')
    utils.plot_confusion_matrix(train_cm, class_labels, prediction_path,
                                'Train confusion matrix')


if __name__ == '__main__':
    if sys.argv[1] == 'plot_cm':
        read_plot_cm(prediction_path=sys.argv[2])
    else:
        sys.argv.write('Unknown argument.\n')