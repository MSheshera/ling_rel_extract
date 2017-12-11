"""
Add code for the baseline model; doc2vec with gensim + logistic regression.
Tutorials etc:
https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104
http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
Doc2vec hparams from: https://github.com/jhlau/doc2vec
"""
from __future__ import print_function
from __future__ import unicode_literals
import os, sys, argparse
import codecs
import time, random, copy

import numpy as np
from sklearn import linear_model
from sklearn import metrics
import gensim.models as gm

import utils, evaluate


class GRECIterator(object):
    def __init__(self, doc_list, doc_id_list):
        self.full_len = len(doc_list)
        self.doc_id_list = doc_id_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in zip(self.doc_id_list, self.doc_list):
              yield gm.doc2vec.TaggedDocument(doc, [idx])

    def return_perm(self):
        rand_indices = range(self.full_len)
        random.shuffle(rand_indices)
        # Shuffle and return.
        perm = [gm.doc2vec.TaggedDocument(self.doc_list[i], self.doc_id_list[i]) for i in rand_indices]
        return perm


def read_grec(vlstm_grec_path):
    splits = ['train', 'dev', 'test']
    docs_id = []
    docs = []
    docs_label = []
    docid = 0
    train_id, dev_id, test_id = 0,0,0
    for split in splits:
        split_file = os.path.join(vlstm_grec_path, split) + '.json'
        print('split: {:s} start docid: {:d}'.format(split, docid))
        train_id = docid if split == 'train' else train_id
        dev_id = docid if split == 'dev' else dev_id
        test_id = docid if split == 'test' else test_id
        with codecs.open(split_file, 'r', 'utf-8') as fp:
            for data_dict in utils.read_json(fp):
                tokenized_text = data_dict['text'].split()
                docs.append(tokenized_text)
                docs_id.append(docid)
                docs_label.append(data_dict['label'])
                docid += 1
                # if docid % 1000 == 0:
                #     break
    return docs_id, docs, docs_label, train_id, dev_id, test_id


def run_baseline(vlstm_grec_path, model_path):
    # read the data.
    docs_id, docs, docs_label, train_id, dev_id, test_id = \
        read_grec(vlstm_grec_path)

    # initialize model.
    # doc2vec parameters
    vector_size, window_size, min_count, sampling_threshold = 200, 15, 1, 1e-5
    negative_size, train_epoch, dm = 5, 20, 0
    worker_count = 1  # number of parallel processes

    doc_iterator = GRECIterator(doc_list=docs, doc_id_list=docs_id)

    print('Training Doc2Vec')
    start=time.time()
    model = gm.Doc2Vec(doc_iterator, size=vector_size, window=window_size,
                       min_count=min_count, sample=sampling_threshold,
                       workers=worker_count, hs=0, dm=dm,
                       negative=negative_size, dbow_words=1, dm_concat=1,
                       iter=train_epoch)
    print(model)
    print('Doc2Vec train time: {:.4f}'.format(time.time()-start))
    model.save(os.path.join(model_path, 'grec_lstm.d2v'))
    print()
    doc_vecs = model.docvecs[docs_id]

    # Form X, Y pairs.
    train_X, train_y  = doc_vecs[train_id:dev_id,:], np.array(docs_label[train_id:dev_id])
    dev_X, dev_y = doc_vecs[dev_id:test_id, :], np.array(docs_label[dev_id:test_id])
    test_X, test_y = doc_vecs[test_id:, :], np.array(docs_label[test_id:])
    print(train_X.shape, dev_X.shape, test_X.shape)
    # save vectors
    model_file = os.path.join(model_path, 'grec_slstm_d2v-train.npy'); np.save(model_file, train_X)
    print('Wrote: {:s}'.format(model_file))
    model_file = os.path.join(model_path, 'grec_slstm_d2v-dev.npy'); np.save(model_file, dev_X)
    print('Wrote: {:s}'.format(model_file))
    model_file = os.path.join(model_path, 'grec_slstm_d2v-test.npy'); np.save(model_file, test_X)
    print('Wrote: {:s}'.format(model_file))
    print()

    # Training logistic regression.
    print('Training LR')
    start = time.time()
    clf = linear_model.LogisticRegression(
        penalty='l2', dual=False, tol=1e-4, C=0.01, fit_intercept=True,
        intercept_scaling=1, solver='lbfgs', max_iter=100,
        multi_class='multinomial', verbose=10, warm_start=False, n_jobs=1)
    print(clf)
    clf.fit(train_X, train_y)
    print('LR train time: {:.4f}'.format(time.time() - start))
    print()

    # Make predictions and save.
    train_preds = clf.predict(train_X)
    dev_preds = clf.predict(dev_X)
    test_preds = clf.predict(test_X)

    test_preds_file = os.path.join(model_path, 'test_preds.npy')
    evaluate.write_predictions(test_preds_file, test_y, test_preds)

    dev_preds_file = os.path.join(model_path, 'dev_preds.npy')
    evaluate.write_predictions(dev_preds_file, dev_y, test_preds)

    train_preds_file = os.path.join(model_path, 'train_preds.npy')
    evaluate.write_predictions(train_preds_file, train_y, train_preds)

    # Evaluate.
    evaluate.evaluate_preds(train_y, train_preds)
    evaluate.evaluate_preds(dev_y, dev_preds)
    evaluate.evaluate_preds(test_y, test_preds)

    # Plot confusion matrices.
    class_labels = ['institution', 'place_of_birth', 'place_of_death',
                    'date_of_birth', 'education-degree', 'no_relation']
    dev_cm = metrics.confusion_matrix(dev_y, dev_preds)
    train_cm = metrics.confusion_matrix(train_y, train_preds)
    utils.plot_confusion_matrix(dev_cm, class_labels, model_path,
                                'Dev confusion matrix')
    utils.plot_confusion_matrix(train_cm, class_labels, model_path,
                                'Train confusion matrix')


def load_baseline(vlstm_grec_path, model_path):
    """
    Load trained doc2vec and make predictions.
    :param vlstm_grec_path:
    :param model_path:
    :return:
    """
    # read the data.
    docs_id, docs, docs_label, train_id, dev_id, test_id = \
        read_grec(vlstm_grec_path)
    # Read model.
    model = gm.Doc2Vec.load(os.path.join(model_path, 'grec_lstm.d2v'))
    doc_vecs = model.docvecs[docs_id]
    print('Doc vecs: {}'.format(doc_vecs.shape))

    # Form X, Y pairs.
    train_X, train_y = doc_vecs[train_id:dev_id, :], np.array(
        docs_label[train_id:dev_id])
    dev_X, dev_y = doc_vecs[dev_id:test_id, :], np.array(
        docs_label[dev_id:test_id])
    test_X, test_y = doc_vecs[test_id:, :], np.array(docs_label[test_id:])
    print(train_X.shape, dev_X.shape, test_X.shape)

    # Training logistic regression.
    print('Training LR')
    start = time.time()
    best_clf = None
    best_score = 0
    for cur_C in [100, 50, 10, 5, 1, 0.1, 0.01]:
        clf = linear_model.LogisticRegression(
            penalty='l2', dual=False, tol=1e-4, C=cur_C, fit_intercept=True,
            intercept_scaling=1, solver='lbfgs', max_iter=100,
            multi_class='multinomial', verbose=10, warm_start=False, n_jobs=1)
        print(clf)
        clf.fit(train_X, train_y)
        print('LR train time: {:.4f}'.format(time.time() - start))
        dev_preds = clf.predict(dev_X)
        f1_score = metrics.f1_score(dev_y, dev_preds, average='weighted')
        print('C: {:.4f}; f1: {:.4f}'.format(cur_C, f1_score))
        if f1_score > best_score:
            best_score = f1_score
            best_clf = copy.deepcopy(clf)
        print()
    clf = best_clf

    # Make predictions and save.
    train_preds = clf.predict(train_X)
    dev_preds = clf.predict(dev_X)
    test_preds = clf.predict(test_X)

    print('test')
    test_preds_file = os.path.join(model_path, 'test_preds_best.npy')
    evaluate.write_predictions(test_preds_file, test_y, test_preds)

    print('dev')
    dev_preds_file = os.path.join(model_path, 'dev_preds_best.npy')
    evaluate.write_predictions(dev_preds_file, dev_y, dev_preds)

    print('train')
    train_preds_file = os.path.join(model_path, 'train_preds_best.npy')
    evaluate.write_predictions(train_preds_file, train_y, train_preds)

    # Evaluate.
    evaluate.evaluate_preds(train_y, train_preds)
    evaluate.evaluate_preds(dev_y, dev_preds)
    evaluate.evaluate_preds(test_y, test_preds)

    # Plot confusion matrices.
    class_labels = ['institution', 'place_of_birth', 'place_of_death',
                    'date_of_birth', 'education-degree', 'no_relation']
    dev_cm = metrics.confusion_matrix(dev_y, dev_preds)
    train_cm = metrics.confusion_matrix(train_y, train_preds)
    utils.plot_confusion_matrix(dev_cm, class_labels, model_path,
                                'Dev confusion matrix-best')
    utils.plot_confusion_matrix(train_cm, class_labels, model_path,
                                'Train confusion matrix-best')
    print('best model')
    print(clf)

if __name__ == '__main__':
    #run_baseline(sys.argv[1], sys.argv[2])
    load_baseline(sys.argv[1], sys.argv[2])
