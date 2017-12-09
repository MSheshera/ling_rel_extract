"""
Define the model architecture and define forward and predict computations.
"""
from __future__ import print_function
import sys, os, math, time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import model_utils as mu


class SentsLSTM(torch.nn.Module):
    """
    Run an LSTM on each sentence of the passed paragraph, sum the sentence
    representations and make a prediction.s
    """
    def __init__(self, word2idx, embedding_path, num_classes=6, num_layers=1,
                 embedding_dim=200, hidden_dim=50, dropout=0.3):
        super(SentsLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # using to keep track of vocab (that comes from the training data)
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in self.word2idx.iteritems()}
        self.vocab_size = len(self.word2idx)

        self.in_drop = torch.nn.Dropout(p=dropout)
        self.word_embeddings = mu.init_pretrained_glove(embedding_path,
                                                        word2idx, embedding_dim)
        self.lstm = torch.nn.LSTM(self.embedding_dim, self.hidden_dim)
        # linear layer for softmax classification after final hidden layer
        self.h2p_drop = torch.nn.Dropout(p=dropout)
        self.hidden2pred = torch.nn.Linear(self.hidden_dim, self.num_classes)
        self.criterion_ce = torch.nn.CrossEntropyLoss(size_average=False)

        # Move model to the GPU.
        if torch.cuda.is_available():
            print('Running on GPU.')
            self.in_drop = self.in_drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            self.h2p_drop = self.h2p_drop.cuda()
            self.hidden2pred = self.hidden2pred.cuda()

    def objective(self, batch_X, batch_y):
        """
        Pass through a forward pass and return the loss.
        :param batch_X: dict of the form:
            {'X': Torch Tensor; the padded and sorted-by-length sentence.
            'lengths': list(int); lengths of all sequences in X.
            'sorted_indices': list(int); rearranging X with this gives the
                    original unsorted order.
            'sorted_docrefs': list(int); ints saying which seq in X came
                    from which document. ints in range [0, len(int_mapped_docs)]
            'num_docs': int; says how many docs there are.
        :param batch_y: torch Tensor; labels for each document in X. [shorter than X]
        :return: loss; torch Variable.
        """
        X, lengths, doc_refs, num_docs = batch_X['X'], batch_X['lengths'], \
                                         batch_X['sorted_docrefs'], \
                                         batch_X['num_docs']
        total_sents = X.size(0)
        # Make initialized hidden and cell states.
        h0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        # Make the doc masks.
        doc_refs = np.array(doc_refs)
        doc_masks = np.zeros((num_docs, total_sents, self.hidden_dim))
        for ref in xrange(num_docs):
            doc_masks[ref, doc_refs == ref, :] = 1
        doc_masks = torch.FloatTensor(doc_masks)

        # Make all model variables to Variables and move to the GPU.
        h0, c0 = Variable(h0), Variable(c0)
        X, batch_y = Variable(X), Variable(batch_y)
        doc_masks = Variable(doc_masks)
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
            X, batch_y = X.cuda(), batch_y.cuda()
            doc_masks = doc_masks.cuda()
        # Pass forward.
        embeds = self.word_embeddings(X)
        dropped_embeds = self.in_drop(embeds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(dropped_embeds,
                                                         lengths,
                                                         batch_first=True)
        out, (hidden, cell) = self.lstm(packed, (h0, c0))
        agg_hidden = torch.sum(hidden*doc_masks, dim=1)
        dropped_hidden = self.h2p_drop(agg_hidden)
        scores = self.hidden2pred(dropped_hidden)
        loss = self.criterion_ce(scores, batch_y)
        return loss

    def predict(self, batch_X):
        """
        Make a forward pass and return predictions as a numpy array.
        :param batch_X: dict of the form:
            {'X': Torch Tensor; the padded and sorted-by-length sentence.
            'lengths': list(int); lengths of all sequences in X.
            'sorted_indices': list(int); rearranging X with this gives the
                    original unsorted order.
            'sorted_docrefs': list(int); ints saying which seq in X came
                    from which document. ints in range [0, len(int_mapped_docs)]
            'num_docs': int; says how many docs there are.
        :return: preds: numpy array of size (num_docs, )
        """
        X, lengths, doc_refs, num_docs = batch_X['X'], batch_X['lengths'], \
                                         batch_X['sorted_docrefs'],\
                                         batch_X['num_docs']
        total_sents = X.size(0)
        # Make initialized hidden and cell states.
        h0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, total_sents, self.hidden_dim)
        # Make the doc masks.
        doc_refs = np.array(doc_refs)
        doc_masks = np.zeros((num_docs, total_sents, self.hidden_dim))
        for ref in xrange(num_docs):
            doc_masks[ref, doc_refs == ref, :] = 1
        doc_masks = torch.FloatTensor(doc_masks)

        # Make all model variables to Variables and move to the GPU.
        # Do pure inference, no caches stored.
        h0, c0 = Variable(h0, volatile=True), Variable(c0, volatile=True)
        X = Variable(X, volatile=True)
        doc_masks = Variable(doc_masks, volatile=True)
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
            X = X.cuda()
            doc_masks = doc_masks.cuda()
        # Pass forward. No dropout at test time.
        embeds = self.word_embeddings(X)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds,
                                                         lengths,
                                                         batch_first=True)
        out, (hidden, cell) = self.lstm(packed, (h0, c0))
        agg_hidden = torch.sum(hidden * doc_masks, dim=1)
        scores = self.hidden2pred(agg_hidden)
        max_scores, preds = torch.max(scores, dim=1)

        # Make numpy array and return.
        if torch.cuda.is_available():
            preds = preds.cpu().data.numpy()
        else:
            preds = preds.data.numpy()
        assert(preds.shape[0] == num_docs)
        return preds