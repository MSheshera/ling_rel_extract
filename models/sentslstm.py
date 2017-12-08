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
    representations and make a prediction.
    """
    def __init__(self, word2idx, embedding_path, num_classes=6, max_batch_size=64,
                 num_layers=1, embedding_dim=200, hidden_dim=50, dropout=0.3,
                 cuda=False):
        super(SentsLSTM, self).__init__()  # need to inherit pytorch module
        self.max_batch_size = max_batch_size
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
        '''
        forward pass
        see https://discuss.pytorch.org/t/packedsequence-for-seq2seq-model/3907/2 for how to pack and unpack

        must pass in a batched group of X with padded X's and lenghts
        '''
        X, lengths, doc_refs = batch_X['X'], batch_X['lengths'], \
                               batch_X['sorted_docrefs']
        cur_batch_size = X.size(0)
        # Make initialized hidden and cell states.
        h0 = torch.zeros(self.num_layers, cur_batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, cur_batch_size, self.hidden_dim)
        # Make all model variables to Variables and move to the GPU.
        h0, c0 = Variable(h0), Variable(c0)
        X, batch_y = Variable(X), Variable(batch_y)
        if torch.cuda.is_available():
            h0, c0 = h0.cuda(), c0.cuda()
            X, batch_y = X.cuda(), batch_y.cuda()

        # Pass forward.
        embeds = self.word_embeddings(X)
        dropped_embeds = self.in_drop(embeds)
        packed = torch.nn.utils.rnn.pack_padded_sequence(dropped_embeds,
                                                         lengths,
                                                         batch_first=True)
        out, (hidden, cell) = self.lstm(packed, (h0, c0))
        dropped_hidden = self.h2p_drop(hidden)
        scores = self.hidden2pred(dropped_hidden)
        scores = scores.view(scores.size(1), scores.size(2))
        loss = self.criterion_ce(scores, batch_y)
        return loss

    def var_to_numpy(self, pred):
        # change to numpy
        if torch.cuda.is_available():
            pred = pred.cpu().data.numpy()
        else:
            pred = pred.data.numpy()
        return pred

    def predict(self, X):
        # HARD PRECITIONS!
        # run the forward pass of the network with learned parameters
        pp = self.forward(X)
        scores = F.log_softmax(pp)
        _, pred = torch.max(scores, 1)  # take argmax
        return self.var_to_numpy(pred)

    def soft_pred(self, X):
        # SOFT PREDICTIONS!
        # run the forward pass of the network with learned parameters
        pp = self.forward(X)
        scores = F.softmax(pp)
        pred = scores[:, 1]  # predicting the positive class!
        return self.var_to_numpy(pred)