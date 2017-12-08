"""
Miscellaneous utilities to used all over the place.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, random
import codecs
import json, pprint

import numpy as np
import torch

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def init_pretrained_glove(glove_path, word2idx, embedding_dim):
    """
    Initialize the store for embeddings with the pre-trained embeddings
    or the random word vectors.
    :param glove_path:
    :param word2idx:
    :param embedding_dim:
    :return:
    """
    vocab_size = len(word2idx)
    # read in the glove files
    glove_file = os.path.join(glove_path, 'glove.6B.200d.json')
    with open(glove_file, 'r') as fp:
        word2glove = json.load(fp)
    print('Read embeddings: {:s}'.format(glove_file))

    # then make giant matrix with all the matching vocab words
    padding_idx = 0
    # follow Karpahty's advice and initialize really small
    pretrained = torch.randn(vocab_size, embedding_dim) * 0.01
    count = 0
    for word, idx in word2idx.iteritems():
        # reserve the padding idx as 0
        if idx == padding_idx:
            torch.FloatTensor(embedding_dim).zero_()
        # keep as random initialization
        if word not in word2glove:
            continue
        pretrained[idx] = torch.FloatTensor(word2glove[word])

    embed = torch.nn.Embedding(vocab_size, embedding_dim)
    embed.weight = torch.nn.Parameter(pretrained)
    return embed


def pad_sort_data(int_mapped_docs, doc_labels):
    """
    Pad the data and sort such that the sentences are sorted in descending order
    of sentence length. Jumble all the par sentences in this sorting but also
    maintain a list which says which sentence came from which document of the
    same length as the total number of sentences with elements in
    [0, len(int_mapped_docs)]
    :return:
        X_dict: {'X': Torch Tensor; the padded and sorted-by-length sentence.
                'lengths': list(int); lengths of all sequences in X.
                'sorted_indices': list(int); rearranging X with this gives the
                    original unsorted order.
                'sorted_docrefs': Torch Tensor; ints saying which seq in X came
                    from which document. ints in range [0, len(int_mapped_docs)]
        sorted_refed_labels: labels for each seq in X.
    """
    assert(len(int_mapped_docs) == len(doc_labels))
    doc_ref = []
    refed_labels = []
    for ref, (int_mapped_sents, label) in enumerate(zip(int_mapped_docs,
                                                      doc_labels)):
        doc_ref.extend([ref]*len(int_mapped_sents))
        refed_labels.extend([label]*len(int_mapped_sents))

    # Make the list of list of lists a list of lists.
    int_mapped_sents = [val for sublist in int_mapped_docs for val in sublist]
    assert(len(refed_labels) == len(int_mapped_sents) == len(doc_ref))

    # Get sorted indices.
    sorted_indices = sorted(range(len(int_mapped_sents)),
                            key=lambda k: -len(int_mapped_sents[k]))
    max_length = len(int_mapped_sents[sorted_indices[0]])

    # Make the padded sequence.
    X_exp_padded = torch.LongTensor(len(int_mapped_sents), max_length).zero_()
    # Make the sentences into tensors sorted by length and place then into the
    # padded tensor.
    sorted_doc_ref = []
    sorted_refed_labels = []
    sorted_lengths = []
    for i, sent_i in enumerate(sorted_indices):
        tt = torch.LongTensor(int_mapped_sents[sent_i])
        lenght = tt.size(0)
        X_exp_padded[i, 0:lenght] = tt
        # Rearrange the doc refs and labels as well.
        sorted_doc_ref.append(doc_ref[sent_i])
        sorted_refed_labels.append(refed_labels[sent_i])
        # Make this because packedpadded seq asks for it.
        sorted_lengths.append(lenght)

    sorted_doc_ref = torch.LongTensor(sorted_doc_ref)
    sorted_refed_labels = torch.LongTensor(sorted_refed_labels)

    X_dict = {'X': X_exp_padded,  # Torch Tensor
              'lengths': sorted_lengths,  # list(int)
              'sorted_indices': sorted_indices,  # list(int)
              'sorted_docrefs': sorted_doc_ref}  # Torch Tensor.
    return X_dict, sorted_refed_labels


class Batcher():
    """
    Maintain batcher variables and state and such.
    :param full_X: the full dataset. the int-mapped preprocessed datset.
    :param full_y: the labels for the full dataset.
    :param batch_size: the number of documents to have in a batch; so sentence
        count varies.
    """
    def __init__(self, full_X, full_y, batch_size=None):
        self.full_len = len(full_X)
        self.batch_size = batch_size if batch_size != None else self.full_len
        assert(self.full_len == len(full_y))
        if self.full_len > self.batch_size:
            self.num_batches = int(np.ceil(float(self.full_len)/self.batch_size))
        else:
            self.num_batches = 1
        # Get random permutation of the indices but control for randomness.
        # https://stackoverflow.com/a/19307027/3262406
        rand_indices = range(self.full_len)
        random.shuffle(rand_indices)
        # Shuffle once when the class initialized and then keep it that way.
        self.full_X = [full_X[i] for i in rand_indices]
        self.full_y = [full_y[i] for i in rand_indices]
        # Get batch indices.
        self.batch_start = 0
        self.batch_end = batch_size

    def next_batch(self):
        """
        Return the next batch.
        :return:
        """
        for nb in xrange(self.num_batches):
            if self.batch_end < self.full_len:
                batch_X_raw = self.full_X[self.batch_start:self.batch_end]
                batch_y_raw = self.full_y[self.batch_start:self.batch_end]
            else:
                batch_X_raw = self.full_X[self.batch_start:]
                batch_y_raw = self.full_y[self.batch_start:]
            batch_X, batch_y = pad_sort_data(batch_X_raw, batch_y_raw)
            self.batch_start = self.batch_end
            self.batch_end += self.batch_size
            yield batch_X, batch_y

    def full_batch(self):
        """
        Return the full batch. This is called for the dev or the test
        set typically.
        :return:
        """
        full_X, full_y = pad_sort_data(self.full_X, self.full_y)
        return full_X, full_y


if __name__ == '__main__':
    if sys.argv[1] == 'test_pad_sort':
        int_mapped_path = sys.argv[2]
        dev_path = os.path.join(int_mapped_path, 'dev-im-small.json')
        with open(dev_path) as fp:
            X_dev, y_dev = json.load(fp, 'r')
        testX, testy = X_dev[5:9], y_dev[5:9]
        pad_sort_data(testX, testy)
    elif sys.argv[1] == 'test_batcher':
        int_mapped_path = sys.argv[2]
        dev_path = os.path.join(int_mapped_path, 'dev-im-small.json')
        with open(dev_path) as fp:
            X_dev, y_dev = json.load(fp, 'r')
        testX, testy = X_dev[:10], y_dev[:10]
        test_batcher = Batcher(testX, testy, 3)
        for X, y in test_batcher.next_batch():
            print(X['X'].size())
            print(y.size())
        X, y = test_batcher.full_batch()
    elif sys.argv[1] == 'test_glove_init':
        glove_path = sys.argv[2]
        int_mapped_path = sys.argv[3]
        map_path = os.path.join(int_mapped_path, 'word2idx-small.json')
        with open(map_path, 'r') as fp:
            word2idx = json.load(fp)
        embeds = init_pretrained_glove(glove_path=glove_path,
                                       word2idx=word2idx, embedding_dim=200)
        print(embeds.num_embeddings, embeds.embedding_dim, embeds.padding_idx)
    else:
        sys.argv.write('Unknown argument.\n')