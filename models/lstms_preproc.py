"""
Code to pre-process text data and embeddings to be fed to the model.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, argparse
import codecs, json
import time

import utils

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def make_word2glove_dict(glove_dir, embedding_dim):
    """
    Make a dict going from word to embeddings so its easy to read in later.
    :param glove_dir:
    :return:
    """
    word2glove = {}
    glove_file = os.path.join(glove_dir, 'glove.6B.{:d}d.txt'.format(embedding_dim))
    out_glove_file = os.path.join(glove_dir, 'glove.6B.{:d}d.json'.format(embedding_dim))
    if os.path.isfile(out_glove_file):
        sys.stderr.write('File exists; Look into it manually.\n')
        sys.exit(1)
    print('Processing: {:s}'.format(glove_file))
    with codecs.open(glove_file, 'r', 'utf-8') as fp:
        for i, line in enumerate(fp):
            ss = line.split()
            word = ss[0]
            if i % 10000 == 0:
                print('Processed: {:d} lines. Word: {:s}'.format(i, word))
            embeds = [float(x) for x in ss[1:]]
            word2glove[word] = embeds
    print('w2g map length: {:d}'.format(len(word2glove)))
    # Save giant map to a pickle file.
    with open(out_glove_file, 'w') as fp:
        json.dump(word2glove, fp)
    print('Wrote: {:s}'.format(out_glove_file))


def map_split_to_int(split_path, size_str, word2idx={}, update_map=True):
    """
    Convert text to set of int mapped tokens. Mapping words to integers at
    all times, train/dev/test.
    :param split_path:
    :param word2idx:
    :param update_map:
    :return:
    """
    X, y = [], []
    num_oovs, num_docs, num_sents = 0, 0, 0
    # reserve index 0 for 'padding' on the ends and index 1 for 'oov'
    if '<pad>' not in word2idx:
        word2idx['<pad>'] = 0
    if '<oov>' not in word2idx:
        # reserve
        word2idx['<oov>'] = 1

    start_time = time.time()
    with codecs.open(split_path, 'r', 'utf-8') as fp:
        print('Processing: {:s}'.format(split_path))
        for data_json in utils.read_json(fp):
            label = data_json['label']
            # Read the list of sbd sentences tokenized on white space.
            sents = data_json['text']
            num_docs += 1
            if num_docs % 100 == 0:
                print('Processing {:d}th document'.format(num_docs))
            # Make a small dataset if asked for.
            if size_str == 'small' and num_docs % 1000 == 0:
                break
            intmapped_sents = []  # list of lists.
            for sent in sents:
                num_sents += 1
                toks = sent.split()
                # Add start and stop states.
                toks = ['<start>'] + toks + ['<stop>']
                if update_map:
                    for tok in toks:
                        if tok not in word2idx:
                            word2idx[tok] = len(word2idx)
                # Map tokens to integers.
                intmapped_sent = []
                for tok in toks:
                    # This case cant happen for me because im updating the map
                    # for every split. But in case I set update_map to false
                    # this handles it.
                    intmapped_tok = word2idx.get(tok, word2idx['<oov>'])
                    intmapped_sent.append(intmapped_tok)
                    if intmapped_tok == 1:
                        num_oovs += 1
                intmapped_sents.append(intmapped_sent)
            X.append(intmapped_sents)    # list of lists of lists >_<
            y.append(label)
    assert(len(X) == len(y))
    print('Processed: num_documents: {:d}; vocab_size: {:d}; num_oovs: {:d}; '
          'sents_per_doc: {:.4f}'.format(num_docs, len(word2idx), num_oovs,
                                         float(num_sents)/num_docs))
    print('Took: {:4.4f}s'.format(time.time()-start_time))
    return X, y, word2idx


def make_int_maps(in_path, out_path, size_str):
    """
    For each split map all the tokens to integers and create token int maps.
    :param in_path:
    :param out_path:
    :param size_str: says if you want to make a small 1000 example set or full.
    :return:
    """
    splits = ['train', 'dev', 'test']
    word2idx = {}
    for split in splits:
        split_path = os.path.join(in_path, split) + '.json'
        X, y, word2idx = map_split_to_int(split_path, size_str=size_str,
                                          word2idx=word2idx, update_map=True)
        intmapped_out_path = os.path.join(out_path, split) + '-im-{:s}.json'.format(size_str)
        with codecs.open(intmapped_out_path, 'w', 'utf-8') as fp:
            json.dump((X, y), fp)
        print('Wrote: {:s}'.format(intmapped_out_path))
    # Write the map.
    intmap_out_path = os.path.join(out_path, 'word2idx-{:s}.json'.format(size_str))
    with codecs.open(intmap_out_path, 'w', 'utf-8') as fp:
        json.dump(word2idx, fp)
    print('Wrote: {:s}'.format(intmap_out_path))


def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')

    # Make the giant glove map.
    make_w2g = subparsers.add_parser(u'w2g_map')
    make_w2g.add_argument(u'-i', u'--glove_path',
                          required=True,
                          help=u'Path to the glove embeddings directory.')
    make_w2g.add_argument(u'-e', u'--embedding_dim',
                          required=True, type=int,
                          choices=[50, 100, 200, 300],
                          help=u'Path to the glove embeddings directory.')
    # Map sentences to list of int mapped tokens.
    make_int_map = subparsers.add_parser(u'int_map')
    make_int_map.add_argument(u'-i', u'--in_path', required=True,
                              help=u'Path to the processed train/dev/test '
                                   u'splits.')
    make_int_map.add_argument(u'-s', u'--size', required=True,
                              choices=['small', 'full'],
                              help=u'Make a small version with 1000 examples or'
                                   u'a large version with the whole dataset.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'w2g_map':
        make_word2glove_dict(glove_dir=cl_args.glove_path,
                             embedding_dim=cl_args.embedding_dim)
    if cl_args.subcommand == 'int_map':
        make_int_maps(in_path=cl_args.in_path, out_path=cl_args.in_path,
                      size_str=cl_args.size)


if __name__ == '__main__':
    main()