"""
Given the data with marked entity mentions make train, test, dev splits and
assign fixed labels to each example. Also contains code to take a split and
make it thw way Pats, Katies or my models need it.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, argparse
import codecs, json
import re, random, time
import collections

# Tokenization and stuff.
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize

# My imports.
import data_utils as du

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

# Class labels.
int_labels = {'institution': 0, 'place_of_birth': 1, 'place_of_death': 2,
              'date_of_birth': 3, 'education-degree': 4, 'no_relation': 5}


######################################
#       Form the sixth relation      #
######################################
def make_judgement(data_dict, rel_type):
    """
    Given the data for any example, make a binary judgement about whether the
    relation holds based on annotator judgements and augment the data_dict with
    a int mapped relation type.
    :param data_dict: dict; the read in JSON data; modified in place
    :param rel_type: string
    :return: holds: boolean; True if relation holds, else False.
    """
    judgement = collections.Counter([j['judgment'] for j in
                                     data_dict['judgments']])
    holds = True if judgement['yes'] >= 2 else False
    data_dict['int_mapped_rel'] = int_labels[rel_type] if holds else \
        int_labels['no_relation']
    return holds


def form_sixth_rel(dataset_path, out_path):
    """
    Make a judgement on each relation, form the 6th relation, write the 6
    relations to disk, and return the examples for each relation in rels_dict.
    :param dataset_path: unicode; full path to the dir with all rel jsons.
    :return: rels_dict: dict(list(data_json)); big dict with data for
        each relation.
    """
    rel_types = ['institution', 'place_of_birth', 'place_of_death',
                 'date_of_birth', 'education-degree', 'no_relation']
    rels_dict = {'institution': [], 'place_of_birth': [], 'place_of_death': [],
                 'date_of_birth': [], 'education-degree': [], 'no_relation': []}
    no_relation_count = 0
    for rel_type in rel_types[:-1]:
        fname = os.path.join(dataset_path, rel_type) + '.json'
        rel_fname = os.path.join(out_path, rel_type) + '.json'
        rel_file = codecs.open(rel_fname, u'w', u'utf-8')
        with codecs.open(fname, 'r', 'utf-8') as fp:
            rel_pos, rel_neg = 0, 0
            for data_json in du.read_json(fp):
                holds = make_judgement(data_json, rel_type)
                if holds:
                    rel_pos += 1
                    rels_dict[rel_type].append(data_json)
                else:
                    rel_neg += 1
                    rels_dict[rel_types[-1]].append(data_json)
            no_relation_count += rel_neg
            print('Relation: {:s};\t Yes: {:d};\t No: {:d}'.format(
                rel_type, rel_pos, rel_neg))
            # Write positive examples for current rel to disk.
            for data_dict in rels_dict[rel_type]:
                proc_jsons = json.dumps(data_dict, ensure_ascii=False)
                rel_file.write(proc_jsons + '\n')
        rel_file.close()
        print('Wrote: {:s}'.format(rel_fname))

    assert(len(rels_dict[rel_types[-1]]) == no_relation_count)
    print('Relation: {:s};\t Yes: {:d};\t No: 0'.format(rel_types[-1],
                                                        no_relation_count))
    # Write negative examples as a sixth rel to disk.
    rel_fname = os.path.join(out_path, rel_types[-1]) + '.json'
    rel_file = codecs.open(rel_fname, u'w', u'utf-8')
    for data_dict in rels_dict[rel_types[-1]]:
        proc_jsons = json.dumps(data_dict, ensure_ascii=False)
        rel_file.write(proc_jsons + '\n')
    rel_file.close()
    print('Wrote: {:s}'.format(rel_fname))
    # Print class proportions.
    total_rels = sum(list(map(len, rels_dict.values())))
    print('Total examples: {:d}'.format(total_rels))
    for rel_type in rels_dict.keys():
        count = len(rels_dict[rel_type])
        prop = count/float(total_rels)
        print('Relation: {:s};\t proportion: {:.4f};\t count: {:d}'.format(
            rel_type, prop, count))
    return rels_dict


############################################
#       Form the train/dev/test splits     #
############################################
def get_rand_indices(maxnum):
    """
    Return a permutation of the [0:maxnum-1] range.
    :return:
    """
    indices = range(maxnum)
    # Get random permutation of the indices but control for randomness.
    # https://stackoverflow.com/a/19307027/3262406
    random.shuffle(indices, lambda: 0.4186)
    return indices


def make_rel_splits(rel_examples):
    """
    For any given relation make a train, dev, test split.
    :param rel_examples: list(data_json); all the examples for a given rel.
    :return:
    """
    num_examples = len(rel_examples)
    # Get a permutation of the [0:num_examples-1] range.
    indices = get_rand_indices(num_examples)
    train_i = indices[:int(0.8*num_examples)]
    dev_i = indices[int(0.8*num_examples): int(0.9*num_examples)]
    test_i = indices[int(0.9*num_examples):]
    train = [rel_examples[i] for i in train_i]
    dev = [rel_examples[i] for i in dev_i]
    test = [rel_examples[i] for i in test_i]
    return train, dev, test


def make_tdt_split(rels_dict):
    """
    Given the examples split by relation, make a per relation split and join
    them to get a train, dev, test of the dataset.
    :param rels_dict: dict(list(data_json)); big dict with data for
        each relation.
    :return: train, dev, test: lists;
    """
    train, dev, test = [], [], []
    # Join the train dev test splits for each relation into one big train, dev,
    # test split.
    for rel_type, rel_examples in rels_dict.items():
        rel_train, rel_dev, rel_test = make_rel_splits(rel_examples)
        train.extend(rel_train)
        dev.extend(rel_dev)
        test.extend(rel_test)
    # Shuffle each split.
    indices = get_rand_indices(len(train))
    train = [train[i] for i in indices]
    indices = get_rand_indices(len(dev))
    dev = [dev[i] for i in indices]
    indices = get_rand_indices(len(test))
    test = [test[i] for i in indices]
    return train, dev, test


def control_tdt_split(mentionfound_path, out_path):
    """
    Form the sixth relation from the grec-mentionfound data and make a 80:10:10
    train:dev:split. The splits are such that the class proportions in the full
    dataset are maintained in each split.
    :param mentionfound_path:
    :param out_path:
    :return:
    """
    rels_dict = form_sixth_rel(dataset_path=mentionfound_path, out_path=out_path)
    train, dev, test = make_tdt_split(rels_dict=rels_dict)
    print('Train: {}; Dev: {}; Test: {}'.format(len(train), len(dev),
                                                len(test)))

    # Write splits to disk.
    for split, split_str in [(train, 'train'), (dev, 'dev'), (test, 'test')]:
        split_fname = os.path.join(out_path, split_str) + '.json'
        split_file = codecs.open(split_fname, u'w', u'utf-8')
        for data_dict in split:
            proc_jsons = json.dumps(data_dict, ensure_ascii=False)
            split_file.write(proc_jsons + '\n')
        split_file.close()
        print('Wrote: {:s}'.format(split_fname))


############################################
#   Form the train set for Vanilla LSTM    #
############################################
def process_splits(in_path, out_path):
    """
    For a given split process all the examples in it.
    :return:
    """
    obj_pat = re.compile(r'\(\(OBJ\)\)')
    sub_pat = re.compile(r'\(\(SUB\)\)')
    for split_str in ['train', 'dev', 'test']:
        in_split_fname = os.path.join(in_path, split_str) + '.json'
        out_split_fname = os.path.join(out_path, split_str) + '.json'
        out_split_file = codecs.open(out_split_fname, u'w', u'utf-8')
        print('Processing: {:s}'.format(in_split_fname))
        start = time.time()
        with codecs.open(in_split_fname, 'r', 'utf-8') as fp:
            for data_dict in du.read_json(fp):
                out_dict = {}
                text = data_dict['evidences'][0]['snippet']
                replaced, _ = obj_pat.subn(string=text, repl='target_object')
                replaced, _ = sub_pat.subn(string=replaced, repl='target_subject')
                # Tokenize text.
                word_li = word_tokenize(replaced)
                out_dict['text'] = ' '.join(word_li)
                out_dict['label'] = data_dict['int_mapped_rel']
                out_dict['readable_sub'] = data_dict['readable_sub']
                out_dict['readable_obj'] = data_dict['readable_obj']
                proc_jsons = json.dumps(out_dict, ensure_ascii=False)
                out_split_file.write(proc_jsons + '\n')
        out_split_file.close()
        end = time.time()
        print('Wrote: {:s}'.format(out_split_fname))
        print('Took: {:4.4f}s'.format(end-start))


def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest=u'subcommand',
                                       help=u'The action to perform.')

    # Action to make the general train/dev/test split.
    make_tdt_split = subparsers.add_parser(u'split')
    make_tdt_split.add_argument(u'-i', u'--in_path',
                                required=True,
                                help=u'The GREC mentionfound dataset. Would work'
                                     u'with others too but meaningless.')
    make_tdt_split.add_argument(u'-o', u'--out_path',
                                required=True,
                                help=u'Path to which splits should get written.')

    # Action to make the processed dataset that mine and Katies code reads.
    make_glstm_dataset = subparsers.add_parser(u'vlstm')
    make_glstm_dataset.add_argument(u'-i', u'--in_path', required=True,
                                    help=u'Directory with the train/dev/test '
                                         u'split jsons.')
    make_glstm_dataset.add_argument(u'-o', u'--out_path', required=True,
                                    help=u'Directory to which the processed '
                                         u'splits, should be written.')
    cl_args = parser.parse_args()

    if cl_args.subcommand == 'split':
        # Dont want to overwrite existing files by same name. :/
        assert(cl_args.in_path != cl_args.out_path)
        control_tdt_split(mentionfound_path=cl_args.in_path,
                          out_path=cl_args.out_path)
    elif cl_args.subcommand == 'vlstm':
        # Dont want to overwrite existing files by same name. :/
        assert (cl_args.in_path != cl_args.out_path)
        process_splits(in_path=cl_args.in_path, out_path=cl_args.out_path)
    else:
        sys.stderr.write('Unknown action.\n')


if __name__ == '__main__':
    main()
