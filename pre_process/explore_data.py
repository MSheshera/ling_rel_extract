"""
Get counts of what the data looks like.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import codecs
import collections
import pprint
import re

# My imports.
import data_utils as du


def count_rel_negative(fname):
    """
    Count negative examples for the relation in the file passed.
    :param fname:
    :return:
    """
    bad_json = 0
    with codecs.open(fname, 'r', 'utf-8') as fp:
        j_count_dict = collections.defaultdict(int)
        for data_json in du.read_json(fp):
            # Handle for json strings which couldn't be parsed.
            if data_json != {}:
                judgement = collections.Counter([j['judgment'] for j in
                                                 data_json['judgments']])
                # print(dict(judgement))
                # Count as positive example if \geq half the people people
                # say yes.
                judgement = 'yes' if judgement['yes']/float(sum(
                    judgement.values())) > 0.5 else 'no'
                j_count_dict[judgement] += 1
                if judgement == 'no':
                    print(data_json)
                    print()
            else:
                bad_json += 1
    print(j_count_dict)
    print('Skipped {:d} json files.'.format(bad_json))


def count_negatives(dataset_path):
    """
    Count negatives for each relation type.
    :param dataset_path: unicode; full path to the dir with all rel jsons.
    :return:
    """
    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'date_of_birth.json', 'education-degree.json']
    for fname in fnames:
        print(fname)
        fname = os.path.join(dataset_path, fname)
        count_rel_negative(fname)


def count_relation_overlap(dataset_path):
    """
    Count how many of the entity pairs occur in multiple relations.
    :param dataset_path:
    :return:
    """
    total_data_samples = 0
    entity_pair = collections.defaultdict(int)
    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'date_of_birth.json', 'education-degree.json']
    for fname in fnames:
        print(fname)
        fname = os.path.join(dataset_path, fname)
        with codecs.open(fname, 'r', 'utf-8') as fp:
            for data_json in du.read_json(fp):
                if data_json != {}:
                    total_data_samples += 1
                    entity_pair[(data_json['sub'], data_json['obj'])] += 1
    print('unique entity pairs: {:d}'.format(len(entity_pair)))
    print('total entity pairs: {:d}'.format(len(entity_pair)))


if __name__ == '__main__':
    #count_negatives(sys.argv[1])
    count_relation_overlap(sys.argv[1])