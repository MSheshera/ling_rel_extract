"""
Get counts of what the data looks like.
"""
from __future__ import unicode_literals
import os, sys
import codecs, json
import collections
import pprint
import re


def read_json(json_file):
    """
    Read per line JSON and yield.
    :param data_file: Just a open file. file-like with a next method.
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


def count_rel_negative(fname):
    """
    Count negative examples for the relation in the file passed.
    :param fname:
    :return:
    """
    bad_json = 0
    with codecs.open(fname, 'r', 'utf-8') as fp:
        j_count_dict = collections.defaultdict(int)
        for data_json in read_json(fp):
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
    fnames = ['institution.json', 'date_of_birth.json', 'education-degree.json',
              'place_of_birth.json', 'place_of_death.json']
    for fname in fnames:
        print(fname)
        fname = os.path.join(dataset_path, fname)
        count_rel_negative(fname)


if __name__ == '__main__':
    count_negatives(sys.argv[1])
