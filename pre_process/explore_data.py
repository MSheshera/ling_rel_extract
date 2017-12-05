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

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


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
                print(dict(judgement))
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
    :param dataset_path: unicode; full path to the dir with all rel jsons.
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


def play_degree_objs(dataset_path):
    """
    Check if every case of degree has a the object clearly marked out.
    Generally function to play around with these degree objects.
    :param dataset_path: unicode; full path to the dir with all rel jsons.
    :return:
    """
    total_data_samples = 0
    fnames = ['education-degree.json']
    # Non-greedily match and replace:
    # https://stackoverflow.com/q/2403122/3262406
    # https://stackoverflow.com/a/6711631/3262406
    ed_pat_1 = re.compile(r'\(\(NAM:(.*?)\)\)')
    ed_pat_2 = re.compile(r'\(\(NOM:(.*?)\)\)')
    resolved_sample = 0
    for fname in fnames:
        print(fname)
        fname = os.path.join(dataset_path, fname)
        with codecs.open(fname, 'r', 'utf-8') as fp:
            for data_json in du.read_json(fp):
                if data_json != {}:
                    total_data_samples += 1
                    orstring = data_json['evidences'][0]['snippet']
                    newstring, obj_resolved_1 = ed_pat_1.subn(r'((OBJ:\1))', orstring)
                    newstring, obj_resolved_2 = ed_pat_2.subn(r'((OBJ:\1))', newstring)
                    if obj_resolved_1 or obj_resolved_2:
                        print(newstring)
                        print()
                        resolved_sample += 1
    print('total education-degree samples: {:d}'.format(total_data_samples))
    print('total objs resolved: {:d}'.format(resolved_sample))


def play_date_objs(dataset_path):
    """
    Play around with the data objects.
    :param dataset_path:
    :return:
    """
    total_data_samples = 0
    fnames = ['date_of_birth.json']
    resolved_sample = 0
    for fname in fnames:
        print(fname)
        fname = os.path.join(dataset_path, fname)
        with codecs.open(fname, 'r', 'utf-8') as fp:
            for data_json in du.read_json(fp):
                if data_json != {}:
                    total_data_samples += 1
                    orstring = data_json['evidences'][0]['snippet']
                    objstr = data_json['obj']
                    # Some dates start with zeros. Get rid of that.
                    try:
                        objstr = unicode(int(objstr))
                    except ValueError:
                        objstr = objstr
                    date_pat = re.compile(r'({:s})'.format(objstr))
                    newstring, obj_resolved = date_pat.subn(r'((OBJ: \1))',
                                                              orstring)
                    if not obj_resolved:
                        print(newstring)
                        print(objstr)
                        print()
                        resolved_sample += 1
    print('total date_of_birth samples: {:d}'.format(total_data_samples))
    print('total objs resolved: {:d}'.format(resolved_sample))


if __name__ == '__main__':
    if sys.argv[1] == 'count_neg':
        count_negatives(sys.argv[2])
    elif sys.argv[1] == 'count_overlap':
        count_relation_overlap(sys.argv[2])
    elif sys.argv[1] == 'play_deg':
        play_degree_objs(sys.argv[2])
    elif sys.argv[1] == 'date_obj':
        play_date_objs(sys.argv[2])
    else:
        sys.stderr.write('Unknown action. :/')