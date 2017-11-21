"""
Add strings for entities to the dataset from freebase and find entity mentions
in the evidence.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import codecs, json, gzip
import re, copy
import collections

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize

# My imports.
import data_utils as du


def read_fb_to_wiki(map_file):
    """
    Read the giant map file in and return a dict going from the MID to name.
    :param map_file:
    :return:
    """
    # Read lines as unicode so other stuff here works:
    # https://stackoverflow.com/a/1883734/3262406
    ureader = codecs.getreader('utf-8')
    print('Reading in wiki -> fb map')
    with gzip.open(map_file, 'r') as zf:
        zfcontents = ureader(zf)
        lines = [l.strip().split('\t') for l in zfcontents]
        wiki_fb_map = {parts[1]: ('/%s' % parts[0]) for parts in lines if
                       len(parts) == 2}
        fb_wiki_map = {('/%s' % parts[0]): parts[1] for parts in lines if
                       len(parts) == 2}
    # TODO: Look at why the map is smaller than the number of lines.
    print('Map file length: {:d}'.format(len(lines)))
    print('fb_wiki_map len: {:d}'.format(len(fb_wiki_map)))
    return fb_wiki_map, wiki_fb_map


def add_readable_ents(data_dict, fb_wiki_map):
    """
    For the JSON file passed look up the MID and add readable string and return
    the dict to be written out.
    :param data_json:
    :param fb_wiki_map:
    :return:
    """
    sub, obj = data_dict['sub'], data_dict['obj']
    # Readable objects are still underscore seperated wikipedia article titles.
    readable_obj = fb_wiki_map.get(obj, 'unmapped_obj')
    readable_sub = fb_wiki_map.get(sub, 'unmapped_sub')
    if readable_obj == 'unmapped_obj':
        print('unmapped obj: {:s}'.format(obj))
    if readable_sub == 'unmapped_sub':
        print('unmapped sub: {:s}'.format(sub))
    data_dict['readable_obj'] = readable_obj
    data_dict['readable_sub'] = readable_sub
    return data_dict


def augment_dataset(raw_data_path, fb2wiki_map_path, proc_data_path):
    """
    Augment the json files to have actual names of the entities.
    :param dataset_path:
    :return:
    """
    # Read map from MID to article maps.
    fb_wiki_map, wiki_fb_map = read_fb_to_wiki(fb2wiki_map_path)

    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'date_of_birth.json', 'education-degree.json']
    for fname in fnames:
        raw_fname = os.path.join(raw_data_path, fname)
        proc_fname = os.path.join(proc_data_path, fname)
        proc_file = codecs.open(proc_fname, u'w', u'utf-8')
        with codecs.open(raw_fname, 'r', 'utf-8') as raw_file:
            print('Processing: {:s}'.format(raw_fname))
            for data_dict in du.read_json(raw_file):
                if data_dict != {}:
                    # For each json file add the entities.
                    readable_dict = add_readable_ents(data_dict, fb_wiki_map)
                    proc_jsons = json.dumps(readable_dict, ensure_ascii=False)
                    proc_file.write(proc_jsons+'\n')
        proc_file.close()


# Quick and dirty code to find entity mentions based on direct string matches.
def normalize_text(par_str):
    pst = PunktSentenceTokenizer()
    sents = pst.tokenize(par_str)
    par_toks = []
    for sent in sents:
        par_toks.extend(word_tokenize(sent))
    norm_text = ' '.join(par_toks).lower()
    return norm_text


def find_json_mentions(data_dict, relation):
    new_dict = {}
    newstring = data_dict['evidences'][0]['snippet']
    # Resolve the subject.
    sub_pat = re.escape(' '.join(data_dict['readable_sub'].split('_')))
    newstring, sub_resolved = re.subn(sub_pat, '<subject>', newstring)
    if sub_resolved == 0:
        return new_dict
    # Resolve the object.
    if relation == 'education-degree.json':
        ed_pat = r'\(\(NAM:.+\)\)'
        newstring, obj_resolved = re.subn(ed_pat, '<object>', newstring)
    elif relation == 'date_of_birth.json':
        date_pat = data_dict['obj']
        newstring, obj_resolved = re.subn(date_pat, '<object>', newstring)
    else:
        obj_pat = re.escape(' '.join(data_dict['readable_obj'].split('_')))
        newstring, obj_resolved = re.subn(obj_pat, '<object>', newstring)
    if sub_resolved and obj_resolved:
        new_dict['text'] = normalize_text(newstring)
        return new_dict
    else:
        return new_dict


def find_mentions(proc_data_path, out_data_path):
    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'date_of_birth.json', 'education-degree.json']
    for idx, fname in enumerate(fnames):
        proc_fname = os.path.join(proc_data_path, fname)
        out_fname = os.path.join(out_data_path, fname)
        out_file = codecs.open(out_fname, u'w', u'utf-8')
        failcount = 0
        print(fname)
        with codecs.open(proc_fname, 'r', 'utf-8') as proc_file:
            for data_dict in du.read_json(proc_file):
                if data_dict != {}:
                    # Find ent mentions else return {}.
                    new_dict = find_json_mentions(data_dict, '')
                    if new_dict != {}:
                        # Fix label.
                        judgement = collections.Counter([j['judgment'] for j in
                                                         data_dict['judgments']])
                        # Count as positive example if \geq half the people
                        # say yes.
                        new_dict['label'] = idx if judgement['yes'] / float(sum(
                            judgement.values())) > 0.5 else 5
                        out_jsons = json.dumps(new_dict, ensure_ascii=False)
                        out_file.write(out_jsons + '\n')
                    else:
                        failcount += 1
        out_file.close()
        print('Failed to find ents in {:d}'.format(failcount))


if __name__ == '__main__':
    # augment_dataset('/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/google-relation-extraction/google-relation-extraction',
    #                 '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/freebase/freebase_to_dbpedia_fixed.gz',
    #                 '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-ent_resolved')
    find_mentions('/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-ent_resolved',
                  '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-qd')
