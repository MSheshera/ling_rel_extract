"""
Add strings for entities to the dataset from freebase and find entity mentions
in the evidence.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, time
import codecs, json, gzip, bz2
import re, copy
import collections

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
import spacy
nlp = spacy.load('en')

# My imports.
import data_utils as du

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


###############################################
# Augment the dataset with readable entities. #
###############################################
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


def add_readable_ents(data_dict, fb_wiki_map, rel_type):
    """
    For the JSON file passed look up the MID and add readable string and return
    the dict to be written out.
    Readable_* are underscore separated Wikipedia article titles.
    :param data_json:
    :param fb_wiki_map:
    :param rel_type:
    :return:
        data_dict: dict; same as data_dict but with readble entities.
        unmapped_sub: int; 0 if map was successful 1 otherwise.
        unmapped_obj: int; 0 if map was successful 1 otherwise.
    """
    sub, obj = data_dict['sub'], data_dict['obj']
    # Map objects based on the relation type.
    if rel_type == 'education-degree.json':
        readable_obj = fb_wiki_map.get(obj, 'unmapped_obj')
        unmapped_obj = 0
    elif rel_type == 'date_of_birth.json':
        readable_obj = obj
        unmapped_obj = 0
    else:
        readable_obj = fb_wiki_map.get(obj, 'unmapped_obj')
        unmapped_obj = 1 if obj == 'unmapped_obj' else 0
    # Subjects are always names so map them all the same.
    readable_sub = fb_wiki_map.get(sub, 'unmapped_sub')
    unmapped_sub = 1 if sub == 'unmapped_sub' else 0
    data_dict['readable_obj'] = readable_obj
    data_dict['readable_sub'] = readable_sub
    return data_dict, unmapped_sub, unmapped_obj


def augment_dataset(raw_data_path, fb2wiki_map_file, proc_data_path):
    """
    Augment the json files to have actual names of the entities.
    :param dataset_path:
    :return:
    """
    # Read map from MID to article maps.
    fb_wiki_map, wiki_fb_map = read_fb_to_wiki(fb2wiki_map_file)

    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'date_of_birth.json', 'education-degree.json']
    for fname in fnames:
        raw_fname = os.path.join(raw_data_path, fname)
        proc_fname = os.path.join(proc_data_path, fname)
        proc_file = codecs.open(proc_fname, u'w', u'utf-8')
        with codecs.open(raw_fname, 'r', 'utf-8') as raw_file:
            rel_unmapped_count ={'unmapped_sub': 0, 'unmapped_obj': 0}
            print('Processing: {:s}'.format(raw_fname))
            for data_dict in du.read_json(raw_file):
                if data_dict != {}:
                    # For each json file add the entities.
                    readable_dict, unmapped_sub, unmapped_obj = \
                        add_readable_ents(data_dict, fb_wiki_map,
                                          rel_type=fname)
                    # Keep count of how many are fails.
                    rel_unmapped_count['unmapped_sub'] += unmapped_sub
                    rel_unmapped_count['unmapped_obj'] += unmapped_obj
                    # Only write out if both the subject and the object could be
                    # resolved.
                    if unmapped_sub == 0 and unmapped_obj == 0:
                        proc_jsons = json.dumps(readable_dict, ensure_ascii=False)
                        proc_file.write(proc_jsons+'\n')
        print(rel_unmapped_count)
        proc_file.close()


################################################
# Try to find entity mentions with crosswikis. #
################################################
def create_dataset_map(raw_data_path, fb2wiki_map_file, dataset_map_path):
    """
    Create a mid->wiki map just for the dataset.
    :param raw_data_path:
    :return:
    """
    # Read map from MID to article maps.
    fb_wiki_map, wiki_fb_map = read_fb_to_wiki(fb2wiki_map_file)

    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'date_of_birth.json', 'education-degree.json']
    ent_mids = []
    for fname in fnames:
        raw_fname = os.path.join(raw_data_path, fname)
        with codecs.open(raw_fname, 'r', 'utf-8') as raw_file:
            print('Processing: {:s}'.format(raw_fname))
            for data_dict in du.read_json(raw_file):
                if data_dict != {}:
                    sub = data_dict['sub']
                    obj = data_dict['obj']
                    readable_sub = fb_wiki_map.get(sub, 'unmapped')
                    readable_obj = fb_wiki_map.get(obj, 'unmapped')
                    if readable_obj != 'unmapped':
                        ent_mids.append((obj, readable_obj))
                    if readable_sub != 'unmapped':
                        ent_mids.append((sub, readable_sub))
    ent_mids = list(set(ent_mids))
    dataset_map = os.path.join(dataset_map_path, 'grec-fbtowiki.tsv')
    with codecs.open(dataset_map, 'w', 'utf-8') as map_file:
        for ent_mid in ent_mids:
            map_file.write('{:s}\t{:s}\n'.format(ent_mid[0][1:], ent_mid[1]))


def read_crosswikis(cross_wiki_file, wiki_fb_map, out_mention_ent_map):
    print('Reading in cross wikis')
    start = time.time()
    mention_prob_map = {}
    mention_entity_map = {}
    map_match = 0
    no_map_match = 0
    mention_entity_map_file = codecs.open(os.path.join(out_mention_ent_map, 'grec-mention_map.txt'), 'w', 'utf-8')
    mention_prob_map_file = codecs.open(os.path.join(out_mention_ent_map, 'grec-mention_prob_map.txt'), 'w', 'utf-8')
    # Read lines as unicode so other stuff here works:
    # https://stackoverflow.com/a/1883734/3262406
    ureader = codecs.getreader('latin-1')
    with bz2.BZ2File(cross_wiki_file, 'r') as f:
        fcontents = ureader(f)
        for line_num, line in enumerate(fcontents):
            if line_num % 10000 == 0:
                print('Line: %d  \t Matches: %d \t Map Matches: %d \t No Map'
                      ' Matches: %d \r' % (line_num, (map_match+no_map_match),
                                           map_match, no_map_match))
            try:
                mention, parts = line.split('\t')
            except ValueError:
                continue
            #print('mention: {} parts: {}'.format(mention, parts))
            try:
                parts = parts.split(' ', 2)
                prob, wiki, _ = parts
            except:
                continue
            # Only care about it if the wiki article is in the dataset else dc.
            if wiki in wiki_fb_map:
                # print('prob: {} wiki: {}'.format(prob, wiki))
                prob = float(prob)
                # print(wiki, wiki in wiki_fb_map)
                if mention not in mention_prob_map or prob > mention_prob_map[mention]:
                    mention_prob_map[mention] = prob
                    fb = wiki_fb_map[wiki]
                    mention_entity_map[mention] = '%s\t%s' % (wiki, fb)
                    mention_entity_map_file.write('%s\t%s\t%s\n' % (mention, wiki, fb))
                    mention_prob_map_file.write('%s\t%f\n' % (mention, prob))
                    if wiki in wiki_fb_map:
                        map_match += 1
                    else:
                        no_map_match += 1
    mention_entity_map_file.close()
    mention_prob_map_file.close()
    print('\nDone. Took %5.2f seconds' % (time.time() - start))


def crosss_wiki_mentions(out_mention_path, small_fb2wiki_map_file, cross_wiki_file):
    # Read map from MID to article maps.
    fb_wiki_map, wiki_fb_map = read_fb_to_wiki(small_fb2wiki_map_file)
    read_crosswikis(cross_wiki_file, wiki_fb_map, out_mention_path)


############################################################
# Try to find entity mentions with partial string matches. #
############################################################
def get_named_entities(data_dict):
    """
    Given the data_dict return all named entities.
    :return:
    """
    text = data_dict['evidences'][0]['snippet']
    doc = nlp(text)
    return doc.ents


def ner_mentions(readable_data_path):
    # fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
    #           'date_of_birth.json', 'education-degree.json']
    fnames = ['date_of_birth.json']
    dir_sub = 0
    ent_sub = 0
    for fname in fnames:
        raw_fname = os.path.join(readable_data_path, fname)
        with codecs.open(raw_fname, 'r', 'utf-8') as raw_file:
            print('Processing: {:s}'.format(raw_fname))
            sys.stderr.write('Processing: {:s}\n'.format(raw_fname))
            for data_dict in du.read_json(raw_file):
                if data_dict != {}:
                    newstring = data_dict['evidences'][0]['snippet']
                    readable_sub = ' '.join(data_dict['readable_sub'].split('_'))
                    sub_pat = re.escape(re.sub('\(.*?\)', '', readable_sub))
                    #readable_obj = data_dict['readable_obj']
                    sub_pat = re.compile('({:s})'.format(sub_pat))
                    newstring, sub_resolved = sub_pat.subn(r'((SUB: \1))',
                                                           newstring)
                    if sub_resolved == 0:
                        ents = get_named_entities(data_dict)
                        for ent in ents:
                            subscore = fuzz.partial_ratio(ent.text, readable_sub)
                            # objscore = fuzz.partial_ratio(ent.text, readable_obj)
                            if subscore > 60:
                                print((subscore, ent.text, readable_sub))
                                ent_sub += 1
                            # if objscore > 60:
                            #     print((objscore, ent.text, readable_obj))
                            #     resolved_sub += 1
                        print()
                    else:
                        print(newstring)
                        print(readable_sub)
                        print()
                        dir_sub += 1

        print('dir:{}, ent: {}'.format(dir_sub, ent_sub))


###########################################################
# Try to find entity mentions with direct string matches. #
###########################################################
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
    sub = ' '.join(data_dict['readable_sub'].split('_'))
    # Remove any bracketed text in the subject (because these are wiki article
    # titles)
    sub_pat = re.escape(re.sub('\(.*?\)', '', sub))
    sub_pat = re.compile('({:s})'.format(sub_pat))
    newstring, sub_resolved = sub_pat.subn(r'((SUB: \1))', newstring)
    if sub_resolved == 0:
        return new_dict
    # Resolve the object.
    if relation == 'education-degree.json':
        ed_pat_1 = re.compile(r'\(\(NAM:(.*?)\)\)')
        ed_pat_2 = re.compile(r'\(\(NOM:(.*?)\)\)')
        newstring, obj_resolved = ed_pat_1.subn(r'((OBJ:\1))', newstring)
        newstring, obj_resolved = ed_pat_2.subn(r'((OBJ:\1))', newstring)
    elif relation == 'date_of_birth.json':
        date_pat = data_dict['obj']
        newstring, obj_resolved = re.subn(date_pat, '<object>', newstring)
    else:
        obj = ' '.join(data_dict['readable_obj'].split('_'))
        # Remove any bracketed text in the subject (because these are wiki article
        # titles)
        obj_pat = re.sub('\(.*?\)', '', obj)
        obj_pat = re.compile('({:s})'.format(obj_pat))
        newstring, obj_resolved = obj_pat.subn(r'((OBJ: \1))', newstring)
    if sub_resolved and obj_resolved:
        new_dict['text'] = newstring
        return new_dict
    else:
        return new_dict


def dir_mentions(proc_data_path, out_data_path):
    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'education-degree.json']
    # 'date_of_birth.json',
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
    augment_dataset(raw_data_path = '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/google-relation-extraction/google-relation-extraction',
                    fb2wiki_map_file = '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/freebase/freebase_to_dbpedia_fixed.gz',
                    proc_data_path = '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-readable')
    # dir_mentions('/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-readable',
    #               '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-qd')
    # crosss_wiki_mentions(
    #     '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed',
    #     '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-fbtowiki.tsv.gz',
    #     '/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/freebase/dictionary.bz2')
    # ner_mentions('/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed/grec-readable')
    # create_dataset_map(raw_data_path='/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/google-relation-extraction/google-relation-extraction',
    #                    fb2wiki_map_file='/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/datasets/freebase/freebase_to_dbpedia_fixed.gz',
    #                    dataset_map_path='/mnt/nfs/work1/mccallum/smysore/ling_rel_extract/processed')