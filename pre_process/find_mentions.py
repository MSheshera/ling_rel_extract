"""
Find entity mentions in the text with direct string matches where-ever possible
else find them with partial string matches.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys, argparse
import codecs, json
import re
from dateutil import parser as dateparser
import time
from fuzzywuzzy import fuzz

import spacy
nlp = spacy.load('en')

# My imports.
import data_utils as du

# Write unicode to stdout.
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)


def get_named_entities(data_dict):
    """
    Given the data_dict return all named entities.
    :return: list; list of named entities, each element is a spacy span object.
    """
    text = data_dict['evidences'][0]['snippet']
    doc = nlp(text)
    return list(doc.ents)


def find_date_mentions(data_dict, doc_ents):
    """
    Find the date mentions in the text snippet. If direct matches can do the job
    be done with it. Else look for dates in the entities and try to match it to
    the object. Replace the dates with OBJ and build a list of the dates in the
    text order.
    :param data_dict: dict; the read in JSON data; modified in place
    :param doc_ents: list of named entities, each element is a spacy span object
    :return: resolved: bool; True if a degree mention was found else False.
    """
    resolved = False
    orstring = data_dict['evidences'][0]['snippet']
    date_str = data_dict['obj']
    # Some dates start with zeros. Get rid of that.
    try:
        date_str = unicode(int(date_str))
    except ValueError:
        date_str = date_str
    date_pat = re.compile(r'({:s})'.format(date_str))
    date_mentions = date_pat.findall(orstring)
    newstring, obj_resolved = date_pat.subn(r'((OBJ))', orstring)
    if obj_resolved:
        data_dict['evidences'][0]['snippet'] = newstring
        data_dict['OBJ_mentions'] = date_mentions
        resolved = True
        return resolved
    # If the direct match couldn't resolve it then look at the entities.
    date_str_dtobj = dateparser.parse(date_str)
    date_ents = [ent.text for ent in doc_ents if ent.label_ == 'DATE']
    for date_ent in date_ents:
        # Parse it into a datetime object.
        try:
            date_ent_dtobj = dateparser.parse(date_ent)
        except ValueError:
            continue
        # If its the same as the specified obj then mark it in the text.
        if date_ent_dtobj == date_str_dtobj:
            date_pat = re.compile(r'({:s})'.format(date_ent))
            date_mentions.extend(date_pat.findall(newstring))
            newstring, obj_resolved = date_pat.subn(r'((OBJ))', newstring)
            if obj_resolved:
                data_dict['evidences'][0]['snippet'] = newstring
                data_dict['OBJ_mentions'] = date_mentions
                resolved = True
    return resolved


def find_degree_mentions(data_dict):
    """
    Replace all NAM or NOM objects with OBJ and build the corresponding list of
    objs in the order in which they appear in the text.
    :param data_dict: dict; the read in JSON data; modified in place
    :return: resolved: bool; True if a degree mention was found else False.
    """
    ed_pat_1 = re.compile(r'\(\(NAM: (.*?)\)\)')
    ed_pat_2 = re.compile(r'\(\(NOM: (.*?)\)\)')
    resolved = False
    orstring = data_dict['evidences'][0]['snippet']
    newstring, obj_resolved_1 = ed_pat_1.subn(r'((OBJ:\1))', orstring)
    newstring, obj_resolved_2 = ed_pat_2.subn(r'((OBJ:\1))', newstring)
    ed_pat = re.compile(r'\(\(OBJ:(.*?)\)\)')
    ed_mentions = ed_pat.findall(newstring)
    newstring, obj_resolved = ed_pat.subn(r'((OBJ))', newstring)
    if obj_resolved:
        data_dict['evidences'][0]['snippet'] = newstring
        data_dict['OBJ_mentions'] = ed_mentions
        resolved = True
    return resolved


def find_namedent_mentions(data_dict, doc_ents, readable_ent, type_str, partial_ratio_thresh):
    """
    Find the readable_ent in the text snippet in data_dict. Replace the ents
    with type_str and build a list of the substituted ents.
    :param data_dict: dict; the read in JSON data; modified in place
    :param doc_ents: list of named entities, each element is a spacy span object
    :param partial_ratio_thresh: int; threshold for accepting a mention.
    :return: resolved: bool; True if a degree mention was found else False.
    """
    resolved = False
    orstring = data_dict['evidences'][0]['snippet']
    labels = [U'LOC', U'PRODUCT', U'NORP', U'WORK_OF_ART', U'GPE', U'PERSON',
              U'FAC', U'ORG']
    # Get rid of underscores and text in brackets because these are wikipedia
    # page titles.
    ent_str = ' '.join(readable_ent.split('_'))
    ent_str = re.sub('\(.*?\)', '', ent_str).strip()
    # Form patterns and escape random spurious chars you cant control.
    ent_pat = re.compile(r'({:s})'.format(re.escape(ent_str)))
    # In case NER misses it, get whatever matches with a direct string match.
    ent_mentions = [(m.start(0),m.end(0),m.group(0)) for m in ent_pat.finditer(orstring)]
    newstring, dir_resolved = ent_pat.subn(r'(({:s}))'.format(type_str), orstring)
    # Next match partially with named entities.
    par_resolved = 0
    doc_ents = set([ent.text for ent in doc_ents if ent.label_ in labels])
    for ent in doc_ents:
        # If its the directly matched name or an already seen ent then skip it.
        if ent == ent_str:
            continue
        match_score = fuzz.partial_ratio(ent, ent_str)
        if match_score > partial_ratio_thresh:
            ent_pat = re.compile(r'({:s})'.format(re.escape(ent)))
            ent_mentions.extend([(m.start(0), m.end(0), m.group(0)) for m in ent_pat.finditer(newstring)])
            newstring, ret_resolved = ent_pat.subn(r'(({:s}))'.format(type_str),
                                                   newstring)
            par_resolved += ret_resolved
    # Only replace if resolved.
    if dir_resolved or par_resolved:
        data_dict['evidences'][0]['snippet'] = newstring
        # Sort on the start index of the entity mentions. This isnt perfect
        # since some of the indexes come from newstring. But im overlooking that
        # for now.
        # TODO: Account for newstring and orstring indexes. --lowpri
        ent_mentions = sorted(ent_mentions, key=lambda elm: elm[0])
        data_dict['{:s}_mentions'.format(type_str)] = [m[2] for m in ent_mentions]
        resolved = True
        return resolved
    return resolved


def process_rels(readable_data_path, proc_data_path, partial_thresh):
    """
    Process examples in reach relation and call appropriate functions to
    find mentions based on the relation type.
    :return:
    """
    fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
              'date_of_birth.json', 'education-degree.json']
    rels_resolved = dict(zip(fnames, [0, 0, 0, 0, 0]))
    rels_total = dict(zip(fnames, [0, 0, 0, 0, 0]))
    for fname in fnames:
        start = time.time()
        count = 0
        raw_fname = os.path.join(readable_data_path, fname)
        proc_fname = os.path.join(proc_data_path, fname)
        proc_file = codecs.open(proc_fname, u'w', u'utf-8')
        with codecs.open(raw_fname, 'r', 'utf-8') as raw_file:
            print('Processing: {:s}'.format(raw_fname))
            for data_dict in du.read_json(raw_file):
                # if count > 99:
                #     break
                rels_total[fname] += 1
                count += 1
                # Get the named entities in the text.
                doc_ents = get_named_entities(data_dict)
                # Resolve the object based on the type of the relation.
                if fname == 'education-degree.json':
                    obj_resolved = find_degree_mentions(data_dict)
                elif fname == 'date_of_birth.json':
                    obj_resolved = find_date_mentions(data_dict, doc_ents)
                else:
                    obj_resolved = find_namedent_mentions(
                        data_dict=data_dict, doc_ents=doc_ents,
                        readable_ent=data_dict['readable_obj'], type_str='OBJ',
                        partial_ratio_thresh=partial_thresh)
                # If you resolved the object bother about the subject.
                if obj_resolved == False:
                    continue
                sub_resolved = find_namedent_mentions(
                    data_dict=data_dict, doc_ents=doc_ents,
                    readable_ent=data_dict['readable_sub'], type_str='SUB',
                    partial_ratio_thresh=partial_thresh)
                if sub_resolved and obj_resolved:
                    rels_resolved[fname] += 1
                    proc_jsons = json.dumps(data_dict, ensure_ascii=False)
                    proc_file.write(proc_jsons + '\n')
        proc_file.close()
        print('Wrote: {:s}'.format(proc_fname))
        end = time.time()
        print('Took {:5.2f}s'.format(end-start))
    print(rels_resolved)
    print(rels_total)


def main():
    """
    Parse command line arguments and call all the above routines.
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--in_dir',
                        help='Directory to read readable GREC from.')
    parser.add_argument('-o', '--out_dir',
                        help='Directory to write results to.')
    parser.add_argument('-t', '--partial_thresh',
                        type=int,
                        help='Partial ratio value to threshold at.')
    cl_args = parser.parse_args()
    process_rels(readable_data_path=cl_args.in_dir,
                 proc_data_path=cl_args.out_dir,
                 partial_thresh=cl_args.partial_thresh)


if __name__ == '__main__':
    main()