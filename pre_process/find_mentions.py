"""
Find entity mentions in the text with direct string matches where-ever possible
else find them with partial string matches.
"""
from __future__ import unicode_literals
from __future__ import print_function
import os, sys
import codecs, json
import re
import collections
import datetime
from dateutil import parser as dateparser

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
    the object.
    :param data_dict: dict; the reas in JSON data; modified in place
    :param doc_ents:
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
    newstring, obj_resolved = date_pat.subn(r'((OBJ: \1))', orstring)
    if obj_resolved:
        data_dict['evidences'][0]['snippet'] = newstring
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
            newstring, obj_resolved = date_pat.subn(r'((OBJ: \1))', newstring)
            if obj_resolved:
                data_dict['evidences'][0]['snippet'] = newstring
                resolved = True
    return resolved


def find_degree_mentions(data_dict):
    """
    Theres nothing to find here, simply replace all NAM or NOMs with OBJ in
    data_dict.
    :param data_dict: dict; the reas in JSON data; modified in place
    :return: resolved: bool; True if a degree mention was found else False.
    """
    ed_pat_1 = re.compile(r'\(\(NAM:(.*?)\)\)')
    ed_pat_2 = re.compile(r'\(\(NOM:(.*?)\)\)')
    resolved = False
    orstring = data_dict['evidences'][0]['snippet']
    newstring, obj_resolved_1 = ed_pat_1.subn(r'((OBJ:\1))',
                                              orstring)
    newstring, obj_resolved_2 = ed_pat_2.subn(r'((OBJ:\1))',
                                              newstring)
    if obj_resolved_1 or obj_resolved_2:
        data_dict['evidences'][0]['snippet'] = newstring
        resolved = True
    return resolved


def process_rels(readable_data_path):
    """
    Process examples in reach relation and call appropriate functions to
    find mentions based on the relation type.
    :return:
    """
    # fnames = ['institution.json', 'place_of_birth.json', 'place_of_death.json',
    #           'date_of_birth.json', 'education-degree.json']
    fnames = ['date_of_birth.json']
    for fname in fnames:
        raw_fname = os.path.join(readable_data_path, fname)
        rels_resolved = {'obj':0, 'sub':0}
        with codecs.open(raw_fname, 'r', 'utf-8') as raw_file:
            print('Processing: {:s}'.format(raw_fname))
            for data_dict in du.read_json(raw_file):
                obj_resolved = False
                # Get the named entities in the text.
                doc_ents = get_named_entities(data_dict)
                # Resolve the object of the easier to handle relations.
                if fname == 'education-degree.json':
                    obj_resolved = find_degree_mentions(data_dict)
                elif fname == 'date_of_birth.json':
                    obj_resolved = find_date_mentions(data_dict, doc_ents)
                if obj_resolved != True:
                    continue
                else:
                    rels_resolved['obj'] += 1
    print(rels_resolved)


if __name__ == '__main__':
    process_rels(readable_data_path=sys.argv[1])