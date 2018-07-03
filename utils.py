import os
from collections import defaultdict
from nltk.corpus import wordnet as wn
from lxml import etree
from my_classes import Sentence, Token
from datetime import datetime

def get_comp_paths(competition):
    """
    load path to competition paths:
    1. xml (system input)
    2. key (answers)

    :param str competition: senseval2 | senseval3 | semeval2007 | semeval2013 | semeval2015

    :rtype: dict
    :return: {'xml' : path to xml, 'key' : path to answers}
    """
    supported = {'senseval2',
                 'senseval3',
                 'semeval2007',
                 'semeval2013',
                 'semeval2015'}
    assert competition in supported, '{competition} not in supported competitions ones: {supported}'.format_map(locals())


    xml = os.path.join('resources',
                       'WSD_Unified_Evaluation_Datasets',
                       competition,
                       '{competition}.data.xml'.format_map(locals()))
    assert os.path.exists(xml)

    # add fake root
    fake_root_path = xml + '.fake_root'
    if not os.path.exists(fake_root_path):
        with open(fake_root_path, 'w') as outfile:
            with open(xml) as infile:
                for counter, line in enumerate(infile):
                    if counter == 1:
                        outfile.write('<root>\n')
                    outfile.write(line)
            outfile.write('</root>\n')

    xml = fake_root_path

    key = os.path.join('resources',
                       'WSD_Unified_Evaluation_Datasets',
                       competition,
                       '{competition}.gold.key.txt'.format_map(locals()))
    assert os.path.exists(key)

    return {'xml': xml,
            'key': key,
            'source' : {competition}
            }


def get_training_paths(corpora):
    """
    load path to competition paths:
    1. xml (system input)
    2. key (answers)

    :param str corpora: SemCor | SemCor+OMSTI

    :rtype: dict
    :return: {'xml' : path to xml, 'key' : path to answers}
    """
    supported = {'SemCor',
                 'OMSTI',
                 'SemCor+OMSTI'}
    assert corpora in supported, '{corpora} not in supported corpora ones: {supported}'.format_map(locals())

    # set sources
    if corpora == 'SemCor':
        source={'semcor'}
    elif corpora == 'OMSTI':
        source = {'mun'}
    elif corpora == 'SemCor+OMSTI':
        source = {'semcor', 'mun'}

    if corpora == 'OMSTI':
        corpora = 'SemCor+OMSTI'

    basename = '{corpora}.data.xml'.format_map(locals())
    xml = os.path.join('resources',
                       'WSD_Training_Corpora',
                       corpora,
                       basename.lower())
    assert os.path.exists(xml)

    # add fake root
    fake_root_path = xml + '.fake_root'
    if not os.path.exists(fake_root_path):
        with open(fake_root_path, 'w') as outfile:
            with open(xml) as infile:
                for counter, line in enumerate(infile):
                    if counter == 1:
                        outfile.write('<root>\n')
                    outfile.write(line)
            outfile.write('</root>\n')

    xml = fake_root_path

    basename = '{corpora}.gold.key.txt'.format_map(locals())
    key = os.path.join('resources',
                       'WSD_Training_Corpora',
                       corpora,
                       basename.lower())
    assert os.path.exists(key)

    return {'xml': xml,
            'key': key,
            'source': source}


def get_sensekey2synset():
    """
    get mapping from sensekey to wordnet synset object

    :rtype: dict
    :return: mapping sensekey (str) -> synset object (nltk.corpus.reader.wordnet.Synset)
    """
    key2synset = dict()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            key2synset[lemma.key()] = synset

    return key2synset


def represented_sensekeys(path_to_key_file):
    """
    load all sensekey represented in corpus (competition or training corpus)

    :param str path_to_key_file: endswith .gold.key.txt

    :rtype: tuple
    :return: (sensekey -> instance ids,
              instance_id -> sensekeys)
    """
    sensekey2ids = defaultdict(set)
    id2sensekeys = defaultdict(set)

    with open(path_to_key_file) as infile:
        for line in infile:
            id, *keys = line.strip().split()
            for key in keys:
                sensekey2ids[key].add(id)
                id2sensekeys[id].add(key)

    return sensekey2ids, id2sensekeys


def load_into_classes(competition, sensekey2offset, debug=0):
    """
    load competition info into classes
    
    :param str competition: senseval2 | senseval3 | semeval2007 | semeval2013 | semeval2015
    :param dict sensekey2offset: mapping wordnet sensekey -> synset identifier
    :param int verbose: verbosity of debugging info
    
    
    :rtype: tuple
    :return: (sensekey -> set of ids that contain them,
              synset -> set of ids that contain them,
              token id -> id that contains them
              instance_id -< sent object)
    """
    if competition in {'SemCor', 'OMSTI', 'SemCor+OMSTI'}:
        paths = get_training_paths(competition)
    else:
        paths = get_comp_paths(competition)

    sensekey2ids, id2sensekeys = represented_sensekeys(paths['key'])

    sensekey2instance_ids = defaultdict(set)
    synset2instance_ids = defaultdict(set)
    instance_id2instance_obj = dict()
    tokenid2instance_obj = dict()

    if debug >= 1:
        print(datetime.now(), 'started loading', paths['xml'])

    doc = etree.parse(paths['xml'])

    if debug >= 1:
        print(datetime.now(), 'finished loading', paths['xml'])

    all_pos = set()

    for corpus_el in doc.iterfind('corpus'):
        if corpus_el.get('source') in paths['source']:
            for sent_el in corpus_el.iterfind('text/sentence'):
                sent_id = sent_el.get('id')

                tokens = []

                for token_el in sent_el.getchildren():

                    token = token_el.text
                    lemma = token_el.get('lemma')
                    pos = token_el.get('pos')
                    id_ = None
                    sensekeys = set()
                    synsets = set()

                    if token_el.tag == 'instance':
                        id_ = token_el.get('id')
                        sensekeys = id2sensekeys[id_]
                        all_pos.add(pos)

                        synsets = set()
                        for sensekey in sensekeys:

                            sensekey2instance_ids[sensekey].add(sent_id)

                            if sensekey in sensekey2offset:
                                synset_id = sensekey2offset[sensekey]
                                synsets.add(synset_id)
                                synset2instance_ids[synset_id].add(sent_id)
                            else:
                                print('no synset found for: %s' % synset_id)

                    token_obj = Token(token_id=id_,
                                      text=token,
                                      lemma=lemma,
                                      universal_pos=pos,
                                      lexkeys=sensekeys,
                                      synsets=synsets)

                    tokens.append(token_obj)

                sent_obj = Sentence(sent_id, tokens)
                instance_id2instance_obj[sent_id] = sent_obj

    return sensekey2instance_ids, synset2instance_ids, instance_id2instance_obj

def synset2identifier(synset, wn_version):
    """
    return synset identifier of
    nltk.corpus.reader.wordnet.Synset instance

    :param nltk.corpus.reader.wordnet.Synset synset: a wordnet synset
    :param str wn_version: supported: '171 | 21 | 30'

    :rtype: str
    :return: eng-VERSION-OFFSET-POS (n | v | r | a)
    e.g.
    """
    offset = str(synset.offset())
    offset_8_char = offset.zfill(8)

    pos = synset.pos()
    if pos in {'j', 's'}:
        pos = 'a'

    identifier = 'eng-{wn_version}-{offset_8_char}-{pos}'.format_map(locals())

    return identifier
