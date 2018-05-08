import os
import shutil
import argparse
import pickle

import utils
from sensekey_utils import load_mapping_sensekey2offset
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser(description='Converts WordNet gloss corpus to our intermediate ULM format')
parser.add_argument('-v',  action='version', version='1.0')
parser.add_argument('-i',  dest='corpora', required=True, help='SemCor | SemCor+OMSTI')
parser.add_argument('-o',  dest='output_folder', required=True, help='Output folder')
args = parser.parse_args()

# paths
if os.path.exists(args.output_folder):
    shutil.rmtree(args.output_folder)

os.mkdir(args.output_folder)

# wordnet
path_to_wn_dict_folder = str(wn._get_root())  # change this for other wn versions
path_to_wn_index_sense = os.path.join(path_to_wn_dict_folder, 'index.sense')  # change this for other wn versions
sensekey2offset = load_mapping_sensekey2offset(path_to_wn_index_sense,
                                               '30')

sensekey2instance_ids, \
synset2instance_ids, \
instance_id2instance_obj = utils.load_into_classes(args.corpora, sensekey2offset)



for basename, info in [('instances.bin', instance_id2instance_obj),
                        ('sensekey_index.bin', sensekey2instance_ids),
                        ('synset_index.bin', synset2instance_ids)]:

        # Save the instance object
        output_path = os.path.join(args.output_folder, basename)
        with open(output_path, 'wb') as outfile:
            pickle.dump(info, outfile, protocol=3)

print('Total number of instances: %d' % len(instance_id2instance_obj))

stats = dict()
for label, d in [('sensekey', sensekey2instance_ids),
                 ('synset', synset2instance_ids)]:
    counts = [len(value)
              for value in d.values()]

    avg = round(sum(counts) / len(counts), 2)
    stats[label] = avg

print('stats for sensekeys: #%s avg of %s' % (len(sensekey2instance_ids),
                                              stats['sensekey']))
print('stats for synsets: #%s avg of %s' % (len(synset2instance_ids),
                                            stats['synset']))
