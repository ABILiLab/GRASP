#!/usr/bin/env python3

import os
import re
import sys

_ad = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ad not in sys.path:
    sys.path.insert(0, _ad)
from _paths import linearfold_binary_candidates, tracker_data, tracker_features
import pickle
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
import itertools
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, KFold

def read_fasta_to_df(file_path):
    records = list(SeqIO.parse(file_path, 'fasta'))
    df = pd.DataFrame({'Description': [str(record.description).split('|')[0] for record in records], 'Sequence': [str(record.seq) for record in records], 'Label': [','.join(str(record.id).split('|')[-1].split(',')) for record in records]})
    return df

def linear_fold(sequences, ids, out_fasta_name, linearfold_path='./linearfold_v'):
    if not os.path.exists(linearfold_path):
        for path in linearfold_binary_candidates():
            if os.path.exists(path):
                linearfold_path = path
                break
        else:
            raise FileNotFoundError(f'LinearFold binary not found: {linearfold_path}')
    for seq, id in zip(sequences, ids):
        print(f'{id} is processing...')
        with open('tmp.fasta', 'w') as ofile:
            ofile.write(f'>{id}\n{seq}\n')
        os.system(f'cat tmp.fasta | {linearfold_path} > tmp.dot')
        in_lines = open('tmp.dot', 'r').readlines()
        with open('clean_tmp.dot', 'w') as out_file:
            for line in in_lines:
                if '>' in line:
                    out_file.write(':'.join(line.split(':')[1:]).strip() + '\n')
                else:
                    out_file.write(line)
        os.system('cat ' + 'clean_tmp.dot' + ' >> ' + out_fasta_name + '.fasta')

def dot_fasta_to_pkl(file, out_pkl):
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    seq_dotbracket = {}
    for fasta in records:
        valueList = []
        array = fasta.split('\n')
        sequence, dot_bracket = (array[1], array[2])
        sequence = re.sub('U', 'T', sequence)
        if 'N' in sequence:
            sequence = re.sub('N', 'G', sequence)
            print(array[0])
        dot_bracket_list = dot_bracket.split()
        ev = float(dot_bracket_list[1].split('(')[1].split(')')[0])
        valueList.append(dot_bracket_list[0])
        valueList.append(ev)
        seq_dotbracket[sequence] = valueList
    with open(out_pkl, 'wb') as handle:
        pickle.dump(seq_dotbracket, handle)
    return seq_dotbracket

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta', type=str, default=tracker_data('sequences.fasta'))
    parser.add_argument('--output_pkl', type=str, default=tracker_features('sequences_linearfold.pkl'))
    parser.add_argument('--linearfold_path', type=str, default='./linearfold_v')
    args = parser.parse_args()
    print('=' * 80)
    print('LinearFold secondary structure extraction')
    print('=' * 80)
    print(f'Input FASTA: {args.input_fasta}')
    print(f'Output PKL: {args.output_pkl}')
    print('=' * 80)
    if not os.path.exists(args.input_fasta):
        raise FileNotFoundError(f'Input file not found: {args.input_fasta}')
    output_dir = os.path.dirname(args.output_pkl) if os.path.dirname(args.output_pkl) else '.'
    os.makedirs(output_dir, exist_ok=True)
    print('\nReading FASTA...')
    df = read_fasta_to_df(args.input_fasta)
    sequences = df['Sequence'].tolist()
    ids = df['Description'].tolist()
    print(f'Loaded {len(sequences)} sequences')
    print('\nRunning LinearFold...')
    temp_output_fasta = os.path.join(output_dir, 'linearfold_output')
    linear_fold(sequences, ids, temp_output_fasta, args.linearfold_path)
    output_fasta_file = temp_output_fasta + '.fasta'
    if not os.path.exists(output_fasta_file):
        raise FileNotFoundError(f'LinearFold output not found: {output_fasta_file}')
    print('\nConverting to PKL...')
    seq_dotbracket = dot_fasta_to_pkl(output_fasta_file, args.output_pkl)
    print(f'\nDone.')
    print(f'  Output: {args.output_pkl}')
    print(f'  Sequences: {len(seq_dotbracket)}')
    for temp_file in ['tmp.fasta', 'tmp.dot', 'clean_tmp.dot']:
        if os.path.exists(temp_file):
            os.remove(temp_file)
if __name__ == '__main__':
    main()
