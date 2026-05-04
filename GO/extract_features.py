#!/usr/bin/env python3

import os
import sys

_ad = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ad not in sys.path:
    sys.path.insert(0, _ad)
from _paths import general_tracker_root, go_analyze, tracker_features, tracker_references
import pickle
import re
import argparse
import numpy as np
from collections import Counter
import itertools
sys.path.insert(0, general_tracker_root())
import utils
myDiIndex = {'AA': 0, 'AC': 1, 'AG': 2, 'AT': 3, 'CA': 4, 'CC': 5, 'CG': 6, 'CT': 7, 'GA': 8, 'GC': 9, 'GG': 10, 'GT': 11, 'TA': 12, 'TC': 13, 'TG': 14, 'TT': 15}

def generate_property_pairs(my_property_name):
    pairs = []
    for i in range(len(my_property_name)):
        for j in range(i + 1, len(my_property_name)):
            pairs.append([my_property_name[i], my_property_name[j]])
            pairs.append([my_property_name[j], my_property_name[i]])
    return pairs

def extract_dacc_features(fastas, my_property_name, my_property_value, lag=2, kmer=2):
    my_index = myDiIndex
    encoding = {}
    print(f'Extracting DACC features, lag={lag}, kmer={kmer}')
    print(f'Using {len(my_property_name)} physicochemical properties: {my_property_name}')
    property_pairs = generate_property_pairs(my_property_name)
    for num, i in enumerate(fastas):
        if (num + 1) % 100 == 0:
            print(f'  Progress: {num + 1}/{len(fastas)}')
        name, sequence, label = (i[0], re.sub('-', '', i[1]), i[2])
        code = []
        for p in my_property_name:
            mean_value = 0
            for j in range(len(sequence) - kmer + 1):
                dinucleotide = sequence[j:j + kmer]
                if dinucleotide in my_index:
                    mean_value = mean_value + float(my_property_value[p][my_index[dinucleotide]])
            mean_value = mean_value / (len(sequence) - kmer + 1)
            for l in range(1, lag + 1):
                ac_value = 0
                for j in range(len(sequence) - kmer - l + 1):
                    dinuc1 = sequence[j:j + kmer]
                    dinuc2 = sequence[j + l:j + l + kmer]
                    if dinuc1 in my_index and dinuc2 in my_index:
                        ac_value = ac_value + (float(my_property_value[p][my_index[dinuc1]]) - mean_value) * (float(my_property_value[p][my_index[dinuc2]]) - mean_value)
                if len(sequence) - kmer - l + 1 > 0:
                    ac_value = ac_value / (len(sequence) - kmer - l + 1)
                code.append(ac_value)
        for pair in property_pairs:
            mean_p1 = 0
            mean_p2 = 0
            for j in range(len(sequence) - kmer + 1):
                dinucleotide = sequence[j:j + kmer]
                if dinucleotide in my_index:
                    mean_p1 = mean_p1 + float(my_property_value[pair[0]][my_index[dinucleotide]])
                    mean_p2 = mean_p2 + float(my_property_value[pair[1]][my_index[dinucleotide]])
            mean_p1 = mean_p1 / (len(sequence) - kmer + 1)
            mean_p2 = mean_p2 / (len(sequence) - kmer + 1)
            for l in range(1, lag + 1):
                cc_value = 0
                for j in range(len(sequence) - kmer - l + 1):
                    dinuc1 = sequence[j:j + kmer]
                    dinuc2 = sequence[j + l:j + l + kmer]
                    if dinuc1 in my_index and dinuc2 in my_index:
                        cc_value = cc_value + (float(my_property_value[pair[0]][my_index[dinuc1]]) - mean_p1) * (float(my_property_value[pair[1]][my_index[dinuc2]]) - mean_p2)
                if len(sequence) - kmer - l + 1 > 0:
                    cc_value = cc_value / (len(sequence) - kmer - l + 1)
                code.append(cc_value)
        encoding[sequence] = code
    print(f'DACC extraction done, {len(encoding)} sequences')
    return encoding

def load_physicochemical_properties(data_file):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f'Physicochemical data file not found: {data_file}')
    with open(data_file, 'rb') as f:
        my_property_value = pickle.load(f)
    my_property_name = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']
    for prop in my_property_name:
        if prop not in my_property_value:
            print(f"Warning: property '{prop}' missing from data file")
    return (my_property_name, my_property_value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta', type=str, default=os.path.join(go_analyze('lncRNA'), 'sequences.fasta'))
    parser.add_argument('--kmer_path', type=str, default=tracker_features('sequences_5mer.pkl'))
    parser.add_argument('--dacc_path', type=str, default=tracker_features('sequences_dacc.pkl'))
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--lag', type=int, default=2)
    parser.add_argument('--phyche_data', type=str, default=tracker_references('iLearn-master', 'data', 'dirnaPhyche.data'))
    parser.add_argument('--extract_kmer', action='store_true')
    parser.add_argument('--extract_dacc', action='store_true')
    args = parser.parse_args()
    if not args.extract_kmer and (not args.extract_dacc):
        args.extract_kmer = True
        args.extract_dacc = True
    print('=' * 80)
    print('Feature extraction')
    print('=' * 80)
    print(f'Input FASTA: {args.input_fasta}')
    print(f'Extract k-mer: {args.extract_kmer} (k={args.k})')
    print(f'Extract DACC: {args.extract_dacc} (lag={args.lag})')
    print('=' * 80)
    if not os.path.exists(args.input_fasta):
        raise FileNotFoundError(f'Input file not found: {args.input_fasta}')
    if args.extract_kmer:
        print('\n[1/2] Extracting k-mer features...')
        kw = {'order': 'ACGT'}
        kmer_encoding = utils.Kmer(args.input_fasta, k=args.k, type='RNA', upto=False, normalize=True, **kw)
        if args.kmer_path is None:
            base_name = os.path.splitext(os.path.basename(args.input_fasta))[0]
            args.kmer_path = f'{base_name}_kmer_{args.k}.pkl'
        output_dir = os.path.dirname(args.kmer_path) if os.path.dirname(args.kmer_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        with open(args.kmer_path, 'wb') as handle:
            pickle.dump(kmer_encoding, handle)
        print(f'K-mer features saved to: {args.kmer_path}')
        print(f'  Count: {len(kmer_encoding)}')
        if len(kmer_encoding) > 0:
            sample_seq = list(kmer_encoding.keys())[0]
            print(f'  Dimension: {len(kmer_encoding[sample_seq])}')
    if args.extract_dacc:
        print('\n[2/2] Extracting DACC features...')
        my_property_name, my_property_value = load_physicochemical_properties(args.phyche_data)
        fastas = utils.read_nucleotide_sequences(args.input_fasta)
            print(f'  Loaded {len(fastas)} sequences')
        dacc_encoding = extract_dacc_features(fastas, my_property_name, my_property_value, lag=args.lag, kmer=2)
        if args.dacc_path is None:
            base_name = os.path.splitext(os.path.basename(args.input_fasta))[0]
            args.dacc_path = f'{base_name}_dacc_lag{args.lag}.pkl'
        output_dir = os.path.dirname(args.dacc_path) if os.path.dirname(args.dacc_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        with open(args.dacc_path, 'wb') as handle:
            pickle.dump(dacc_encoding, handle)
        print(f'DACC features saved to: {args.dacc_path}')
        print(f'  Count: {len(dacc_encoding)}')
        if len(dacc_encoding) > 0:
            sample_seq = list(dacc_encoding.keys())[0]
            print(f'  Dimension: {len(dacc_encoding[sample_seq])}')
    print('\n' + '=' * 80)
    print('Feature extraction finished.')
    print('=' * 80)
if __name__ == '__main__':
    main()
