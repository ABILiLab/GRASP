#!/usr/bin/env python3

import argparse
import os
import sys
from collections import defaultdict

_ad = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ad not in sys.path:
    sys.path.insert(0, _ad)
from _paths import general_tracker_root, tracker_data

def load_benchmark_genes(benchmark_file):
    benchmark_genes = set()
    with open(benchmark_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                parts = line[1:].split('|')
                if len(parts) >= 3:
                    rna_symbol = parts[2].strip()
                    if rna_symbol:
                        benchmark_genes.add(rna_symbol.upper())
    return benchmark_genes

def load_transcripts_with_gene_mapping(fasta_file):
    gene_to_enst = defaultdict(list)
    enst_data = {}
    current_enst = None
    current_header = None
    current_seq = []
    current_ensg_id = None
    current_gene_symbol = None
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_enst and current_seq:
                    enst_data[current_enst] = {'header': current_header, 'seq': ''.join(current_seq), 'gene_symbol': current_gene_symbol, 'ensg_id': current_ensg_id}
                    if current_gene_symbol:
                        gene_to_enst[current_gene_symbol.upper()].append(current_enst)
                current_header = line
                header_parts = line[1:].split('|')
                if header_parts:
                    current_enst = header_parts[0].strip()
                    current_ensg_id = header_parts[1].strip() if len(header_parts) > 1 else ''
                    current_gene_symbol = header_parts[5].strip() if len(header_parts) > 5 else ''
                    current_seq = []
            elif current_enst:
                current_seq.append(line)
        if current_enst and current_seq:
            enst_data[current_enst] = {'header': current_header, 'seq': ''.join(current_seq), 'gene_symbol': current_gene_symbol, 'ensg_id': current_ensg_id}
            if current_gene_symbol:
                gene_to_enst[current_gene_symbol.upper()].append(current_enst)
    return (gene_to_enst, enst_data)

def main():
    t = general_tracker_root()
    parser = argparse.ArgumentParser(description='Remove transcripts whose genes appear in a benchmark FASTA (match by RNA symbol)')
    parser.add_argument('--benchmark_fasta', type=str, default=tracker_data('benchmark.fasta'))
    parser.add_argument('--transcripts_fasta', type=str, default=os.path.join(t, 'mRNA_GO', 'lncRNA', 'transcripts.fa'))
    parser.add_argument('--output_fasta', type=str, default=os.path.join(t, 'mRNA_GO', 'lncRNA', 'sequences.fasta'))
    args = parser.parse_args()
    benchmark_file = args.benchmark_fasta
    transcripts_fa = args.transcripts_fasta
    output_file = args.output_fasta
    print('=' * 60)
    print('Filter full transcript FASTA: exclude genes present in benchmark')
    print('Matching by RNA symbol')
    print('=' * 60)
    print('\n[1/3] Loading benchmark FASTA...')
    benchmark_genes = load_benchmark_genes(benchmark_file)
    print(f'    Benchmark RNA symbols: {len(benchmark_genes)}')
    print('\n[2/3] Loading transcript FASTA...')
    gene_to_enst, enst_data = load_transcripts_with_gene_mapping(transcripts_fa)
    print(f'    Loaded {len(enst_data)} transcripts')
    print(f'    Gene symbol index size: {len(gene_to_enst)}')
    print('\n[3/3] Matching genes and marking transcripts to exclude...')
    excluded_enst_ids = set()
    matched_genes = set()
    unmatched_genes = set()
    for gene_symbol in benchmark_genes:
        if gene_symbol in gene_to_enst:
            matched_genes.add(gene_symbol)
            for enst_id in gene_to_enst[gene_symbol]:
                excluded_enst_ids.add(enst_id)
        else:
            unmatched_genes.add(gene_symbol)
    print(f'    Matched benchmark genes: {len(matched_genes)}')
    print(f'    Unmatched benchmark symbols: {len(unmatched_genes)} (missing in GENCODE or symbol mismatch)')
    print(f'    Transcripts to exclude (in benchmark): {len(excluded_enst_ids)}')
    print(f'    Transcripts to keep (not in benchmark): {len(enst_data) - len(excluded_enst_ids)}')
    print('\nWriting output FASTA...')
    kept_count = 0
    excluded_count = 0
    with open(output_file, 'w') as outfile:
        for enst_id, data in enst_data.items():
            if enst_id not in excluded_enst_ids:
                seq = data['seq']
                gene_symbol = data['gene_symbol']
                ensg_id = data.get('ensg_id', '')
                new_header = f'>{enst_id}|{ensg_id}|{gene_symbol}|'
                outfile.write(new_header + '\n')
                outfile.write(seq + '\n')
                kept_count += 1
            else:
                excluded_count += 1
    print(f'\n' + '=' * 60)
    print('Done.')
    print('=' * 60)
    print(f'Kept transcripts: {kept_count} (not in benchmark)')
    print(f'Excluded transcripts: {excluded_count} (in benchmark)')
    print(f'Output: {output_file}')
    print('=' * 60)
    if unmatched_genes and len(unmatched_genes) <= 20:
        print(f'\nUnmatched genes (in benchmark but not in GENCODE), up to 20:')
        for gene in sorted(list(unmatched_genes))[:20]:
            print(f'  - {gene}')
    elif unmatched_genes:
        print(f'\nSample unmatched genes (in benchmark but not in GENCODE), first 10:')
        for gene in sorted(list(unmatched_genes))[:10]:
            print(f'  - {gene}')
        print(f'  ... and {len(unmatched_genes) - 10} more unmatched')
if __name__ == '__main__':
    main()
