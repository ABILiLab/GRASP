#!/usr/bin/env python3

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def parse_gaf_file(gaf_file):
    gene_go_map = defaultdict(lambda: {'P': set(), 'F': set(), 'C': set()})
    background_genes = set()
    with open(gaf_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('!') or not line:
                continue
            fields = line.split('\t')
            if len(fields) < 15:
                continue
            db_object_id = fields[1]
            go_id = fields[4]
            aspect = fields[8]
            if aspect in ['P', 'F', 'C']:
                gene_go_map[db_object_id][aspect].add(go_id)
                background_genes.add(db_object_id)
    print(f'  Loaded {len(background_genes)} background genes from GAF')
    return (gene_go_map, background_genes)


def hypergeometric_test(query_genes, background_genes, background_go_map, background_go_counts, aspect):
    results = []
    N = len(background_genes)
    query_go_counts = defaultdict(int)
    for gene_id in query_genes:
        if gene_id in background_go_map:
            for go_term in background_go_map[gene_id][aspect]:
                query_go_counts[go_term] += 1
    for go_term, k in query_go_counts.items():
        if go_term not in background_go_counts:
            continue
        K = background_go_counts[go_term][aspect]
        n = len(query_genes)
        if K == 0 or n == 0:
            continue
        pvalue = hypergeom.sf(k - 1, N, K, n)
        expected = K / N * n
        enrichment_ratio = k / expected if expected > 0 else 0
        results.append({'GO_term': go_term, 'Query_count': k, 'Background_count': K, 'Enrichment_ratio': enrichment_ratio, 'Pvalue': pvalue, 'Aspect': aspect})
    return results


def run_go_enrichment(gaf_file, csv_file, filtered_lists_dir, output_dir, threshold):
    os.makedirs(output_dir, exist_ok=True)
    print('=' * 80)
    print('GO enrichment analysis')
    print('=' * 80)
    print('\n1. Loading background genome (GAF)...')
    parse_gaf_file(gaf_file)
    print('\n2. Loading CSV for gene–GO term mapping...')
    df = pd.read_csv(csv_file)
    ensembl_go_map = defaultdict(lambda: {'P': set(), 'F': set(), 'C': set()})
    ensembl_background_genes = set()
    for idx, row in df.iterrows():
        gene_id = str(row['hits']).strip()
        if pd.isna(gene_id) or gene_id == 'nan':
            continue
        ensembl_background_genes.add(gene_id)
        for aspect, col_name in [('P', 'Biological Process (GO)'), ('F', 'Molecular Function (GO)'), ('C', 'Cellular Component (GO)')]:
            go_terms_str = row[col_name]
            if pd.notna(go_terms_str):
                go_terms = [go.strip() for go in str(go_terms_str).split(';') if go.strip().startswith('GO:')]
                ensembl_go_map[gene_id][aspect].update(go_terms)
    print(f'  Loaded {len(ensembl_background_genes)} background genes from CSV')
    background_genes = ensembl_background_genes
    background_go_map = ensembl_go_map
    print('\n3. Building background GO term counts...')
    background_go_counts = defaultdict(lambda: {'P': 0, 'F': 0, 'C': 0})
    for gene_id, go_dict in background_go_map.items():
        for aspect in ['P', 'F', 'C']:
            for go_term in go_dict[aspect]:
                background_go_counts[go_term][aspect] += 1
    print(f'  Distinct GO terms in background: {len(background_go_counts)}')
    print('\n4. Running GO enrichment per localization...')
    location_files = glob.glob(os.path.join(filtered_lists_dir, f'*_threshold_{threshold}.txt'))
    if not location_files:
        print(f'  Warning: no files for threshold {threshold}; trying other thresholds...')
        location_files = glob.glob(os.path.join(filtered_lists_dir, '*_threshold_*.txt'))
    location_files.sort()
    print(f'  Found {len(location_files)} localization list files')
    for location_file in location_files:
        location_name = os.path.basename(location_file).replace(f'_threshold_{threshold}.txt', '')
        print(f'\n  Localization: {location_name}')
        query_genes = set()
        with open(location_file, 'r', encoding='utf-8') as f:
            for line in f:
                gene_id = line.strip()
                if gene_id:
                    query_genes.add(gene_id)
        print(f'    Query genes: {len(query_genes)}')
        query_genes_in_background = query_genes.intersection(background_genes)
        print(f'    Query genes in background: {len(query_genes_in_background)}')
        if len(query_genes_in_background) < 5:
            print(f'    Warning: too few query genes; skipping enrichment')
            continue
        all_results = []
        for aspect, aspect_name in [('P', 'Biological_Process'), ('F', 'Molecular_Function'), ('C', 'Cellular_Component')]:
            results = hypergeometric_test(query_genes_in_background, background_genes, background_go_map, background_go_counts, aspect)
            for r in results:
                r['Aspect_name'] = aspect_name
            all_results.extend(results)
        if all_results:
            results_df = pd.DataFrame(all_results)
            from statsmodels.stats.multitest import multipletests
            for aspect in ['P', 'F', 'C']:
                mask = results_df['Aspect'] == aspect
                if mask.sum() > 0:
                    pvalues = results_df.loc[mask, 'Pvalue'].values
                    _, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh')
                    results_df.loc[mask, 'FDR'] = pvals_corrected
            results_df = results_df.sort_values(['FDR', 'Pvalue', 'Enrichment_ratio'], ascending=[True, True, False])
            output_file = os.path.join(output_dir, f'{location_name}_enrichment_results.csv')
            results_df.to_csv(output_file, index=False)
            print(f'    Saved: {output_file}')
            significant = results_df[results_df['FDR'] < 0.05]
            print(f'    Significant GO terms (FDR < 0.05): {len(significant)}')
            if len(significant) > 0:
                print(f'    Top 5 GO terms by significance:')
                for idx, row in significant.head(5).iterrows():
                    print(f"      {row['GO_term']} ({row['Aspect_name']}): Enrichment={row['Enrichment_ratio']:.2f}, Pvalue={row['Pvalue']:.2e}, FDR={row['FDR']:.2e}")
        else:
            print(f'    No enrichment results')
    print('\n' + '=' * 80)
    print('GO enrichment finished.')
    print(f'Outputs written under: {output_dir}')
    print('=' * 80)


def main():
    import sys
    _ad = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ad not in sys.path:
        sys.path.insert(0, _ad)
    from _paths import go_analyze

    parser = argparse.ArgumentParser(description='GO hypergeometric enrichment (localization gene lists vs background CSV)')
    parser.add_argument('--gaf_file', type=str, default=os.path.join(go_analyze('mRNA', 'Enrichment'), 'goa_human.gaf'))
    parser.add_argument('--gene_go_csv', type=str, default=os.path.join(go_analyze('mRNA', 'Enrichment'), 'gene_hits.csv'))
    parser.add_argument('--filtered_lists_dir', type=str, default=os.path.join(go_analyze('mRNA', 'results_bce_noR', 'filtered_lists')))
    parser.add_argument('--output_dir', type=str, default=os.path.join(go_analyze('mRNA', 'Enrichment'), 'enrichment_results'))
    parser.add_argument('--threshold', type=float, default=0.8)
    args = parser.parse_args()
    run_go_enrichment(args.gaf_file, args.gene_go_csv, args.filtered_lists_dir, args.output_dir, args.threshold)


if __name__ == '__main__':
    main()
