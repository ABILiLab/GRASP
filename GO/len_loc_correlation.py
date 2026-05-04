#!/usr/bin/env python3

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def fasta_description_to_length(fasta_path: Path) -> Tuple[Dict[str, int], int]:
    lengths: Dict[str, int] = {}
    n_records = 0
    cur_key: str | None = None
    cur_chunks: List[str] = []

    def flush() -> None:
        nonlocal cur_key, cur_chunks, n_records
        if cur_key is not None:
            seq = ''.join(cur_chunks).upper().replace('U', 'T')
            lengths[cur_key] = len(seq)
            n_records += 1
        cur_key = None
        cur_chunks = []
    with open(fasta_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                flush()
                hdr = line[1:].split()[0]
                parts = hdr.split('|')
                cur_key = '|'.join(parts[:2]) if len(parts) >= 2 else parts[0]
            else:
                if cur_key is None:
                    continue
                cur_chunks.append(line)
        flush()
    return (lengths, n_records)

def run_analysis(predictions_csv: Path, fasta_path: Path, out_csv: Path, out_summary: Path, tag: str) -> pd.DataFrame:
    pred = pd.read_csv(predictions_csv)
    if 'Description' not in pred.columns:
        raise ValueError(f'{predictions_csv} is missing column Description')
    loc_cols = [c for c in pred.columns if c != 'Description']
    if not loc_cols:
        raise ValueError('No localization probability columns found')
    for c in loc_cols:
        pred[c] = pd.to_numeric(pred[c], errors='coerce')
    len_map, n_fasta = fasta_description_to_length(fasta_path)
    pred['seq_len_nt'] = pred['Description'].astype(str).map(len_map)
    n_pred = len(pred)
    missing_len = int(pred['seq_len_nt'].isna().sum())
    usable = pred.dropna(subset=['seq_len_nt']).copy()
    n_use = len(usable)
    rows = []
    k = len(loc_cols)
    for loc in loc_cols:
        sub = usable[['seq_len_nt', loc]].dropna()
        n = len(sub)
        if n < 3:
            rows.append({'location': loc, 'n': n, 'pearson_r': np.nan, 'pearson_p': np.nan, 'spearman_rho': np.nan, 'spearman_p': np.nan})
            continue
        x = sub['seq_len_nt'].to_numpy(dtype=float)
        y = sub[loc].to_numpy(dtype=float)
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y, nan_policy='omit')
        rows.append({'location': loc, 'n': n, 'pearson_r': float(pr), 'pearson_p': float(pp), 'spearman_rho': float(sr), 'spearman_p': float(sp)})
    res = pd.DataFrame(rows)
    res['pearson_p_bonferroni'] = np.minimum(res['pearson_p'] * k, 1.0)
    res['spearman_p_bonferroni'] = np.minimum(res['spearman_p'] * k, 1.0)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    lines = [f'{tag}: sequence length vs predicted probability correlation\n', f'predictions: {predictions_csv}\n', f'fasta: {fasta_path}\n', f'FASTA records: {n_fasta}\n', f'prediction rows: {n_pred}\n', f'rows without matched length: {missing_len}\n', f'rows used for correlation (per column n below):\n', f'localization columns: {k}; Bonferroni multiplier: {k}\n', '\n', res.to_string(index=False), '\n']
    out_summary.write_text(''.join(lines), encoding='utf-8')
    return res

def build_argparser(default_pred: Path, default_fasta: Path, default_tag: str) -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=f'Sequence length vs localization probability correlation ({default_tag})')
    p.add_argument('--predictions_csv', type=Path, default=default_pred)
    p.add_argument('--fasta', type=Path, default=default_fasta)
    p.add_argument('--out_csv', type=Path, default=here / f'{default_tag.lower()}_len_loc_correlation.csv')
    p.add_argument('--out_summary', type=Path, default=here / f'{default_tag.lower()}_len_loc_correlation_summary.txt')
    p.add_argument('--tag', type=str, default=default_tag)
    return p

def main_preset(default_pred: Path, default_fasta: Path, default_tag: str) -> None:
    parser = build_argparser(default_pred, default_fasta, default_tag)
    args = parser.parse_args()
    run_analysis(args.predictions_csv, args.fasta, args.out_csv, args.out_summary, args.tag)
    print(f'Wrote: {args.out_csv}')
    print(f'Wrote: {args.out_summary}')
