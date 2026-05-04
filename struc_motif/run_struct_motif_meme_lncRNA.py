#!/usr/bin/env python3

import argparse
import csv
import math
import os
import pickle
import sys
import shlex
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
from Bio import SeqIO

_GO_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GO'))
_DEFAULT_LNC_PREDICT = os.path.join(_GO_DIR, 'lncRNA_predict.py')
_DEFAULT_MRNA_PREDICT = os.path.join(_GO_DIR, 'mRNA_predict.py')
_AD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AD not in sys.path:
    sys.path.insert(0, _AD)
from _paths import meme_suite_dir

DEFAULT_MEME_SUITE_DIR = meme_suite_dir()

@dataclass
class RegionOccurrence:
    transcript_id: str
    location: str
    confidence: float
    region_type: str
    region_seq: str
    region_dotbracket: str
    region_len: int
    gc_content: float
    indices_0: Tuple[int, ...]
    recurrence: int = 1
    importance_score: float = 0.0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rna_type', choices=['lncRNA', 'mRNA', 'custom'], default='lncRNA')
    parser.add_argument('--predict_script', default='')
    parser.add_argument('--input_fasta', required=True)
    parser.add_argument('--folding_pkl', required=True)
    parser.add_argument('--prediction_csv', default='')
    parser.add_argument('--run_prediction', action='store_true')
    parser.add_argument('--predict_extra_args', default='')
    parser.add_argument('--confidence_threshold', type=float, default=0.8)
    parser.add_argument('--region_rank', choices=['frequency', 'none'], default='none')
    parser.add_argument('--top_percent', type=float, default=0.2)
    parser.add_argument('--min_region_len', type=int, default=6)
    parser.add_argument('--max_region_len', type=int, default=80)
    parser.add_argument('--position_base', type=int, choices=[0, 1], default=0)
    parser.add_argument('--meme_suite_dir', default=DEFAULT_MEME_SUITE_DIR)
    parser.add_argument('--meme_bin', default='')
    parser.add_argument('--run_meme', action='store_true')
    parser.add_argument('--meme_nmotifs', type=int, default=5)
    parser.add_argument('--meme_minw', type=int, default=6)
    parser.add_argument('--meme_maxw', type=int, default=20)
    parser.add_argument('--meme_mod', choices=['oops', 'zoops', 'anr'], default='zoops')
    parser.add_argument('--meme_objfun', default='')
    parser.add_argument('--meme_evt', type=float, default=None)
    parser.add_argument('--meme_time', type=float, default=None)
    parser.add_argument('--meme_maxsize', type=int, default=None)
    parser.add_argument('--meme_searchsize', type=int, default=None)
    parser.add_argument('--meme_neg', default='')
    parser.add_argument('--meme_bfile', default='')
    parser.add_argument('--meme_markov_order', type=int, default=None)
    parser.add_argument('--meme_seed', type=int, default=None)
    parser.add_argument('--meme_brief', type=int, default=None)
    parser.add_argument('--meme_maxsites', type=int, default=500000)
    parser.add_argument('--meme_text_only', action='store_true')
    parser.add_argument('--meme_html_output', action='store_true')
    parser.add_argument('--meme_nostatus', action='store_true')
    parser.add_argument('--meme_extra_args', default='')
    parser.add_argument('--enrichment_mode', choices=['none', 'ratio', 'hypergeom'], default='ratio')
    parser.add_argument('--enrichment_max_motifs', type=int, default=5000)
    parser.add_argument('--enrichment_progress_interval', type=int, default=2000)
    parser.add_argument('--hypergeom_min_k', type=int, default=2)
    parser.add_argument('--hypergeom_min_ratio', type=float, default=1.0)
    parser.add_argument('--hypergeom_top_ratio_per_group', type=float, default=0.1)
    parser.add_argument('--output_dir', required=True)
    parser.set_defaults(meme_text_only=True)
    return parser.parse_args()

def normalize_seq_key(s) -> str:
    if s is None:
        return ''
    if isinstance(s, bytes):
        s = s.decode('ascii', errors='replace')
    return str(s).upper().replace('T', 'U').strip()

def foldings_dict_normalized(raw: dict) -> dict:
    out: dict = {}
    for k, v in raw.items():
        nk = normalize_seq_key(k)
        if nk:
            out[nk] = v
    return out

def read_fasta_desc_seq(fasta_path: str) -> Dict[str, str]:
    desc_to_seq: Dict[str, str] = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        parts = str(record.description).split('|')
        if len(parts) >= 2:
            desc = '|'.join(parts[:2])
        else:
            desc = parts[0] if parts else ''
        desc_to_seq[desc] = normalize_seq_key(str(record.seq))
    return desc_to_seq

def resolve_meme_executable(meme_suite_dir: str, meme_bin: str) -> str:
    if meme_bin.strip():
        return meme_bin.strip()
    for sub in ('bin', os.path.join('install', 'bin')):
        cand = os.path.join(meme_suite_dir, sub, 'meme')
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return 'meme'

def build_meme_env(meme_suite_dir: str) -> dict:
    env = os.environ.copy()
    prefixes = []
    for sub in ('bin', os.path.join('install', 'bin')):
        p = os.path.join(meme_suite_dir, sub)
        if os.path.isdir(p):
            prefixes.append(p)
    if prefixes:
        env['PATH'] = ':'.join(prefixes) + ':' + env.get('PATH', '')
    return env

def choose_predict_script(args: argparse.Namespace) -> str:
    if args.predict_script:
        return args.predict_script
    if args.rna_type == 'lncRNA':
        return _DEFAULT_LNC_PREDICT
    if args.rna_type == 'mRNA':
        return _DEFAULT_MRNA_PREDICT
    raise ValueError('rna_type=custom requires --predict_script')

def run_prediction_if_needed(args: argparse.Namespace, out_dir: str) -> str:
    if args.prediction_csv:
        if not os.path.exists(args.prediction_csv):
            raise FileNotFoundError(f'prediction_csv not found: {args.prediction_csv}')
        return args.prediction_csv
    if not args.run_prediction:
        raise ValueError('Provide --prediction_csv or use --run_prediction')
    predict_script = choose_predict_script(args)
    if not os.path.exists(predict_script):
        raise FileNotFoundError(f'Predict script not found: {predict_script}')
    pred_out_dir = os.path.join(out_dir, 'prediction_results')
    os.makedirs(pred_out_dir, exist_ok=True)
    cmd = ['python', predict_script, '--input_fasta', args.input_fasta, '--folding_path', args.folding_pkl, '--output_dir', pred_out_dir]
    if args.predict_extra_args.strip():
        cmd.extend(args.predict_extra_args.strip().split())
    print(f"[{datetime.now()}] Running predict: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    pred_csv = os.path.join(pred_out_dir, 'predictions.csv')
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f'Prediction finished but predictions.csv missing: {pred_csv}')
    return pred_csv

def get_pair_map(dot_bracket: str) -> Dict[int, int]:
    stack: List[int] = []
    pair_map: Dict[int, int] = {}
    for i, ch in enumerate(dot_bracket):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if not stack:
                continue
            j = stack.pop()
            pair_map[i] = j
            pair_map[j] = i
    return pair_map

def extract_loop_regions(dot_bracket: str) -> List[List[int]]:
    regions: List[List[int]] = []
    i = 0
    n = len(dot_bracket)
    while i < n:
        if dot_bracket[i] == '.':
            s = i
            while i < n and dot_bracket[i] == '.':
                i += 1
            regions.append(list(range(s, i)))
        else:
            i += 1
    return regions

def extract_stem_regions(dot_bracket: str) -> List[List[int]]:
    pair_map = get_pair_map(dot_bracket)
    visited: set = set()
    stems: List[List[int]] = []
    for i, ch in enumerate(dot_bracket):
        if ch != '(' or i in visited or i not in pair_map:
            continue
        j = pair_map[i]
        left = i
        right = j
        left_arm: List[int] = []
        right_arm: List[int] = []
        while left < right and left in pair_map and (pair_map[left] == right):
            left_arm.append(left)
            right_arm.append(right)
            visited.add(left)
            visited.add(right)
            left += 1
            right -= 1
        region = sorted(left_arm + right_arm)
        if region:
            stems.append(region)
    return stems

def region_sequence(sequence: str, indices: Sequence[int]) -> str:
    return ''.join((sequence[i] for i in indices if 0 <= i < len(sequence)))

def format_region_positions(indices_0: Tuple[int, ...], base: int) -> Tuple[int, int, str]:
    if not indices_0:
        return (0, 0, '')
    ordered = tuple(sorted(indices_0))
    lo, hi = (ordered[0], ordered[-1])
    if base == 0:
        start, end = (lo, hi + 1)
        idx_str = ','.join((str(i) for i in ordered))
    else:
        start, end = (lo + 1, hi + 1)
        idx_str = ','.join((str(i + 1) for i in ordered))
    return (start, end, idx_str)

def gc_ratio(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum((1 for c in seq if c in {'G', 'C'}))
    return gc / len(seq)

def enrichment_ratio_value(k: int, n: int, N: int, M: int) -> float:
    if N <= 0 or M <= 0 or n <= 0:
        return float('nan')
    exp = n / M
    if exp <= 0:
        return float('nan')
    return k / N / exp

def hypergeom_right_tail(k: int, M: int, n: int, N: int) -> float:
    if min(k, M, n, N) < 0:
        return 1.0
    if n > M or N > M:
        return 1.0
    max_k = min(n, N)
    if k > max_k:
        return 1.0
    denom = math.comb(M, N)
    if denom == 0:
        return 1.0
    s = 0.0
    for x in range(k, max_k + 1):
        s += math.comb(n, x) * math.comb(M - n, N - x) / denom
    return min(1.0, max(0.0, s))

def sanitize_name(s: str) -> str:
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() or ch in {'_', '-', '.'} else '_')
    return ''.join(out)

def save_fasta(records: List[Tuple[str, str]], fasta_path: str) -> None:
    with open(fasta_path, 'w', encoding='utf-8') as f:
        for rid, seq in records:
            f.write(f'>{rid}\n{seq}\n')

def build_meme_command(args: argparse.Namespace, meme_bin: str, input_fasta: str, output_dir: str) -> List[str]:
    cmd: List[str] = [meme_bin, input_fasta, '-rna', '-oc', output_dir, '-nmotifs', str(args.meme_nmotifs), '-minw', str(args.meme_minw), '-maxw', str(args.meme_maxw), '-mod', args.meme_mod]
    if args.meme_text_only and (not args.meme_html_output):
        cmd.append('-text')
    if args.meme_objfun.strip():
        cmd.extend(['-objfun', args.meme_objfun.strip()])
    if args.meme_evt is not None:
        cmd.extend(['-evt', str(args.meme_evt)])
    if args.meme_time is not None:
        cmd.extend(['-time', str(args.meme_time)])
    if args.meme_maxsize is not None:
        cmd.extend(['-maxsize', str(args.meme_maxsize)])
    if args.meme_searchsize is not None:
        cmd.extend(['-searchsize', str(args.meme_searchsize)])
    if args.meme_neg.strip():
        cmd.extend(['-neg', args.meme_neg.strip()])
    if args.meme_bfile.strip():
        cmd.extend(['-bfile', args.meme_bfile.strip()])
    if args.meme_markov_order is not None:
        cmd.extend(['-markov_order', str(args.meme_markov_order)])
    if args.meme_seed is not None:
        cmd.extend(['-seed', str(args.meme_seed)])
    if args.meme_brief is not None:
        cmd.extend(['-brief', str(args.meme_brief)])
    if args.meme_maxsites is not None and args.meme_maxsites > 0:
        cmd.extend(['-maxsites', str(args.meme_maxsites)])
    if args.meme_nostatus:
        cmd.append('-nostatus')
    if args.meme_extra_args.strip():
        cmd.extend(shlex.split(args.meme_extra_args))
    return cmd

def run_meme_on_fasta(args: argparse.Namespace, meme_bin: str, input_fasta: str, output_dir: str) -> Tuple[bool, str]:
    cmd = build_meme_command(args, meme_bin, input_fasta, output_dir)
    env = build_meme_env(args.meme_suite_dir)
    try:
        subprocess.run(cmd, check=True, env=env)
        return (True, 'ok')
    except Exception as e:
        return (False, f'{type(e).__name__}: {e}')

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print('=' * 80)
    print('Structural motif pipeline started')
    print(f'Start: {datetime.now()}')
    print(f'Output directory: {args.output_dir}')
    print(f'Region ranking: {args.region_rank}' + (f', top_percent={args.top_percent}' if args.region_rank != 'none' else ' (all stems/loops, no top cutoff)'))
    print(f'Coordinates position_base={args.position_base} (0=half-open 0-based, 1=closed 1-based)')
    meme_exe = ''
    if args.run_meme:
        meme_exe = resolve_meme_executable(args.meme_suite_dir, args.meme_bin)
        print(f'MEME binary: {meme_exe} (suite_dir={args.meme_suite_dir})')
        if meme_exe == 'meme' and (not shutil.which('meme')):
            print('Warning: meme not on PATH and not under meme_suite_dir (bin/ or install/bin/). Install MEME and/or pass --meme_bin.')
        elif meme_exe != 'meme' and (not (os.path.isfile(meme_exe) and os.access(meme_exe, os.X_OK))):
            print(f'Warning: MEME path not executable: {meme_exe}')
    else:
        print('MEME disabled (no --run_meme); skipping meme_inputs/meme_outputs')
    print('=' * 80)
    pred_csv = run_prediction_if_needed(args, args.output_dir)
    print(f'Using predictions: {pred_csv}')
    if not os.path.exists(args.folding_pkl):
        raise FileNotFoundError(f'folding pkl not found: {args.folding_pkl}')
    foldings_raw = pickle.load(open(args.folding_pkl, 'rb'))
    foldings = foldings_dict_normalized(foldings_raw)
    desc_to_seq = read_fasta_desc_seq(args.input_fasta)
    print(f'Folding entries (normalized keys): {len(foldings)}, FASTA transcripts: {len(desc_to_seq)}')
    print(f"Enrichment mode: {args.enrichment_mode} (max motifs per group: {args.enrichment_max_motifs or 'unlimited'})")
    pred_df = pd.read_csv(pred_csv)
    if 'Description' not in pred_df.columns:
        raise ValueError('predictions.csv missing Description column')
    location_cols = [c for c in pred_df.columns if c != 'Description']
    if not location_cols:
        raise ValueError('predictions.csv has no localization probability columns')
    long_rows = []
    for _, row in pred_df.iterrows():
        tid = str(row['Description'])
        for loc in location_cols:
            prob = float(row[loc])
            if prob >= args.confidence_threshold:
                long_rows.append((tid, loc, prob))
    hc_df = pd.DataFrame(long_rows, columns=['Description', 'Location', 'Confidence'])
    hc_path = os.path.join(args.output_dir, 'high_confidence_transcripts.csv')
    hc_df.to_csv(hc_path, index=False)
    print(f'High-confidence rows: {len(hc_df)}, file: {hc_path}')
    occurrences: List[RegionOccurrence] = []
    motif_to_transcripts_global: Dict[str, set] = defaultdict(set)
    location_transcripts: Dict[str, set] = defaultdict(set)
    skip_no_seq = 0
    skip_no_fold = 0
    skip_dot_bad = 0
    fold_ok_but_len_filtered = 0
    for row in hc_df.itertuples(index=False):
        desc = row.Description
        location = row.Location
        conf = float(row.Confidence)
        seq = desc_to_seq.get(desc, '')
        if not seq:
            skip_no_seq += 1
            continue
        if seq not in foldings:
            skip_no_fold += 1
            continue
        dot = foldings[seq][0]
        if not isinstance(dot, str) or len(dot) != len(seq):
            skip_dot_bad += 1
            continue
        location_transcripts[location].add(desc)
        loop_regions = extract_loop_regions(dot)
        stem_regions = extract_stem_regions(dot)
        row_added = False
        for rtype, regs in (('loop', loop_regions), ('stem', stem_regions)):
            for idx_list in regs:
                if not args.min_region_len <= len(idx_list) <= args.max_region_len:
                    continue
                idx_sorted = tuple(sorted(idx_list))
                subseq = region_sequence(seq, idx_sorted)
                if len(subseq) < args.min_region_len:
                    continue
                gc = gc_ratio(subseq)
                occ = RegionOccurrence(transcript_id=desc, location=location, confidence=conf, region_type=rtype, region_seq=subseq, region_dotbracket=region_sequence(dot, idx_sorted), region_len=len(subseq), gc_content=gc, indices_0=idx_sorted)
                occurrences.append(occ)
                row_added = True
                motif_key = f'{rtype}:{subseq}'
                motif_to_transcripts_global[motif_key].add(desc)
        if not row_added:
            fold_ok_but_len_filtered += 1
    if not occurrences:
        diag_path = os.path.join(args.output_dir, 'structure_extract_diag.txt')
        lines = ['Stem/loop extraction diagnostic (no occurrences)\n', f'High-confidence rows: {len(hc_df)}\n', f'skipped_no_fasta_seq: {skip_no_seq}\n', f'skipped_no_folding_key: {skip_no_fold}\n', f'skipped_bad_dotbracket: {skip_dot_bad}\n', f'folding_ok_but_no_region_in_len_range: {fold_ok_but_len_filtered}\n', f'Current min_region_len={args.min_region_len}, max_region_len={args.max_region_len}\n', '\nSuggestions:\n', '1) Ensure --folding_pkl matches --input_fasta; T/U keys are normalized.\n', '2) If folding_ok_but_no_region_in_len_range is large, try larger --max_region_len (e.g. 200–500).\n', '3) If skipped_no_folding_key is large, verify folding pkl path matches your FASTA batch.\n']
        msg = ''.join(lines)
        print(msg)
        with open(diag_path, 'w', encoding='utf-8') as df:
            df.write(msg)
        raise RuntimeError('No structural regions extracted; see console and ' + diag_path)
    counts = Counter(((o.location, o.region_type, o.region_seq) for o in occurrences))
    for o in occurrences:
        o.recurrence = counts[o.location, o.region_type, o.region_seq]
        o.importance_score = float(o.recurrence)
    occ_csv = os.path.join(args.output_dir, 'region_occurrences.csv')
    with open(occ_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Description', 'Location', 'Confidence', 'RegionType', 'RegionSeq', 'RegionDotBracket', 'RegionLen', 'RegionStart', 'RegionEnd', 'RegionIndices', 'GC', 'Recurrence', 'ImportanceScore'])
        pb = args.position_base
        for o in occurrences:
            rs, re, ridx = format_region_positions(o.indices_0, pb)
            writer.writerow([o.transcript_id, o.location, f'{o.confidence:.6f}', o.region_type, o.region_seq, o.region_dotbracket, o.region_len, rs, re, ridx, f'{o.gc_content:.6f}', o.recurrence, f'{o.importance_score:.8f}'])
    grouped: Dict[Tuple[str, str], List[RegionOccurrence]] = defaultdict(list)
    for o in occurrences:
        grouped[o.location, o.region_type].append(o)
    selected_records: List[dict] = []
    meme_inputs_dir = os.path.join(args.output_dir, 'meme_inputs')
    meme_out_dir = os.path.join(args.output_dir, 'meme_outputs')
    if args.run_meme:
        os.makedirs(meme_inputs_dir, exist_ok=True)
        os.makedirs(meme_out_dir, exist_ok=True)
    motif_enrichment_rows: List[dict] = []
    all_hc_transcripts = set(hc_df['Description'].astype(str).tolist())
    M = len(all_hc_transcripts)
    for (location, rtype), items in grouped.items():
        items = sorted(items, key=lambda x: (-x.recurrence, -x.region_len, x.region_seq, x.transcript_id))
        if args.region_rank == 'none':
            top_items = items
        else:
            keep_n = max(1, int(len(items) * args.top_percent))
            top_items = items[:keep_n]
        fasta_records = []
        seen = set()
        pb = args.position_base
        for i, it in enumerate(top_items, 1):
            rs, re, _ = format_region_positions(it.indices_0, pb)
            rid = f'{sanitize_name(location)}_{rtype}_{i}_{sanitize_name(it.transcript_id)}_p{rs}_{re}'
            fasta_records.append((rid, it.region_seq))
            key = (location, rtype, it.region_seq)
            if key not in seen:
                _, _, ex_idx = format_region_positions(it.indices_0, pb)
                selected_records.append({'Location': location, 'RegionType': rtype, 'RegionSeq': it.region_seq, 'RegionDotBracket': it.region_dotbracket, 'Recurrence': it.recurrence, 'MeanConfidence': it.confidence, 'ImportanceScore': it.importance_score, 'GC': it.gc_content, 'ExampleDescription': it.transcript_id, 'RegionStart': rs, 'RegionEnd': re, 'RegionIndices': ex_idx})
                seen.add(key)
        if args.run_meme:
            in_fa = os.path.join(meme_inputs_dir, f'{sanitize_name(location)}_{rtype}.fa')
            save_fasta(fasta_records, in_fa)
            this_meme_out = os.path.join(meme_out_dir, f'{sanitize_name(location)}_{rtype}')
            os.makedirs(this_meme_out, exist_ok=True)
            ok, msg = run_meme_on_fasta(args, meme_bin=meme_exe, input_fasta=in_fa, output_dir=this_meme_out)
            print(f"MEME [{location}/{rtype}] -> {('ok' if ok else 'fail')}: {msg}")
        if args.enrichment_mode != 'none':
            this_loc_transcripts = location_transcripts.get(location, set())
            N = len(this_loc_transcripts)
            uniq_motifs = list({f'{rtype}:{x.region_seq}' for x in top_items})
            uniq_motifs.sort(key=lambda mk: len(motif_to_transcripts_global[mk]), reverse=True)
            cap = args.enrichment_max_motifs
            if cap > 0 and len(uniq_motifs) > cap:
                print(f'Enrich [{location}/{rtype}] unique motifs {len(uniq_motifs)}, capped globally to top {cap}')
                uniq_motifs = uniq_motifs[:cap]
            motif_stats = []
            for motif_key in uniq_motifs:
                global_transcripts = motif_to_transcripts_global[motif_key]
                n = len(global_transcripts)
                k = len(global_transcripts & this_loc_transcripts)
                ratio = enrichment_ratio_value(k, n, N, M)
                motif_seq = motif_key.split(':', 1)[1]
                motif_stats.append({'motif_key': motif_key, 'motif_seq': motif_seq, 'n': n, 'k': k, 'ratio': ratio})
            hypergeom_candidate_keys = set()
            if args.enrichment_mode == 'hypergeom':
                prelim = []
                for st in motif_stats:
                    if st['k'] < args.hypergeom_min_k:
                        continue
                    if pd.isna(st['ratio']) or st['ratio'] < args.hypergeom_min_ratio:
                        continue
                    prelim.append(st)
                ratio_cap = args.hypergeom_top_ratio_per_group
                if ratio_cap and ratio_cap > 0:
                    prelim.sort(key=lambda x: x['ratio'], reverse=True)
                    keep_n = max(1, math.ceil(len(prelim) * ratio_cap)) if prelim else 0
                    prelim = prelim[:keep_n]
                hypergeom_candidate_keys = {x['motif_key'] for x in prelim}
                print(f'Hypergeom candidates [{location}/{rtype}]: {len(hypergeom_candidate_keys)}/{len(motif_stats)} (k>={args.hypergeom_min_k}, ratio>={args.hypergeom_min_ratio}, top_ratio={args.hypergeom_top_ratio_per_group})')
            prog_iv = args.enrichment_progress_interval
            for im, st in enumerate(motif_stats, start=1):
                if prog_iv > 0 and im % prog_iv == 0:
                    print(f'Enrichment progress [{location}/{rtype}] {im}/{len(motif_stats)}')
                if args.enrichment_mode == 'hypergeom':
                    if st['motif_key'] in hypergeom_candidate_keys and M > 0:
                        p = hypergeom_right_tail(k=st['k'], M=M, n=st['n'], N=N)
                        tested = 1
                    else:
                        p = float('nan')
                        tested = 0
                else:
                    p = float('nan')
                    tested = 0
                motif_enrichment_rows.append({'Location': location, 'RegionType': rtype, 'MotifSeq': st['motif_seq'], 'k_in_location': st['k'], 'N_location_total': N, 'n_global_with_motif': st['n'], 'M_global_total': M, 'enrichment_ratio': st['ratio'], 'hypergeom_pvalue': p, 'hypergeom_tested': tested})
    selected_df = pd.DataFrame(selected_records).sort_values(['Location', 'RegionType', 'ImportanceScore'], ascending=[True, True, False])
    selected_csv = os.path.join(args.output_dir, 'top20_region_candidates.csv')
    selected_df.to_csv(selected_csv, index=False)
    enrich_df = pd.DataFrame(motif_enrichment_rows)
    if len(enrich_df) > 0:
        if args.enrichment_mode == 'hypergeom':
            enrich_df = enrich_df.sort_values('hypergeom_pvalue', ascending=True)
        elif args.enrichment_mode == 'ratio':
            enrich_df = enrich_df.sort_values('enrichment_ratio', ascending=False, na_position='last')
    enrich_csv = os.path.join(args.output_dir, 'motif_hypergeom_enrichment.csv')
    enrich_df.to_csv(enrich_csv, index=False)
    summary_txt = os.path.join(args.output_dir, 'analysis_summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('Structural motif enrichment summary\n')
        f.write(f'Confidence threshold >= {args.confidence_threshold}\n')
        f.write(f'Region ranking: {args.region_rank}\n')
        if args.region_rank == 'none':
            f.write('Top fraction: unused (all stems/loops to MEME)\n')
        else:
            f.write(f'Top fraction: {args.top_percent}\n')
        f.write(f'High-confidence records: {len(hc_df)}\n')
        f.write(f'Total extracted regions: {len(occurrences)}\n')
        f.write(f'Enrichment mode: {args.enrichment_mode}\n\n')
        if args.enrichment_mode == 'none':
            f.write('Enrichment disabled (--enrichment_mode none)\n')
        elif args.enrichment_mode == 'ratio':
            f.write('Top localization/global enrichment ratios (by enrichment_ratio, top 10):\n')
        else:
            f.write('Localization-specific motifs (hypergeom right-tail p, top 10):\n')
        if len(enrich_df) == 0:
            f.write('  (none)\n')
        else:
            topn = enrich_df.head(10)
            for _, r in topn.iterrows():
                if args.enrichment_mode == 'hypergeom':
                    f.write(f"  - {r['Location']} | {r['RegionType']} | motif={r['MotifSeq']} | p={float(r['hypergeom_pvalue']):.3e} | k={int(r['k_in_location'])}/{int(r['N_location_total'])}\n")
                elif args.enrichment_mode == 'ratio':
                    rr = r['enrichment_ratio']
                    rr_s = f'{float(rr):.3f}' if pd.notna(rr) else 'nan'
                    f.write(f"  - {r['Location']} | {r['RegionType']} | motif={r['MotifSeq']} | ratio={rr_s} | k={int(r['k_in_location'])}/{int(r['N_location_total'])} | n={int(r['n_global_with_motif'])}/M={int(r['M_global_total'])}\n")
            mito_stem = topn[(topn['Location'] == 'Mitochondrion') & (topn['RegionType'] == 'stem')]
            if len(mito_stem) > 0:
                best = mito_stem.iloc[0]
                gc = gc_ratio(str(best['MotifSeq']))
                if args.enrichment_mode == 'hypergeom':
                    f.write(f"\nExample: Mitochondrial stem motif GC-enriched, motif={best['MotifSeq']}, GC={gc:.2%}, hypergeom p={float(best['hypergeom_pvalue']):.3e}.\n")
                elif args.enrichment_mode == 'ratio':
                    rr = best['enrichment_ratio']
                    rr_s = f'{float(rr):.3f}' if pd.notna(rr) else 'nan'
                    f.write(f"\nExample: Mitochondrial stem motif={best['MotifSeq']}, GC={gc:.2%}, enrichment ratio={rr_s}.\n")
    print('=' * 80)
    print('Analysis complete')
    print(f'End: {datetime.now()}')
    print(f'High-confidence CSV: {hc_path}')
    print(f'Region details: {occ_csv}')
    print(f'Top-20 candidates: {selected_csv}')
    print(f'Enrichment CSV: {enrich_csv}')
    print(f'Summary: {summary_txt}')
    print('=' * 80)
if __name__ == '__main__':
    main()
