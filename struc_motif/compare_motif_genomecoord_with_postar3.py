#!/usr/bin/env python3

from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

@dataclass
class OccRec:
    occ_id: int
    location: str
    region_type: str
    motif_seq: str
    sequence_id: str
    gene_symbol: str
    chrom: str
    blocks: List[Tuple[int, int]]
    hit_count: int = 0
    first_rbp: str = ''
    first_peak_id: str = ''

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--occ_csv', required=True)
    ap.add_argument('--postar3_txt', required=True)
    ap.add_argument('--output_dir', default='')
    ap.add_argument('--bin_size', type=int, default=100000)
    return ap.parse_args()

def parse_positions_to_blocks(pos_str: str) -> List[Tuple[int, int]]:
    vals: List[int] = []
    for x in (pos_str or '').split(','):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(int(x))
        except Exception:
            continue
    if not vals:
        return []
    vals = sorted(set(vals))
    blocks: List[Tuple[int, int]] = []
    st = vals[0]
    prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        blocks.append((st, prev))
        st = v
        prev = v
    blocks.append((st, prev))
    return blocks

def overlap_len(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0, hi - lo + 1)

def main() -> None:
    args = parse_args()
    occ_csv = Path(args.occ_csv)
    postar = Path(args.postar3_txt)
    out_dir = Path(args.output_dir) if args.output_dir else occ_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if not occ_csv.exists():
        raise FileNotFoundError(f'missing: {occ_csv}')
    if not postar.exists():
        raise FileNotFoundError(f'missing: {postar}')
    occs: List[OccRec] = []
    chrom_bins: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    with occ_csv.open('r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r, start=1):
            chrom = (row.get('GenomeChrom') or '').strip()
            if not chrom:
                continue
            blocks = parse_positions_to_blocks(row.get('GenomePositions') or '')
            if not blocks:
                try:
                    gs = int(row.get('GenomeStart') or 0)
                    ge = int(row.get('GenomeEnd') or 0)
                except Exception:
                    gs, ge = (0, 0)
                if gs > 0 and ge >= gs:
                    blocks = [(gs, ge)]
                else:
                    continue
            rec = OccRec(occ_id=i, location=(row.get('Location') or '').strip(), region_type=(row.get('RegionType') or '').strip(), motif_seq=(row.get('MotifSeq') or '').strip(), sequence_id=(row.get('SequenceID') or '').strip(), gene_symbol=(row.get('GeneSymbol') or '').strip(), chrom=chrom, blocks=blocks)
            occs.append(rec)
            idx = len(occs) - 1
            for b in blocks:
                b0 = b[0] // args.bin_size
                b1 = b[1] // args.bin_size
                for bi in range(b0, b1 + 1):
                    chrom_bins[chrom][bi].append(idx)
    total_peaks = 0
    hit_peaks = 0
    hit_occ_ids: Set[int] = set()
    peak_hits_rows: List[dict] = []
    with postar.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            cols = line.split('\t')
            if len(cols) < 10:
                continue
            chrom = cols[0]
            try:
                p_start = int(cols[1]) + 1
                p_end = int(cols[2])
            except Exception:
                continue
            if p_end < p_start:
                continue
            total_peaks += 1
            peak_id = cols[3]
            strand = cols[4]
            rbp = cols[5]
            assay = cols[6]
            cell = cols[7]
            cand: Set[int] = set()
            b0 = p_start // args.bin_size
            b1 = p_end // args.bin_size
            cidx = chrom_bins.get(chrom, {})
            for bi in range(b0, b1 + 1):
                for idx in cidx.get(bi, []):
                    cand.add(idx)
            peak_occ_n = 0
            for idx in cand:
                rec = occs[idx]
                hit = False
                for blk in rec.blocks:
                    if overlap_len(blk, (p_start, p_end)) > 0:
                        hit = True
                        break
                if not hit:
                    continue
                peak_occ_n += 1
                rec.hit_count += 1
                if not rec.first_rbp:
                    rec.first_rbp = rbp
                    rec.first_peak_id = peak_id
                hit_occ_ids.add(idx)
            if peak_occ_n > 0:
                hit_peaks += 1
                peak_hits_rows.append({'PeakChrom': chrom, 'PeakStart': p_start, 'PeakEnd': p_end, 'PeakId': peak_id, 'PeakStrand': strand, 'RBP': rbp, 'Assay': assay, 'Cell': cell, 'OverlappedOccurrenceCount': peak_occ_n})
    total_occ = len(occs)
    hit_occ = len(hit_occ_ids)
    occ_hits_out = out_dir / 'motif_postar3_occurrence_hits.csv'
    with occ_hits_out.open('w', encoding='utf-8', newline='') as f:
        fields = ['OccurrenceId', 'Location', 'RegionType', 'MotifSeq', 'SequenceID', 'GeneSymbol', 'GenomeChrom', 'GenomeBlocks', 'HitCount', 'FirstRBP', 'FirstPeakId', 'Overlapped']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, rec in enumerate(occs):
            w.writerow({'OccurrenceId': rec.occ_id, 'Location': rec.location, 'RegionType': rec.region_type, 'MotifSeq': rec.motif_seq, 'SequenceID': rec.sequence_id, 'GeneSymbol': rec.gene_symbol, 'GenomeChrom': rec.chrom, 'GenomeBlocks': ';'.join((f'{a}-{b}' for a, b in rec.blocks)), 'HitCount': rec.hit_count, 'FirstRBP': rec.first_rbp, 'FirstPeakId': rec.first_peak_id, 'Overlapped': 1 if i in hit_occ_ids else 0})
    peak_hits_out = out_dir / 'motif_postar3_peak_hits.csv'
    with peak_hits_out.open('w', encoding='utf-8', newline='') as f:
        if peak_hits_rows:
            fields = list(peak_hits_rows[0].keys())
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(peak_hits_rows)
        else:
            w = csv.writer(f)
            w.writerow(['PeakChrom', 'PeakStart', 'PeakEnd', 'PeakId', 'PeakStrand', 'RBP', 'Assay', 'Cell', 'OverlappedOccurrenceCount'])
    summary_out = out_dir / 'motif_postar3_overlap_summary.csv'
    with summary_out.open('w', encoding='utf-8', newline='') as f:
        fields = ['Metric', 'Value']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({'Metric': 'occurrences_total', 'Value': total_occ})
        w.writerow({'Metric': 'occurrences_overlapped', 'Value': hit_occ})
        w.writerow({'Metric': 'occurrence_overlap_rate', 'Value': f'{(hit_occ / total_occ if total_occ else 0.0):.6f}'})
        w.writerow({'Metric': 'postar3_peaks_total', 'Value': total_peaks})
        w.writerow({'Metric': 'postar3_peaks_overlapped', 'Value': hit_peaks})
        w.writerow({'Metric': 'postar3_peak_overlap_rate', 'Value': f'{(hit_peaks / total_peaks if total_peaks else 0.0):.6f}'})
    print('=' * 80)
    print('Done motif genome coord vs POSTAR3 overlap')
    print(f'occ_csv: {occ_csv}')
    print(f'postar3_txt: {postar}')
    print(f'occurrences_total: {total_occ}')
    print(f'occurrences_overlapped: {hit_occ}')
    print(f'postar3_peaks_total: {total_peaks}')
    print(f'postar3_peaks_overlapped: {hit_peaks}')
    print(f'summary_csv: {summary_out}')
    print(f'occurrence_hits_csv: {occ_hits_out}')
    print(f'peak_hits_csv: {peak_hits_out}')
    print('=' * 80)
if __name__ == '__main__':
    main()
