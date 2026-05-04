import json
import argparse
import os
import sys
from collections import defaultdict

_ad = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ad not in sys.path:
    sys.path.insert(0, _ad)
from _paths import tracker_explain

def parse_bed_file(bed_path: str):
    modsites_by_symbol = defaultdict(list)
    try:
        with open(bed_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 19:
                    continue
                symbols_str = parts[15]
                if not symbols_str or symbols_str == 'na':
                    continue
                sequence_fragment = parts[18] if len(parts) > 18 else ''
                if not sequence_fragment or sequence_fragment == 'na' or len(sequence_fragment) < 3:
                    continue
                mod_type = parts[6] if len(parts) > 6 else 'unknown'
                symbols = [s.strip() for s in symbols_str.split(',')]
                for symbol in symbols:
                    if symbol:
                        modsites_by_symbol[symbol].append({'sequence_fragment': sequence_fragment, 'mod_type': mod_type, 'chr': parts[0], 'start': int(parts[1]), 'end': int(parts[2]), 'strand': parts[5] if len(parts) > 5 else '+'})
    except Exception as e:
        print(f'[WARN] Error parsing BED file: {e}')
    return modsites_by_symbol

def find_modsite_types_from_bed(sequence: str, modsite_indices: list, bed_modsite_info_list: list):
    modsite_types = []
    if not sequence or not modsite_indices or (not bed_modsite_info_list):
        return modsite_types
    sequence_upper = sequence.upper()
    modsite_indices_set = set((int(m) for m in modsite_indices if m is not None))
    for modsite_idx in sorted(modsite_indices_set):
        found_type = None
        found_info = None
        for modsite_info in bed_modsite_info_list:
            fragment = modsite_info.get('sequence_fragment', '').upper()
            if not fragment or len(fragment) < 3:
                continue
            fragment_mid = len(fragment) // 2
            start = 0
            while True:
                pos = sequence_upper.find(fragment, start)
                if pos == -1:
                    break
                fragment_modsite_pos = pos + fragment_mid
                if fragment_modsite_pos == modsite_idx:
                    found_type = modsite_info.get('mod_type', 'unknown')
                    found_info = {'mod_type': found_type, 'sequence_fragment': fragment, 'fragment_start': pos, 'fragment_end': pos + len(fragment), 'fragment_modsite_pos': fragment_modsite_pos, 'chr': modsite_info.get('chr', ''), 'start': modsite_info.get('start', 0), 'end': modsite_info.get('end', 0), 'strand': modsite_info.get('strand', '+')}
                    break
                start = pos + 1
            if found_type:
                break
        if found_type:
            modsite_types.append({'modsite_index': modsite_idx, 'mod_type': found_type, 'genomic_info': found_info})
        else:
            modsite_types.append({'modsite_index': modsite_idx, 'mod_type': 'unknown', 'genomic_info': None})
    return modsite_types

def find_all_modsite_positions_in_sequence(sequence: str, bed_modsite_info_list: list):
    all_modsites = []
    if not sequence or not bed_modsite_info_list:
        return all_modsites
    sequence_upper = sequence.upper()
    seen_positions = set()
    for modsite_info in bed_modsite_info_list:
        fragment = modsite_info.get('sequence_fragment', '').upper()
        if not fragment or len(fragment) < 3:
            continue
        fragment_mid = len(fragment) // 2
        start = 0
        while True:
            pos = sequence_upper.find(fragment, start)
            if pos == -1:
                break
            modsite_pos = pos + fragment_mid
            if modsite_pos not in seen_positions:
                seen_positions.add(modsite_pos)
                all_modsites.append({'modsite_index': modsite_pos, 'mod_type': modsite_info.get('mod_type', 'unknown'), 'genomic_info': {'mod_type': modsite_info.get('mod_type', 'unknown'), 'sequence_fragment': fragment, 'fragment_start': pos, 'fragment_end': pos + len(fragment), 'fragment_modsite_pos': modsite_pos, 'chr': modsite_info.get('chr', ''), 'start': modsite_info.get('start', 0), 'end': modsite_info.get('end', 0), 'strand': modsite_info.get('strand', '+')}})
            start = pos + 1
    all_modsites.sort(key=lambda x: x['modsite_index'])
    return all_modsites

def extract_modsite_types_from_json(json_file: str, bed_file: str, output_file: str=None):
    print(f'Reading JSON: {json_file}')
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rna_symbol = data.get('rna_symbol', 'Unknown')
    sequence = data.get('sequence', '')
    sequence_index = data.get('sequence_index', -1)
    print(f'RNA Symbol: {rna_symbol}')
    print(f'Sequence index: {sequence_index}')
    print(f'Sequence length: {len(sequence)}')
    print(f'\nReading BED: {bed_file}')
    bed_modsites = parse_bed_file(bed_file)
    if rna_symbol not in bed_modsites:
        print(f'[WARN] RNA symbol not found in BED: {rna_symbol}')
        print(f'Sample RNA symbols in BED: {list(bed_modsites.keys())[:10]}...')
        return
    bed_modsite_info_list = bed_modsites[rna_symbol]
    print(f'Found {len(bed_modsite_info_list)} modification-site records in BED')
    print('\nMatching modification sites to sequence positions and types...')
    all_modsites = find_all_modsite_positions_in_sequence(sequence, bed_modsite_info_list)
    print(f'Matched {len(all_modsites)} modification sites on sequence')
    all_covered_modsites = set()
    structures_with_modsites = []
    for struct in data.get('all_structures', []):
        coverage_info = struct.get('coverage_info', {})
        if coverage_info:
            covered_indices = coverage_info.get('covered_modsite_indices', [])
            if covered_indices:
                all_covered_modsites.update(covered_indices)
                structures_with_modsites.append({'rank': struct.get('rank', -1), 'structure_type': struct.get('structure_type', 'unknown'), 'structure_index': struct.get('structure_index', -1), 'importance': struct.get('importance', 0.0), 'covered_modsite_indices': covered_indices})
    for modsite in all_modsites:
        modsite['is_covered'] = modsite['modsite_index'] in all_covered_modsites
    print(f'Of these, {len(all_covered_modsites)} sites overlap substructures')
    print(f'Substructures touching sites: {len(structures_with_modsites)}')
    mod_type_counts = defaultdict(int)
    covered_mod_type_counts = defaultdict(int)
    for item in all_modsites:
        mod_type_counts[item['mod_type']] += 1
        if item['is_covered']:
            covered_mod_type_counts[item['mod_type']] += 1
    print(f'\nModification type counts (all sites):')
    for mod_type, count in sorted(mod_type_counts.items(), key=lambda x: x[1], reverse=True):
        covered_count = covered_mod_type_counts.get(mod_type, 0)
        print(f'  {mod_type}: {count} sites ({covered_count} covered by substructures)')
    mod_type_counts = defaultdict(int)
    for item in modsite_types:
        mod_type_counts[item['mod_type']] += 1
    print(f'\nModification type summary:')
    for mod_type, count in sorted(mod_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f'  {mod_type}: {count} sites')
    output_data = {'sequence_index': sequence_index, 'rna_symbol': rna_symbol, 'sequence_length': len(sequence), 'total_modsites': len(all_modsites), 'total_covered_modsites': len(all_covered_modsites), 'all_modification_sites': all_modsites, 'modification_type_summary': {'all': dict(mod_type_counts), 'covered': dict(covered_mod_type_counts)}, 'structures_covering_modsites': structures_with_modsites}
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        output_dir = os.path.dirname(json_file)
        output_file = os.path.join(output_dir, f'{base_name}_all_modsite_types.json')
    print(f'\nWriting results to: {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    csv_data = []
    for item in all_modsites:
        genomic_info = item.get('genomic_info', {})
        csv_data.append({'Modsite_Index': item['modsite_index'], 'Modification_Type': item['mod_type'], 'Is_Covered': 'Yes' if item.get('is_covered', False) else 'No', 'Sequence_Fragment': genomic_info.get('sequence_fragment', '') if genomic_info else '', 'Fragment_Start': genomic_info.get('fragment_start', '') if genomic_info else '', 'Fragment_End': genomic_info.get('fragment_end', '') if genomic_info else '', 'Chr': genomic_info.get('chr', '') if genomic_info else '', 'Genomic_Start': genomic_info.get('start', '') if genomic_info else '', 'Genomic_End': genomic_info.get('end', '') if genomic_info else '', 'Strand': genomic_info.get('strand', '') if genomic_info else ''})
    csv_file = output_file.replace('.json', '.csv')
    import csv
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
    print(f'Wrote CSV: {csv_file}')
    print('\nDone.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--bed_file', type=str, default=tracker_explain('modifications.bed'))
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()
    if not os.path.exists(args.json_file):
        print(f'[ERROR] JSON file not found: {args.json_file}')
        return
    if not os.path.exists(args.bed_file):
        print(f'[ERROR] BED file not found: {args.bed_file}')
        return
    extract_modsite_types_from_json(args.json_file, args.bed_file, args.output_file)
if __name__ == '__main__':
    main()
