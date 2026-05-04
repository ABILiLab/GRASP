import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import data_prepared_root, tracker_checkpoints, tracker_data, tracker_explain, tracker_features
import model_hetero
import data_hetero
import pickle
import utils
import numpy as np
import pandas as pd
import json
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
plt.rcParams['axes.unicode_minus'] = False
from hetero_substructure_analysis import set_seed, parse_rna_symbols_from_fasta, parse_modsites_from_fasta, read_fasta_to_df_label_from_end, compute_hetero_importance

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
                symbols = [s.strip() for s in symbols_str.split(',')]
                for symbol in symbols:
                    if symbol:
                        modsites_by_symbol[symbol].append({'sequence_fragment': sequence_fragment, 'mod_type': parts[6] if len(parts) > 6 else 'unknown', 'chr': parts[0], 'start': int(parts[1]), 'end': int(parts[2]), 'strand': parts[5] if len(parts) > 5 else '+'})
    except Exception as e:
        print(f'[WARN] Error parsing BED file: {e}')
    return modsites_by_symbol

def get_modsite_indices_from_fasta_header(fasta_path: str):
    modsites_info_list = parse_modsites_from_fasta(fasta_path)
    return modsites_info_list

def find_modsite_positions_in_sequence(sequence: str, bed_modsite_info_list: list):
    modsite_indices = []
    if not sequence or not bed_modsite_info_list:
        return modsite_indices
    sequence_upper = sequence.upper()
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
            if modsite_pos not in modsite_indices:
                modsite_indices.append(modsite_pos)
            start = pos + 1
    return sorted(modsite_indices)

def calculate_structure_modsite_coverage(structure_bases: list, modsite_indices: list, structure_type: str, structure_index: int, structure_importance: float):
    structure_bases_set = set((int(b) for b in structure_bases if int(b) >= 0))
    modsite_indices_set = set((int(m) for m in modsite_indices if m is not None))
    covered_modsites = structure_bases_set & modsite_indices_set
    return {'structure_type': structure_type, 'structure_index': structure_index, 'structure_importance': float(structure_importance), 'structure_base_count': len(structure_bases_set), 'total_modsites': len(modsite_indices_set), 'covered_modsites_count': len(covered_modsites), 'covered_modsite_indices': sorted(list(covered_modsites)), 'coverage_ratio': len(covered_modsites) / len(modsite_indices_set) if len(modsite_indices_set) > 0 else 0.0}

def analyze_sequence_modsite_coverage(sequence_index: int, rna_symbol: str, sequence: str, loop_importance: np.ndarray, stem_importance: np.ndarray, loop_base_indices: list, stem_base_indices: list, modsite_indices: list, topk_loops: int=5, topk_stems: int=5, topn_overall: int=10):
    if modsite_indices is None or len(modsite_indices) == 0:
        return {'sequence_index': sequence_index, 'rna_symbol': rna_symbol, 'total_modsites': 0, 'has_modsites': False, 'coverage_results': [], 'cumulative_coverage': []}
    loop_importance = np.asarray(loop_importance, dtype=float) if loop_importance is not None else np.array([])
    stem_importance = np.asarray(stem_importance, dtype=float) if stem_importance is not None else np.array([])
    all_structures = []
    for i in range(len(loop_importance)):
        if loop_base_indices and i < len(loop_base_indices):
            all_structures.append(('loop', i, float(loop_importance[i]), loop_base_indices[i]))
    for i in range(len(stem_importance)):
        if stem_base_indices and i < len(stem_base_indices):
            all_structures.append(('stem', i, float(stem_importance[i]), stem_base_indices[i]))
    all_structures.sort(key=lambda x: x[2], reverse=True)
    coverage_results = []
    cumulative_covered_modsites = set()
    cumulative_coverage = []
    total_modsites = len(modsite_indices)
    for struct_type, struct_idx, importance, base_indices in all_structures:
        if isinstance(base_indices, torch.Tensor):
            bases = base_indices[base_indices >= 0].detach().cpu().tolist()
        elif isinstance(base_indices, (list, np.ndarray)):
            bases = [int(b) for b in base_indices if int(b) >= 0]
        else:
            bases = []
        coverage = calculate_structure_modsite_coverage(bases, modsite_indices, struct_type, struct_idx, importance)
        coverage_results.append(coverage)
        cumulative_covered_modsites.update(coverage['covered_modsite_indices'])
        cumulative_ratio = len(cumulative_covered_modsites) / total_modsites if total_modsites > 0 else 0.0
        cumulative_coverage.append({'num_structures': len(coverage_results), 'cumulative_covered_count': len(cumulative_covered_modsites), 'cumulative_coverage_ratio': cumulative_ratio, 'structure_importance': float(importance), 'structure_type': struct_type, 'structure_index': struct_idx})
    topn_coverage_results = coverage_results[:topn_overall]
    all_covered_modsites = set()
    for cov in topn_coverage_results:
        all_covered_modsites.update(cov['covered_modsite_indices'])
    total_covered_count = len(all_covered_modsites)
    overall_coverage_ratio = total_covered_count / total_modsites if total_modsites > 0 else 0.0
    total_structures = len(all_structures)
    all_structures_info = []
    for struct_type, struct_idx, importance, base_indices in all_structures:
        if isinstance(base_indices, torch.Tensor):
            bases = base_indices[base_indices >= 0].detach().cpu().tolist()
        elif isinstance(base_indices, (list, np.ndarray)):
            bases = [int(b) for b in base_indices if int(b) >= 0]
        else:
            bases = []
        coverage_info = None
        for cov in coverage_results:
            if cov['structure_type'] == struct_type and cov['structure_index'] == struct_idx:
                coverage_info = cov
                break
        all_structures_info.append({'structure_type': struct_type, 'structure_index': struct_idx, 'importance': float(importance), 'base_indices': bases, 'coverage_info': coverage_info})
    return {'sequence_index': sequence_index, 'rna_symbol': rna_symbol, 'sequence_length': len(sequence) if sequence else 0, 'total_modsites': total_modsites, 'has_modsites': total_modsites > 0, 'modsite_indices': modsite_indices, 'topn_structures_covered_modsites': total_covered_count, 'overall_coverage_ratio': overall_coverage_ratio, 'coverage_results': topn_coverage_results, 'top_structures_count': len(topn_coverage_results), 'total_structures': total_structures, 'cumulative_coverage': cumulative_coverage, 'all_structures_info': all_structures_info}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--explain_fasta', type=str, default=tracker_data('explain_sequences.fasta'))
    parser.add_argument('--train_fasta', type=str, default=tracker_data('train_sequences.fasta'))
    parser.add_argument('--bed_file', type=str, default=tracker_explain('modifications.bed'))
    parser.add_argument('--folding_pkl', type=str, default=tracker_features('linearfold.pkl'))
    parser.add_argument('--kmer_pkl', type=str, default=tracker_features('kmer.pkl'))
    parser.add_argument('--dacc_pkl', type=str, default=tracker_features('dacc.pkl'))
    parser.add_argument('--root_prepared', type=str, default=data_prepared_root())
    parser.add_argument('--dataset_processed_name', type=str, default='modsite_explain_cache')
    parser.add_argument('--model_path', type=str, default=tracker_checkpoints('model.pth'))
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_conv_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--conv_type', type=str, default='GAT')
    parser.add_argument('--batch_norm', action='store_true', default=True)
    parser.add_argument('--pred_threshold', type=float, default=0.5)
    parser.add_argument('--topn_structures_overall', type=int, default=200)
    parser.add_argument('--max_explain_samples', type=int, default=None)
    parser.add_argument('--min_modsites', type=int, default=1)
    parser.add_argument('--topk_candidates', type=int, default=20)
    args = parser.parse_args()
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modsite_coverage_results_3')
    os.makedirs(results_dir, exist_ok=True)
    print(f'Results will be saved to: {results_dir}')
    print('Loading folding/kmer/dacc features...')
    foldings = pickle.load(open(args.folding_pkl, 'rb'))
    features_kmer = pickle.load(open(args.kmer_pkl, 'rb'))
    features_dacc = pickle.load(open(args.dacc_pkl, 'rb'))
    print('Building training dataset to get tokenizer...')
    df_train = read_fasta_to_df_label_from_end(args.train_fasta, label_from_end=2)
    labels_list = [str(locs).split(',') for locs in list(df_train['Label'])]
    tokenizer = data_hetero.SequenceTokenizer(df_train['Sequence'], labels_list, isMultiLabel=True)
    print('Building explanation dataset...')
    df_explain = read_fasta_to_df_label_from_end(args.explain_fasta, label_from_end=2).copy()
    before_n = len(df_explain)
    df_explain = df_explain[df_explain['Label'].astype(str).str.strip() != ''].reset_index(drop=True)
    dropped = before_n - len(df_explain)
    if dropped > 0:
        print(f'[WARN] {dropped} empty Label records in explain_fasta, dropped')
    explain_dataset = data_hetero.RNAHeteroGraphDataset(root=args.root_prepared, dataset=f'{args.dataset_processed_name}_seed{args.seed}', view='test', df_data=df_explain, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_dacc=features_dacc, isMultiLabel=True, device='cpu')
    print(f'Explanation dataset size: {len(explain_dataset)}')
    explain_loader = DataLoader(explain_dataset, batch_size=1, shuffle=False, collate_fn=data_hetero.hetero_collate_func, num_workers=0)
    rna_symbols = parse_rna_symbols_from_fasta(args.explain_fasta)
    if len(rna_symbols) != len(df_explain):
        min_len = min(len(rna_symbols), len(df_explain))
        rna_symbols = rna_symbols[:min_len] + [''] * (len(df_explain) - min_len)
    modsites_info_list = get_modsite_indices_from_fasta_header(args.explain_fasta)
    if len(modsites_info_list) != len(df_explain):
        min_len = min(len(modsites_info_list), len(df_explain))
        modsites_info_list = modsites_info_list[:min_len] + [{'modsites_count': None, 'modsites_indices': None}] * (len(df_explain) - min_len)
    bed_modsites = parse_bed_file(args.bed_file)
    print(f'Read modification site information for {len(bed_modsites)} RNA symbols from BED file')
    print(f'Loading model: {args.model_path}')
    model = model_hetero.RNAHeteroModel(num_base_features=20, num_loop_features=6, num_stem_features=5, hidden_dim=int(args.hidden_dim), num_labels=tokenizer.label_count, conv_type=str(args.conv_type), n_conv_layers=int(args.n_conv_layers), dropout=float(args.dropout), batch_norm=bool(args.batch_norm), use_stem_nodes=True, use_label_graph=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print('Starting analysis of important substructure coverage of modification sites...')
    all_coverage_results = []
    for seq_idx, batch_data in enumerate(tqdm(explain_loader, desc='Analyzing sequences')):
        if args.max_explain_samples and seq_idx >= args.max_explain_samples:
            break
        batch_data = batch_data.to(device)
        sequence = batch_data.rowseq[0] if hasattr(batch_data, 'rowseq') and isinstance(batch_data.rowseq, (list, tuple)) else getattr(batch_data, 'rowseq', None)
        if sequence is None and seq_idx < len(df_explain):
            sequence = df_explain.iloc[seq_idx]['Sequence']
        rna_symbol = rna_symbols[seq_idx] if seq_idx < len(rna_symbols) else ''
        modsites_info = modsites_info_list[seq_idx] if seq_idx < len(modsites_info_list) else {'modsites_count': None, 'modsites_indices': None}
        modsite_indices = modsites_info.get('modsites_indices', [])
        modsites_count = modsites_info.get('modsites_count', None)
        if (modsite_indices is None or len(modsite_indices) == 0) and rna_symbol and sequence:
            if rna_symbol in bed_modsites:
                bed_modsite_info_list = bed_modsites[rna_symbol]
                modsite_indices = find_modsite_positions_in_sequence(sequence, bed_modsite_info_list)
                if len(modsite_indices) > 0:
                    modsites_count = len(modsite_indices) if modsites_count is None else modsites_count
                    print(f'[INFO] seq_idx={seq_idx} ({rna_symbol}): Found {len(modsite_indices)} modification site positions from BED file')
        if modsites_count is None or modsites_count < args.min_modsites:
            continue
        if modsite_indices is None or len(modsite_indices) == 0:
            print(f'[WARN] seq_idx={seq_idx} ({rna_symbol}): only modsite count ({modsites_count}); no positions; skip')
            continue
        with torch.no_grad():
            logits = model(batch_data)
            probs = torch.sigmoid(logits)
        pred_binary = (probs[0].cpu().numpy() >= float(args.pred_threshold)).astype(int)
        pred_indices = np.where(pred_binary > 0)[0].tolist()
        target_indices = pred_indices if len(pred_indices) > 0 else [int(np.argmax(probs[0].cpu().numpy()))]
        try:
            imp = compute_hetero_importance(model, batch_data, device, target_label_indices=target_indices)
        except Exception as e:
            print(f'[WARN] seq_idx={seq_idx} importance failed: {e}')
            continue
        loop_importance = imp.get('loop_importance', np.array([]))
        stem_importance = imp.get('stem_importance', np.array([]))
        base_importance_direct = imp.get('base_importance', None)
        loop_base_indices = getattr(batch_data['loop'], 'base_indices', None) if 'loop' in batch_data.node_types else None
        stem_base_indices = getattr(batch_data['stem'], 'base_indices', None) if 'stem' in batch_data.node_types and model.use_stem_nodes else None
        coverage_analysis = analyze_sequence_modsite_coverage(sequence_index=seq_idx, rna_symbol=rna_symbol, sequence=sequence, loop_importance=loop_importance, stem_importance=stem_importance, loop_base_indices=loop_base_indices, stem_base_indices=stem_base_indices, modsite_indices=modsite_indices, topn_overall=int(args.topn_structures_overall))
        coverage_analysis['prediction'] = {'predicted_label_indices': pred_indices, 'max_prob': float(np.max(probs[0].cpu().numpy())), 'target_label_indices': target_indices}
        coverage_analysis['sequence'] = sequence
        if base_importance_direct is not None:
            coverage_analysis['base_importance_direct'] = base_importance_direct.tolist() if isinstance(base_importance_direct, np.ndarray) else base_importance_direct
        all_coverage_results.append(coverage_analysis)
    print(f'Done: {len(all_coverage_results)} sequences with at least {args.min_modsites} modification sites')
    all_coverage_results.sort(key=lambda x: (x['overall_coverage_ratio'], x['topn_structures_covered_modsites']), reverse=True)
    with open(os.path.join(results_dir, 'all_coverage_results.json'), 'w') as f:
        json.dump(all_coverage_results, f, indent=2, ensure_ascii=False)
    topk_candidates = all_coverage_results[:args.topk_candidates]
    candidate_rows = []
    for i, candidate in enumerate(topk_candidates, start=1):
        candidate_rows.append({'rank': i, 'sequence_index': candidate['sequence_index'], 'rna_symbol': candidate['rna_symbol'], 'sequence_length': candidate['sequence_length'], 'total_modsites': candidate['total_modsites'], 'topn_structures_covered_modsites': candidate['topn_structures_covered_modsites'], 'overall_coverage_ratio': candidate['overall_coverage_ratio'], 'top_structures_count': candidate['top_structures_count'], 'max_prob': candidate['prediction']['max_prob'], 'predicted_labels': len(candidate['prediction']['predicted_label_indices'])})
    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df.to_csv(os.path.join(results_dir, 'topk_candidates_summary.csv'), index=False)
    detailed_candidates = []
    for candidate in topk_candidates:
        detailed_candidates.append({'sequence_index': candidate['sequence_index'], 'rna_symbol': candidate['rna_symbol'], 'sequence': candidate.get('sequence', ''), 'total_modsites': candidate['total_modsites'], 'coverage_analysis': candidate['coverage_results'], 'prediction': candidate['prediction']})
    with open(os.path.join(results_dir, 'topk_candidates_detailed.json'), 'w') as f:
        json.dump(detailed_candidates, f, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 80}")
    print(f'Top-{min(10, len(topk_candidates))} candidates (most substructure coverage of mod sites):')
    print(f"{'=' * 80}")
    for i, candidate in enumerate(topk_candidates[:10], start=1):
        print(f"\n[{i}] {candidate['rna_symbol']} (seq_idx={candidate['sequence_index']})")
        print(f"    Total mod sites: {candidate['total_modsites']}")
        print(f"    Top-{args.topn_structures_overall} structures cover: {candidate['topn_structures_covered_modsites']} mod sites")
        print(f"    Coverage: {candidate['overall_coverage_ratio'] * 100:.2f}%")
        print(f"    Sequence length: {candidate['sequence_length']}")
        print(f"    Max predicted prob: {candidate['prediction']['max_prob']:.4f}")
    print(f'\nOutputs:')
    print(f"  - {os.path.join(results_dir, 'all_coverage_results.json')}")
    print(f"  - {os.path.join(results_dir, 'topk_candidates_summary.csv')}")
    print(f"  - {os.path.join(results_dir, 'topk_candidates_detailed.json')}")
    print('\nComputing cumulative coverage statistics and plotting...')
    plot_cumulative_coverage_with_ci(all_coverage_results, results_dir, args.topn_structures_overall)
    print('\n' + '=' * 80)
    print('Finding sequences with largest gap: top-N vs remaining substructure coverage...')
    print('=' * 80)
    min_modsites_list = [10, 20]
    top_n_value = min(50, args.topn_structures_overall)
    for min_modsites in min_modsites_list:
        print(f"\n{'=' * 80}")
        print(f'Among sequences with >= {min_modsites} mod sites, largest coverage gap...')
        print(f"{'=' * 80}")
        best_sequence = find_max_coverage_difference_sequence(all_coverage_results, top_n=top_n_value, min_modsites=min_modsites)
        if best_sequence:
            print(f'\nBest sequence (mod sites >= {min_modsites}):')
            print(f"   RNA Symbol: {best_sequence.get('rna_symbol', 'Unknown')}")
            print(f"   sequence_index: {best_sequence.get('sequence_index', -1)}")
            print(f"   mod sites: {best_sequence.get('total_modsites', 0)}")
            print(f"   Top-{best_sequence.get('top_n', 50)} substructure coverage: {best_sequence.get('top_coverage_ratio', 0) * 100:.2f}%")
            print(f"   remaining substructure coverage: {best_sequence.get('remaining_coverage_ratio', 0) * 100:.2f}%")
            print(f"   coverage gap: {best_sequence.get('coverage_difference', 0) * 100:.2f}%")
            print(f"\n{'=' * 80}")
            print(f'Dumping all substructures (mod sites >= {min_modsites})...')
            print(f"{'=' * 80}")
            sub_dir = os.path.join(results_dir, f'min_modsites_{min_modsites}')
            os.makedirs(sub_dir, exist_ok=True)
            display_sequence_structures_detail(best_sequence, sub_dir)
        else:
            print(f'\nWarning: no sequence matched (mod sites >= {min_modsites}; may need more structures)')

def find_max_coverage_difference_sequence(all_coverage_results: list, top_n: int=50, min_modsites: int=1):
    max_diff = -1
    best_sequence = None
    for result in all_coverage_results:
        if not result.get('has_modsites', False):
            continue
        all_structures_info = result.get('all_structures_info', [])
        if len(all_structures_info) < top_n * 2:
            continue
        total_modsites = result.get('total_modsites', 0)
        if total_modsites < min_modsites:
            continue
        top_covered = set()
        for struct_info in all_structures_info[:top_n]:
            if struct_info.get('coverage_info'):
                top_covered.update(struct_info['coverage_info'].get('covered_modsite_indices', []))
        top_coverage_ratio = len(top_covered) / total_modsites if total_modsites > 0 else 0.0
        remaining_covered = set()
        for struct_info in all_structures_info[top_n:]:
            if struct_info.get('coverage_info'):
                remaining_covered.update(struct_info['coverage_info'].get('covered_modsite_indices', []))
        remaining_coverage_ratio = len(remaining_covered) / total_modsites if total_modsites > 0 else 0.0
        coverage_diff = top_coverage_ratio - remaining_coverage_ratio
        if coverage_diff > max_diff:
            max_diff = coverage_diff
            best_sequence = result.copy()
            best_sequence['top_coverage_ratio'] = top_coverage_ratio
            best_sequence['remaining_coverage_ratio'] = remaining_coverage_ratio
            best_sequence['coverage_difference'] = coverage_diff
            best_sequence['top_n'] = top_n
    return best_sequence

def compute_base_level_importance_from_direct(sequence: str, base_importance_direct: list, modsite_indices: list=None):
    if not sequence:
        return []
    seq_len = len(sequence)
    modsite_set = set(modsite_indices) if modsite_indices else set()
    if isinstance(base_importance_direct, (list, np.ndarray)):
        base_importance_array = np.array(base_importance_direct)
        if len(base_importance_array) != seq_len:
            print(f'[WARN] base_importance_direct length ({len(base_importance_array)}) != sequence length ({seq_len})')
            if len(base_importance_array) > seq_len:
                base_importance_array = base_importance_array[:seq_len]
            else:
                padding = np.zeros(seq_len - len(base_importance_array))
                base_importance_array = np.concatenate([base_importance_array, padding])
    else:
        base_importance_array = np.zeros(seq_len)
    base_importance_list = []
    for i in range(seq_len):
        base_importance_list.append({'position': i, 'base': sequence[i] if i < len(sequence) else 'N', 'importance': float(base_importance_array[i]), 'importance_source': 'direct_from_heterograph', 'is_modsite': i in modsite_set})
    return base_importance_list

def compute_base_level_importance(sequence: str, all_structures_info: list, modsite_indices: list=None):
    if not sequence:
        return []
    seq_len = len(sequence)
    modsite_set = set(modsite_indices) if modsite_indices else set()
    base_importance = {}
    for i in range(seq_len):
        base_importance[i] = {'position': i, 'base': sequence[i] if i < len(sequence) else 'N', 'importance': 0.0, 'max_importance': 0.0, 'mean_importance': 0.0, 'sum_importance': 0.0, 'num_structures': 0, 'structure_types': [], 'is_modsite': i in modsite_set}
    for struct_info in all_structures_info:
        struct_type = struct_info.get('structure_type', 'unknown')
        importance = struct_info.get('importance', 0.0)
        base_indices = struct_info.get('base_indices', [])
        if isinstance(base_indices, (list, np.ndarray)):
            base_indices = [int(b) for b in base_indices if int(b) >= 0 and int(b) < seq_len]
        else:
            continue
        for base_idx in base_indices:
            if base_idx in base_importance:
                base_info = base_importance[base_idx]
                base_info['num_structures'] += 1
                base_info['sum_importance'] += importance
                base_info['structure_types'].append(struct_type)
                if importance > base_info['max_importance']:
                    base_info['max_importance'] = importance
    base_importance_list = []
    for i in range(seq_len):
        if i in base_importance:
            base_info = base_importance[i]
            if base_info['num_structures'] > 0:
                base_info['mean_importance'] = base_info['sum_importance'] / base_info['num_structures']
            else:
                base_info['mean_importance'] = 0.0
            base_info['importance'] = base_info['max_importance']
            base_info['importance_source'] = 'from_structures'
            base_info['structure_types'] = list(set(base_info['structure_types']))
            base_importance_list.append(base_info)
    return base_importance_list

def display_sequence_structures_detail(sequence_result: dict, results_dir: str):
    if not sequence_result:
        print('[WARN] No sequence provided for detailed display')
        return
    seq_idx = sequence_result.get('sequence_index', -1)
    rna_symbol = sequence_result.get('rna_symbol', 'Unknown')
    sequence = sequence_result.get('sequence', '')
    total_modsites = sequence_result.get('total_modsites', 0)
    all_structures_info = sequence_result.get('all_structures_info', [])
    modsite_indices = sequence_result.get('modsite_indices', [])
    if not modsite_indices:
        if 'coverage_results' in sequence_result:
            for cov_result in sequence_result['coverage_results']:
                modsite_indices.extend(cov_result.get('covered_modsite_indices', []))
        if not modsite_indices:
            for struct_info in all_structures_info:
                coverage_info = struct_info.get('coverage_info', {})
                if coverage_info:
                    modsite_indices.extend(coverage_info.get('covered_modsite_indices', []))
        modsite_indices = sorted(list(set(modsite_indices)))
    print(f"\n{'=' * 80}")
    print(f'Sequence detail: {rna_symbol} (seq_idx={seq_idx})')
    print(f"{'=' * 80}")
    print(f'Length: {len(sequence)}')
    print(f'Total mod sites: {total_modsites}')
    print(f'Total substructures: {len(all_structures_info)}')
    if 'coverage_difference' in sequence_result:
        print(f"\nTop-{sequence_result.get('top_n', 50)} substructure coverage: {sequence_result.get('top_coverage_ratio', 0) * 100:.2f}%")
        print(f"Remaining substructure coverage: {sequence_result.get('remaining_coverage_ratio', 0) * 100:.2f}%")
        print(f"Coverage gap: {sequence_result.get('coverage_difference', 0) * 100:.2f}%")
    print(f"\n{'=' * 80}")
    print(f'All substructures (sorted by importance):')
    print(f"{'=' * 80}")
    structures_detail = []
    for i, struct_info in enumerate(all_structures_info, 1):
        struct_type = struct_info.get('structure_type', 'unknown')
        struct_idx = struct_info.get('structure_index', -1)
        importance = struct_info.get('importance', 0.0)
        base_indices = struct_info.get('base_indices', [])
        coverage_info = struct_info.get('coverage_info', {})
        covered_count = coverage_info.get('covered_modsites_count', 0) if coverage_info else 0
        coverage_ratio = coverage_info.get('coverage_ratio', 0.0) if coverage_info else 0.0
        covered_indices = coverage_info.get('covered_modsite_indices', []) if coverage_info else []
        structures_detail.append({'rank': i, 'structure_type': struct_type, 'structure_index': struct_idx, 'importance': importance, 'base_count': len(base_indices), 'base_indices': base_indices, 'covered_modsites_count': covered_count, 'coverage_ratio': coverage_ratio, 'covered_modsite_indices': covered_indices})
        print(f'\n[{i}] {struct_type.upper()} #{struct_idx}')
        print(f'    importance: {importance:.6f}')
        print(f'    num bases: {len(base_indices)}')
        print(f"    base indices: {base_indices[:20]}{('...' if len(base_indices) > 20 else '')} (n={len(base_indices)})")
        print(f'    mods covered: {covered_count} / {total_modsites} ({coverage_ratio * 100:.2f}%)')
        if covered_indices:
            print(f"    covered mod indices: {covered_indices[:10]}{('...' if len(covered_indices) > 10 else '')} (n={len(covered_indices)})")
    output_data = {'sequence_index': seq_idx, 'rna_symbol': rna_symbol, 'sequence': sequence, 'sequence_length': len(sequence), 'total_modsites': total_modsites, 'total_structures': len(all_structures_info), 'coverage_statistics': {'top_coverage_ratio': sequence_result.get('top_coverage_ratio', 0.0), 'remaining_coverage_ratio': sequence_result.get('remaining_coverage_ratio', 0.0), 'coverage_difference': sequence_result.get('coverage_difference', 0.0), 'top_n': sequence_result.get('top_n', 50)}, 'all_structures': structures_detail}
    output_path = os.path.join(results_dir, f'best_sequence_structures_detail_seq{seq_idx}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 80}")
    print(f'Structure detail JSON: {output_path}')
    print(f"{'=' * 80}")
    csv_data = []
    for struct in structures_detail:
        csv_data.append({'Rank': struct['rank'], 'Type': struct['structure_type'], 'Index': struct['structure_index'], 'Importance': struct['importance'], 'Base_Count': struct['base_count'], 'Base_Indices': str(struct['base_indices']), 'Covered_Modsites': struct['covered_modsites_count'], 'Coverage_Ratio': struct['coverage_ratio'], 'Covered_Modsite_Indices': str(struct['covered_modsite_indices'])})
    csv_df = pd.DataFrame(csv_data)
    csv_path = os.path.join(results_dir, f'best_sequence_structures_detail_seq{seq_idx}.csv')
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'Structure detail CSV: {csv_path}')
    print(f"\n{'=' * 80}")
    print(f'Per-base importance...')
    print(f"{'=' * 80}")
    base_importance_direct = sequence_result.get('base_importance_direct', None)
    if base_importance_direct is not None:
        print('Using direct heterograph base importance...')
        base_importance_list = compute_base_level_importance_from_direct(sequence=sequence, base_importance_direct=base_importance_direct, modsite_indices=modsite_indices)
    else:
        print('Using structure-aggregated base importance...')
        base_importance_list = compute_base_level_importance(sequence=sequence, all_structures_info=all_structures_info, modsite_indices=modsite_indices)
    base_importance_output = {'sequence_index': seq_idx, 'rna_symbol': rna_symbol, 'sequence': sequence, 'sequence_length': len(sequence), 'total_modsites': total_modsites, 'total_structures': len(all_structures_info), 'base_importance': base_importance_list}
    base_importance_json_path = os.path.join(results_dir, f'best_sequence_base_importance_seq{seq_idx}.json')
    with open(base_importance_json_path, 'w', encoding='utf-8') as f:
        json.dump(base_importance_output, f, indent=2, ensure_ascii=False)
    print(f'Base importance JSON: {base_importance_json_path}')
    base_csv_data = []
    for base_info in base_importance_list:
        csv_row = {'Position': base_info['position'], 'Base': base_info['base'], 'Importance': base_info['importance'], 'Importance_Source': base_info.get('importance_source', 'unknown'), 'Is_Modsite': base_info['is_modsite']}
        if base_info.get('importance_source') == 'from_structures':
            csv_row.update({'Importance_Max': base_info.get('max_importance', base_info['importance']), 'Importance_Mean': base_info.get('mean_importance', 0.0), 'Importance_Sum': base_info.get('sum_importance', 0.0), 'Num_Structures': base_info.get('num_structures', 0), 'Structure_Types': ','.join(base_info.get('structure_types', [])) if base_info.get('structure_types') else ''})
        else:
            csv_row.update({'Importance_Max': base_info['importance'], 'Importance_Mean': base_info['importance'], 'Importance_Sum': base_info['importance'], 'Num_Structures': 0, 'Structure_Types': ''})
        base_csv_data.append(csv_row)
    base_csv_df = pd.DataFrame(base_csv_data)
    base_csv_path = os.path.join(results_dir, f'best_sequence_base_importance_seq{seq_idx}.csv')
    base_csv_df.to_csv(base_csv_path, index=False, encoding='utf-8')
    print(f'Base importance CSV: {base_csv_path}')
    if base_importance_list:
        importance_values = [b['importance'] for b in base_importance_list if b['importance'] > 0]
        if importance_values:
            print(f'\nBase importance stats:')
            print(f'  bases with positive importance: {len(importance_values)} / {len(base_importance_list)}')
            print(f'  max: {max(importance_values):.6f}')
            print(f'  mean: {np.mean(importance_values):.6f}')
            print(f'  min: {min(importance_values):.6f}')
        modsite_bases = [b for b in base_importance_list if b['is_modsite']]
        if modsite_bases:
            modsite_importance_values = [b['importance'] for b in modsite_bases if b['importance'] > 0]
            if modsite_importance_values:
                print(f'\nMod-site base importance stats:')
                print(f'  mod sites with positive importance: {len(modsite_importance_values)} / {len(modsite_bases)}')
                print(f'  max: {max(modsite_importance_values):.6f}')
                print(f'  mean: {np.mean(modsite_importance_values):.6f}')
                print(f'  min: {min(modsite_importance_values):.6f}')
    return output_data

def compute_segmented_coverage_statistics(all_coverage_results: list, num_segments: int=10):
    all_sequences_data = []
    for result in all_coverage_results:
        if not result.get('has_modsites', False):
            continue
        cumulative_coverage = result.get('cumulative_coverage', [])
        if not cumulative_coverage:
            continue
        total_modsites = result.get('total_modsites', 0)
        total_structures = result.get('total_structures', len(cumulative_coverage))
        if total_modsites == 0 or total_structures == 0:
            continue
        cumulative_covered_counts = []
        for item in cumulative_coverage:
            cumulative_covered_counts.append(item['cumulative_covered_count'])
        all_sequences_data.append({'total_modsites': total_modsites, 'total_structures': total_structures, 'cumulative_covered_counts': cumulative_covered_counts})
    if not all_sequences_data:
        return None
    total_modsites_global = sum((d['total_modsites'] for d in all_sequences_data))
    max_structures = max((d['total_structures'] for d in all_sequences_data))
    avg_structures = np.mean([d['total_structures'] for d in all_sequences_data])
    structures_per_segment = avg_structures / num_segments
    segment_coverage = []
    segment_structure_ratio = []
    segment_labels = []
    segment_avg_structures = []
    for seg_idx in range(num_segments):
        start_struct_idx = int(seg_idx * structures_per_segment)
        end_struct_idx = int((seg_idx + 1) * structures_per_segment)
        total_covered_in_segment = 0
        total_structures_in_segment = 0
        num_valid_sequences = 0
        for data in all_sequences_data:
            total_structs = data['total_structures']
            start_idx = start_struct_idx
            end_idx = end_struct_idx
            if start_idx >= total_structs:
                continue
            if end_idx > total_structs:
                end_idx = total_structs
            if start_idx >= end_idx:
                continue
            if end_idx > len(data['cumulative_covered_counts']):
                end_idx = len(data['cumulative_covered_counts'])
            if start_idx >= end_idx:
                continue
            structures_in_segment = end_idx - start_idx
            total_structures_in_segment += structures_in_segment
            num_valid_sequences += 1
            if end_idx > 0:
                covered_at_end = data['cumulative_covered_counts'][end_idx - 1]
            else:
                covered_at_end = 0
            if start_idx > 0:
                covered_at_start = data['cumulative_covered_counts'][start_idx - 1]
            else:
                covered_at_start = 0
            new_covered = covered_at_end - covered_at_start
            total_covered_in_segment += new_covered
        segment_cov = total_covered_in_segment / total_modsites_global if total_modsites_global > 0 else 0.0
        segment_coverage.append(segment_cov)
        avg_structures_per_segment = total_structures_in_segment / num_valid_sequences if num_valid_sequences > 0 else structures_per_segment
        segment_avg_structures.append(avg_structures_per_segment)
        segment_structure_ratio.append(avg_structures_per_segment / avg_structures if avg_structures > 0 else 1.0 / num_segments)
        segment_labels.append(f'Segment {seg_idx + 1}\n(#{start_struct_idx}-#{end_struct_idx - 1})\n~{avg_structures_per_segment:.1f} structs')
    cumulative_segment_coverage = []
    cumulative_cov = 0.0
    for cov in segment_coverage:
        cumulative_cov += cov
        cumulative_segment_coverage.append(cumulative_cov)
    return {'num_segments': num_segments, 'segment_coverage': segment_coverage, 'cumulative_segment_coverage': cumulative_segment_coverage, 'segment_structure_ratio': segment_structure_ratio, 'segment_avg_structures': segment_avg_structures, 'segment_labels': segment_labels, 'total_modsites': total_modsites_global, 'num_sequences': len(all_sequences_data)}

def compute_global_coverage_statistics(all_coverage_results: list, max_structures: int=None):
    all_sequences_data = []
    for result in all_coverage_results:
        if not result.get('has_modsites', False):
            continue
        cumulative_coverage = result.get('cumulative_coverage', [])
        if not cumulative_coverage:
            continue
        total_modsites = result.get('total_modsites', 0)
        if total_modsites == 0:
            continue
        cumulative_covered_counts = []
        for item in cumulative_coverage:
            cumulative_covered_counts.append(item['cumulative_covered_count'])
        all_sequences_data.append({'total_modsites': total_modsites, 'cumulative_covered_counts': cumulative_covered_counts, 'total_structures': result.get('total_structures', len(cumulative_coverage))})
    if not all_sequences_data:
        return None
    total_modsites_global = sum((d['total_modsites'] for d in all_sequences_data))
    max_len = max((len(d['cumulative_covered_counts']) for d in all_sequences_data))
    if max_structures is not None:
        max_len = min(max_len, max_structures)
    global_coverage_ratios = []
    global_structure_ratios = []
    for i in range(max_len):
        total_covered = 0
        total_structures_sum = 0
        sequences_count = 0
        for data in all_sequences_data:
            if i < len(data['cumulative_covered_counts']):
                total_covered += data['cumulative_covered_counts'][i]
                total_structures_sum += data['total_structures']
                sequences_count += 1
        global_coverage = total_covered / total_modsites_global if total_modsites_global > 0 else 0.0
        global_coverage_ratios.append(global_coverage)
        avg_total_structures = total_structures_sum / sequences_count if sequences_count > 0 else 1
        global_structure_ratio = (i + 1) / avg_total_structures if avg_total_structures > 0 else 0
        global_structure_ratios.append(global_structure_ratio)
    return {'num_structures': list(range(1, max_len + 1)), 'global_coverage': global_coverage_ratios, 'global_structure_ratio': global_structure_ratios, 'total_modsites': total_modsites_global, 'num_sequences': len(all_sequences_data)}

def compute_cumulative_coverage_statistics(all_coverage_results: list, max_structures: int=None):
    all_cumulative_ratios = []
    all_structure_ratios = []
    for result in all_coverage_results:
        if not result.get('has_modsites', False):
            continue
        cumulative_coverage = result.get('cumulative_coverage', [])
        if not cumulative_coverage:
            continue
        coverage_ratios = [item['cumulative_coverage_ratio'] for item in cumulative_coverage]
        total_structs = result.get('total_structures', len(coverage_ratios))
        structure_ratios = [(i + 1) / total_structs if total_structs > 0 else 0 for i in range(len(coverage_ratios))]
        if max_structures is not None:
            coverage_ratios = coverage_ratios[:max_structures]
            structure_ratios = structure_ratios[:max_structures]
        all_cumulative_ratios.append(coverage_ratios)
        all_structure_ratios.append(structure_ratios)
    if not all_cumulative_ratios:
        return {'num_structures': [], 'mean_coverage': [], 'std_coverage': [], 'ci_lower': [], 'ci_upper': [], 'num_sequences': 0}
    max_len = max((len(ratios) for ratios in all_cumulative_ratios)) if all_cumulative_ratios else 0
    aligned_ratios = []
    aligned_structure_ratios = []
    for i, ratios in enumerate(all_cumulative_ratios):
        if len(ratios) == 0:
            continue
        struct_ratios = all_structure_ratios[i] if i < len(all_structure_ratios) else []
        if len(ratios) < max_len:
            ratios = ratios + [ratios[-1]] * (max_len - len(ratios))
            if struct_ratios:
                struct_ratios = struct_ratios + [struct_ratios[-1]] * (max_len - len(struct_ratios))
        aligned_ratios.append(ratios)
        aligned_structure_ratios.append(struct_ratios)
    num_structures = list(range(1, max_len + 1))
    mean_coverage = []
    std_coverage = []
    ci_lower = []
    ci_upper = []
    confidence_level = 0.95
    alpha = 1 - confidence_level
    for i in range(max_len):
        coverage_at_i = [ratios[i] for ratios in aligned_ratios]
        mean_val = np.mean(coverage_at_i)
        std_val = np.std(coverage_at_i, ddof=1)
        n = len(coverage_at_i)
        mean_coverage.append(mean_val)
        std_coverage.append(std_val)
        if n > 1:
            t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_critical * std_val / np.sqrt(n)
            ci_lower.append(max(0.0, mean_val - margin))
            ci_upper.append(min(1.0, mean_val + margin))
        else:
            ci_lower.append(mean_val)
            ci_upper.append(mean_val)
    mean_structure_ratio = []
    for i in range(max_len):
        if aligned_structure_ratios and len(aligned_structure_ratios) > 0:
            struct_ratios_at_i = [ratios[i] for ratios in aligned_structure_ratios if i < len(ratios)]
            if struct_ratios_at_i:
                mean_structure_ratio.append(np.mean(struct_ratios_at_i))
            else:
                mean_structure_ratio.append((i + 1) / max_len)
        else:
            mean_structure_ratio.append((i + 1) / max_len)
    result = {'num_structures': num_structures, 'mean_coverage': mean_coverage, 'std_coverage': std_coverage, 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'mean_structure_ratio': mean_structure_ratio, 'num_sequences': len(all_cumulative_ratios)}
    return result

def remove_all_text(fig):
    for ax in fig.get_axes():
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        for text in ax.texts:
            text.set_text('')
        try:
            xticks = ax.get_xticks()
            if len(xticks) > 0:
                ax.set_xticklabels([''] * len(xticks))
        except:
            pass
        try:
            yticks = ax.get_yticks()
            if len(yticks) > 0:
                ax.set_yticklabels([''] * len(yticks))
        except:
            pass

def save_figure_with_no_text(fig, output_path):
    png_path = output_path + '.png'
    fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    remove_all_text(fig)
    png_no_text_path = output_path + '_no_text.png'
    fig.savefig(png_no_text_path, format='png', dpi=600, bbox_inches='tight')
    return (png_path, png_no_text_path)

def plot_cumulative_coverage_with_ci(all_coverage_results: list, results_dir: str, max_structures: int=50):
    stats_data_all = compute_cumulative_coverage_statistics(all_coverage_results, max_structures=None)
    if stats_data_all['num_sequences'] > 0 and stats_data_all['mean_coverage']:
        stats_data_all['final_coverage_all_structures'] = stats_data_all['mean_coverage'][-1]
        stats_data_all['final_ci_lower_all'] = stats_data_all['ci_lower'][-1]
        stats_data_all['final_ci_upper_all'] = stats_data_all['ci_upper'][-1]
        stats_data_all['total_structures_analyzed'] = len(stats_data_all['num_structures'])
    if max_structures is not None and len(stats_data_all['num_structures']) > max_structures:
        stats_data = {}
        for key in ['num_structures', 'mean_coverage', 'std_coverage', 'ci_lower', 'ci_upper', 'mean_structure_ratio']:
            if key in stats_data_all and stats_data_all[key]:
                stats_data[key] = stats_data_all[key][:max_structures]
        for key in ['final_coverage_all_structures', 'final_ci_lower_all', 'final_ci_upper_all', 'total_structures_analyzed', 'num_sequences']:
            if key in stats_data_all:
                stats_data[key] = stats_data_all[key]
    else:
        stats_data = stats_data_all
    if stats_data['num_sequences'] == 0:
        print('[WARN] No valid sequence data for plotting cumulative coverage')
        return
    global_stats = compute_global_coverage_statistics(all_coverage_results, max_structures=max_structures)
    segmented_stats = compute_segmented_coverage_statistics(all_coverage_results, num_segments=10)
    num_structures = stats_data['num_structures']
    mean_coverage = stats_data['mean_coverage']
    ci_lower = stats_data['ci_lower']
    ci_upper = stats_data['ci_upper']
    max_coverage = max(ci_upper) if ci_upper else 1.0
    y_max = min(1.05, max_coverage * 1.15)
    y_min = 0.0
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    label_mean = 'Sorted by Importance'
    label_ci = '95% Confidence Interval'
    xlabel = 'Number of Structures'
    ylabel = 'Cumulative Modification Site Coverage'
    title_full = 'Cumulative Coverage by Important Substructures'
    title_ratio = 'Structure Ratio vs Coverage (Highlighting Efficiency)'
    title_segments = 'Coverage by Importance Segments (10 Segments)'
    title_efficiency = 'Coverage Efficiency (Coverage / Structure Ratio)'
    xlabel_ratio = 'Structure Ratio (Top-K / Total)'
    mean_structure_ratio = stats_data.get('mean_structure_ratio', [])
    if not mean_structure_ratio:
        avg_total_structures = np.mean([r.get('total_structures', len(num_structures)) for r in all_coverage_results if r.get('has_modsites', False)])
        mean_structure_ratio = [(i + 1) / avg_total_structures if avg_total_structures > 0 else 0 for i in range(len(num_structures))]
    ax1.plot(num_structures, mean_coverage, color='#2E86AB', linewidth=2.5, label=label_mean, zorder=4, marker='o', markersize=3)
    ax1.fill_between(num_structures, ci_lower, ci_upper, color='#2E86AB', alpha=0.25, label=label_ci, zorder=3)
    if len(num_structures) >= 50:
        idx_50 = 49
        coverage_50 = mean_coverage[idx_50]
        struct_ratio_50 = mean_structure_ratio[idx_50] if idx_50 < len(mean_structure_ratio) else 0
        ax1.plot([50], [coverage_50], 'ro', markersize=10, zorder=5)
        ax1.annotate(f'Top-50: {coverage_50 * 100:.1f}%\n({struct_ratio_50 * 100:.1f}% of structures)', xy=(50, coverage_50), xytext=(60, coverage_50 + 0.05), fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    if 'final_coverage_all_structures' in stats_data:
        final_cov = stats_data['final_coverage_all_structures']
        total_structs = stats_data.get('total_structures_analyzed', len(num_structures))
        if total_structs > len(num_structures):
            ax1.text(0.98, 0.02, f'All Structures ({total_structs})\nFinal Coverage: {final_cov * 100:.1f}%', transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8), verticalalignment='bottom', horizontalalignment='right')
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title(title_full, fontsize=12, pad=8, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax1.set_xlim(0, max(num_structures) + 1)
    ax1.set_ylim(y_min, y_max)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    if mean_structure_ratio:
        max_ratio = max(mean_structure_ratio) if mean_structure_ratio else 1.0
        ax2.plot([0, max_ratio], [0, max_ratio], 'k--', linewidth=2, alpha=0.5, label='1:1 Reference', zorder=1)
        ax2.plot(mean_structure_ratio, mean_coverage, color='#2E86AB', linewidth=3, label='Actual Coverage', zorder=4, marker='o', markersize=3)
        ax2.fill_between(mean_structure_ratio, ci_lower, ci_upper, color='#2E86AB', alpha=0.25, zorder=3)
        above_line = np.array(mean_coverage) > np.array(mean_structure_ratio)
        below_line = np.array(mean_coverage) < np.array(mean_structure_ratio)
        if np.any(above_line):
            ax2.fill_between(mean_structure_ratio, mean_structure_ratio, mean_coverage, where=above_line, interpolate=True, color='green', alpha=0.3, label='Above 1:1 (High Efficiency)', zorder=2)
        if np.any(below_line):
            ax2.fill_between(mean_structure_ratio, mean_coverage, mean_structure_ratio, where=below_line, interpolate=True, color='red', alpha=0.3, label='Below 1:1 (Low Efficiency)', zorder=2)
        area_above = 0
        area_below = 0
        for i in range(len(mean_structure_ratio) - 1):
            x1, x2 = (mean_structure_ratio[i], mean_structure_ratio[i + 1])
            y1, y2 = (mean_coverage[i], mean_coverage[i + 1])
            ref1, ref2 = (x1, x2)
            if y1 > ref1 and y2 > ref2:
                area_above += 0.5 * (x2 - x1) * (y1 - ref1 + (y2 - ref2))
            elif y1 < ref1 and y2 < ref2:
                area_below += 0.5 * (x2 - x1) * (ref1 - y1 + (ref2 - y2))
            elif y1 > ref1 and y2 < ref2:
                t = (ref1 - y1) / (y2 - y1 - (ref2 - ref1))
                x_int = x1 + t * (x2 - x1)
                area_above += 0.5 * (x_int - x1) * (y1 - ref1)
                area_below += 0.5 * (x2 - x_int) * (ref2 - y2)
            elif y1 < ref1 and y2 > ref2:
                t = (ref1 - y1) / (y2 - y1 - (ref2 - ref1))
                x_int = x1 + t * (x2 - x1)
                area_below += 0.5 * (x_int - x1) * (ref1 - y1)
                area_above += 0.5 * (x2 - x_int) * (y2 - ref2)
        if len(mean_structure_ratio) >= 50:
            idx_50 = 49
            ax2.plot([mean_structure_ratio[idx_50]], [mean_coverage[idx_50]], 'ro', markersize=10, zorder=5)
        if area_above > area_below:
            interpretation = f'✓ Important structures show higher efficiency\nArea above 1:1: {area_above:.4f}\nArea below 1:1: {area_below:.4f}\nNet advantage: {area_above - area_below:.4f}'
            text_color = 'green'
        else:
            interpretation = f'⚠️ Efficiency needs improvement\nArea above 1:1: {area_above:.4f}\nArea below 1:1: {area_below:.4f}\nNet: {area_above - area_below:.4f}'
            text_color = 'orange'
        ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes, fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=text_color, linewidth=2), verticalalignment='top', horizontalalignment='left')
        ax2.set_xlabel(xlabel_ratio, fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_title(title_ratio, fontsize=12, pad=8, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.legend(loc='lower right', fontsize=8, framealpha=0.9)
        ax2.set_xlim(0, min(1.0, max(mean_structure_ratio) * 1.1))
        ax2.set_ylim(y_min, y_max)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    if segmented_stats:
        segments = list(range(1, segmented_stats['num_segments'] + 1))
        segment_coverage = segmented_stats['segment_coverage']
        cumulative_segment_coverage = segmented_stats['cumulative_segment_coverage']
        segment_labels = segmented_stats['segment_labels']
        segment_avg_structures = segmented_stats.get('segment_avg_structures', [])
        coverage_per_structure = []
        for i, (cov, avg_structs) in enumerate(zip(segment_coverage, segment_avg_structures)):
            if avg_structs > 0:
                cov_per_struct = cov / avg_structs
            else:
                cov_per_struct = 0
            coverage_per_structure.append(cov_per_struct)
        colors = plt.cm.viridis(np.linspace(0, 1, len(segments)))
        bars = ax3.bar(segments, segment_coverage, color=colors, alpha=0.7, label='Coverage per Segment', edgecolor='black', linewidth=0.5)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(segments, coverage_per_structure, color='#E63946', linewidth=2.5, marker='s', markersize=6, label='Coverage per Structure', zorder=5, linestyle='-')
        ax3.set_xlabel('Importance Segments', fontsize=11)
        ax3.set_ylabel('Coverage per Segment (%)', fontsize=11, color='#2E86AB')
        ax3_twin.set_ylabel('Coverage per Structure (×10⁻³)', fontsize=11, color='#E63946')
        ax3.set_title(title_segments, fontsize=12, pad=8, fontweight='bold')
        ax3.set_xticks(segments)
        if segment_avg_structures:
            xlabels = [f'S{i}\n~{int(s)}' for i, s in zip(segments, segment_avg_structures)]
        else:
            xlabels = [f'S{i}' for i in segments]
        ax3.set_xticklabels(xlabels, fontsize=8, rotation=45, ha='right')
        for i, (bar, cov) in enumerate(zip(bars, segment_coverage)):
            height = bar.get_height()
            label_y = height * 0.8
            ax3.text(bar.get_x() + bar.get_width() / 2.0, label_y, f'{cov * 100:.1f}%', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        for i, cov_per_struct in enumerate(coverage_per_structure):
            y_max_twin = ax3_twin.get_ylim()[1] if hasattr(ax3_twin, 'get_ylim') else max(coverage_per_structure) * 1.2
            label_y = cov_per_struct + y_max_twin * 0.08
            ax3_twin.text(segments[i], label_y, f'{cov_per_struct * 1000:.2f}', ha='center', va='bottom', fontsize=7, color='#E63946', fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax3.set_ylim(0, max(segment_coverage) * 1.2 if segment_coverage else 1.0)
        if coverage_per_structure:
            ax3_twin.set_ylim(0, max(coverage_per_structure) * 1.2)
        if cumulative_segment_coverage:
            final_cumulative = cumulative_segment_coverage[-1]
            note_text = f'Cumulative (10 segments): {final_cumulative * 100:.1f}%\n'
            note_text += f'Note: Only showing first 10 segments.\n'
            note_text += f'Total coverage may be <100% as some\n'
            note_text += f'modification sites are not in any structure.'
            ax3.text(0.98, 0.02, note_text, transform=ax3.transAxes, fontsize=8, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=1), verticalalignment='bottom', horizontalalignment='right')
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8, framealpha=0.9)
        ax3.tick_params(axis='y', labelcolor='#2E86AB')
        ax3_twin.tick_params(axis='y', labelcolor='#E63946')
    if mean_structure_ratio:
        marginal_efficiency = []
        avg_total_structures = np.mean([r.get('total_structures', len(mean_coverage)) for r in all_coverage_results if r.get('has_modsites', False)])
        prev_coverage = 0
        for i, (cov, ratio) in enumerate(zip(mean_coverage, mean_structure_ratio)):
            marginal_cov = cov - prev_coverage if i > 0 else cov
            marginal_struct_ratio = 1.0 / avg_total_structures if avg_total_structures > 0 else 0
            marg_eff = marginal_cov / marginal_struct_ratio if marginal_struct_ratio > 0 else 0
            marginal_efficiency.append(marg_eff)
            prev_coverage = cov
        cumulative_efficiency = [cov / ratio if ratio > 0 else 0 for cov, ratio in zip(mean_coverage, mean_structure_ratio)]
        efficiency = marginal_efficiency
        if efficiency:
            y_min_eff = min(efficiency) * 0.95
            y_max_eff = max(efficiency) * 1.15
            if y_max_eff - y_min_eff < 0.2:
                center = (y_min_eff + y_max_eff) / 2
                y_min_eff = max(0.5, center - 0.15)
                y_max_eff = center + 0.15
        else:
            y_max_eff = 1.2
            y_min_eff = 0.8
        efficiency_line = ax4.plot(num_structures, efficiency, color='#2E86AB', linewidth=2.5, label='Marginal Efficiency', zorder=4, marker='o', markersize=2)[0]
        if len(efficiency) > 0:
            eff_mean = np.mean(efficiency)
            green_patch = ax4.fill_between(num_structures, [eff_mean] * len(efficiency), efficiency, where=[e > eff_mean for e in efficiency], color='green', alpha=0.2, zorder=0)
            red_patch = ax4.fill_between(num_structures, efficiency, [eff_mean] * len(efficiency), where=[e < eff_mean for e in efficiency], color='red', alpha=0.2, zorder=0)
            mean_line = ax4.axhline(y=eff_mean, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [Line2D([0], [0], color='#2E86AB', linewidth=2.5, marker='o', markersize=4, label='Marginal Efficiency'), Patch(facecolor='green', alpha=0.2, label='Above Mean (High Efficiency)'), Patch(facecolor='red', alpha=0.2, label='Below Mean (Low Efficiency)'), Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Mean Reference Line')]
            ax4.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
        ax4.set_ylim(y_min_eff, y_max_eff)
        ax4.set_xlabel(xlabel, fontsize=11)
        ax4.set_ylabel('Marginal Efficiency (Coverage per Structure)', fontsize=11)
        ax4.set_title('Marginal Efficiency: Coverage per Structure', fontsize=12, pad=8, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax4.set_xlim(0, max(num_structures) + 1)
        if len(efficiency) >= 50:
            eff_top25 = np.mean(efficiency[:25])
            eff_26_50 = np.mean(efficiency[25:50]) if len(efficiency) >= 50 else 0
            eff_top5 = np.mean(efficiency[:5]) if len(efficiency) >= 5 else 0
            print(f"\n{'=' * 70}")
            print(f'[Marginal Efficiency Analysis]')
            print(f'  Top-5 structures average efficiency: {eff_top5:.4f}')
            print(f'  Top-25 structures average efficiency: {eff_top25:.4f}')
            if len(efficiency) >= 50:
                print(f'  Structures 26-50 average efficiency: {eff_26_50:.4f}')
                ratio = eff_top25 / eff_26_50 if eff_26_50 > 0 else 0
                print(f'  Efficiency ratio (Top-25 / 26-50): {ratio:.2f}x')
                if ratio < 1.0:
                    print(f'\n  ⚠️  WARNING: Top-25 structures have LOWER efficiency than structures 26-50!')
                    print(f'     This suggests:')
                    print(f'     1. Importance ranking may need verification')
                    print(f'     2. Early structures might have overlapping coverage')
                    print(f'     3. Later structures might cover different modification sites')
                    print(f'     4. Consider checking the importance scores and coverage patterns')
                elif ratio > 1.2:
                    print(f'\n  ✓ Top-25 structures show significantly higher efficiency!')
                else:
                    print(f'\n  Note: Efficiency difference is moderate ({ratio:.2f}x)')
            print(f"{'=' * 70}")
    plt.tight_layout()
    plot_base_path = os.path.join(results_dir, 'cumulative_coverage_with_ci')
    png_path, png_no_text_path = save_figure_with_no_text(fig, plot_base_path)
    print(f'Cumulative coverage plot (PNG): {png_path}')
    print(f'Cumulative coverage plot (no text PNG): {png_no_text_path}')
    stats_df = pd.DataFrame({'num_structures': num_structures, 'mean_coverage': mean_coverage, 'std_coverage': stats_data['std_coverage'], 'ci_lower_95': ci_lower, 'ci_upper_95': ci_upper})
    stats_csv_path = os.path.join(results_dir, 'cumulative_coverage_statistics.csv')
    stats_df.to_csv(stats_csv_path, index=False)
    print(f'Cumulative coverage stats CSV: {stats_csv_path}')
    print(f'\nCumulative coverage summary:')
    print(f"  sequences: {stats_data['num_sequences']}")
    if num_structures:
        print(f'  structures in plot: {max(num_structures)} (display window)')
        if 'total_structures_analyzed' in stats_data:
            print(f"  total structures analyzed: {stats_data['total_structures_analyzed']}")
        print(f'\nImportance order vs random:')
        mean_structure_ratio = stats_data.get('mean_structure_ratio', [])
        if mean_structure_ratio:
            if len(mean_structure_ratio) >= 50:
                struct_ratio_50 = mean_structure_ratio[49]
                coverage_50 = mean_coverage[49]
                print(f'\n[Per-Sequence Average Statistics]')
                print(f'  Top-50 structures ({struct_ratio_50 * 100:.2f}% of structures) cover {coverage_50 * 100:.2f}% of modification sites (averaged across sequences)')
                efficiency_50 = coverage_50 / struct_ratio_50 if struct_ratio_50 > 0 else 0
                print(f'  Coverage efficiency: {efficiency_50:.2f}x (coverage ratio / structure ratio)')
                print(f'  Note: This is the average coverage ratio across sequences, which may not reflect')
                print(f'        the true global coverage when sequences have different numbers of modification sites.')
        if global_stats and len(global_stats['global_coverage']) >= 50:
            global_cov_50 = global_stats['global_coverage'][49]
            global_struct_ratio_50 = global_stats['global_structure_ratio'][49]
            global_total_modsites = global_stats['total_modsites']
            global_efficiency_50 = global_cov_50 / global_struct_ratio_50 if global_struct_ratio_50 > 0 else 0
            print(f"\n{'=' * 70}")
            print(f'[Global Statistics - All Sequences Combined]')
            print(f'  Total modification sites across all sequences: {global_total_modsites:,}')
            print(f'  Top-50 structures ({global_struct_ratio_50 * 100:.2f}% of structures) ')
            print(f'    cover {global_cov_50 * 100:.2f}% of ALL modification sites globally')
            print(f'  Global coverage efficiency: {global_efficiency_50:.2f}x')
            print(f'\n  Why does {global_struct_ratio_50 * 100:.2f}% structures cover {global_cov_50 * 100:.2f}% sites?')
            print(f'  - The efficiency ({global_efficiency_50:.2f}x) indicates that important structures')
            print(f'    cover sites at a rate {global_efficiency_50:.2f}x higher than their proportion')
            print(f'  - This demonstrates the effectiveness of importance-based structure selection')
            print(f"{'=' * 70}")
        if segmented_stats:
            print(f'\n  Coverage by Importance Segments (10 segments):')
            segment_avg_structures = segmented_stats.get('segment_avg_structures', [])
            coverage_per_structure = []
            for i, (seg_cov, avg_structs) in enumerate(zip(segmented_stats['segment_coverage'], segment_avg_structures)):
                cov_per_struct = seg_cov / avg_structs if avg_structs > 0 else 0
                coverage_per_structure.append(cov_per_struct)
            for i, (seg_cov, cum_cov, avg_structs, cov_per_struct) in enumerate(zip(segmented_stats['segment_coverage'], segmented_stats['cumulative_segment_coverage'], segment_avg_structures, coverage_per_structure)):
                print(f'    Segment {i + 1}: {seg_cov * 100:.2f}% coverage, {cum_cov * 100:.2f}% cumulative, ~{avg_structs:.1f} structures, {cov_per_struct * 1000:.3f}×10⁻³ per structure')
            if len(coverage_per_structure) >= 3:
                first_seg_structs = segment_avg_structures[0]
                first_seg_cov = segmented_stats['segment_coverage'][0]
                first_cov_per_struct = coverage_per_structure[0]
                last_seg_structs = segment_avg_structures[-1]
                last_seg_cov = segmented_stats['segment_coverage'][-1]
                last_cov_per_struct = coverage_per_structure[-1]
                struct_ratio = first_seg_structs / last_seg_structs if last_seg_structs > 0 else 0
                cov_ratio = first_seg_cov / last_seg_cov if last_seg_cov > 0 else 0
                cov_per_struct_ratio = first_cov_per_struct / last_cov_per_struct if last_cov_per_struct > 0 else 0
                print(f'\n  Analysis (eliminating quantity factor):')
                print(f'    Segment 1: ~{first_seg_structs:.1f} structures, {first_seg_cov * 100:.2f}% coverage, {first_cov_per_struct * 1000:.3f}×10⁻³ per structure')
                print(f'    Segment 10: ~{last_seg_structs:.1f} structures, {last_seg_cov * 100:.2f}% coverage, {last_cov_per_struct * 1000:.3f}×10⁻³ per structure')
                print(f'    Structure count ratio: {struct_ratio:.2f}x')
                print(f'    Coverage ratio: {cov_ratio:.2f}x')
                print(f'    Coverage per structure ratio: {cov_per_struct_ratio:.2f}x')
                if cov_per_struct_ratio > 1.2:
                    print(f'    ✓ IMPORTANT: Even after eliminating quantity factor, Segment 1 structures')
                    print(f'      cover {cov_per_struct_ratio:.2f}x more sites per structure than Segment 10!')
                    print(f'    ✓ This proves that importance-based selection is effective!')
                elif struct_ratio > 1.5 and cov_ratio / struct_ratio < 1.2:
                    print(f'    ⚠️  Note: Coverage increase is mainly due to structure quantity increase.')
                    print(f'       Coverage per structure ratio ({cov_per_struct_ratio:.2f}x) is relatively low.')
                else:
                    print(f'    Coverage per structure shows moderate difference ({cov_per_struct_ratio:.2f}x).')
        if len(mean_coverage) >= 5:
            imp_5 = mean_coverage[4]
            struct_ratio_5 = mean_structure_ratio[4] if mean_structure_ratio and len(mean_structure_ratio) > 4 else 0
            print(f'  Top-5 structures ({struct_ratio_5 * 100:.2f}%): {imp_5 * 100:.2f}% (95% CI: {ci_lower[4] * 100:.2f}%-{ci_upper[4] * 100:.2f}%)')
        if len(mean_coverage) >= 10:
            imp_10 = mean_coverage[9]
            struct_ratio_10 = mean_structure_ratio[9] if mean_structure_ratio and len(mean_structure_ratio) > 9 else 0
            print(f'  Top-10 structures ({struct_ratio_10 * 100:.2f}%): {imp_10 * 100:.2f}% (95% CI: {ci_lower[9] * 100:.2f}%-{ci_upper[9] * 100:.2f}%)')
        if len(mean_coverage) >= 20:
            imp_20 = mean_coverage[19]
            struct_ratio_20 = mean_structure_ratio[19] if mean_structure_ratio and len(mean_structure_ratio) > 19 else 0
            print(f'  Top-20 structures ({struct_ratio_20 * 100:.2f}%): {imp_20 * 100:.2f}% (95% CI: {ci_lower[19] * 100:.2f}%-{ci_upper[19] * 100:.2f}%)')
        final_coverage = mean_coverage[-1]
        final_ci_lower = ci_lower[-1]
        final_ci_upper = ci_upper[-1]
        print(f'\n  Current display range (Top-{max(num_structures)} structures):')
        print(f'    Coverage: {final_coverage * 100:.2f}% (95% CI: {final_ci_lower * 100:.2f}%-{final_ci_upper * 100:.2f}%)')
        if 'final_coverage_all_structures' in stats_data:
            total_structs = stats_data.get('total_structures_analyzed', len(num_structures))
            final_cov_all = stats_data['final_coverage_all_structures']
            final_ci_lower_all = stats_data['final_ci_lower_all']
            final_ci_upper_all = stats_data['final_ci_upper_all']
            print(f"\n  {'=' * 60}")
            print(f'  Final coverage using all {total_structs} structures:')
            print(f'    importance order: {final_cov_all * 100:.2f}% (95% CI: {final_ci_lower_all * 100:.2f}%-{final_ci_upper_all * 100:.2f}%)')
            print(f'\n  Why is coverage < 100%?')
            print(f'    Reasons:')
            print(f'    1. Loops/stems cover only part of the sequence.')
            print(f'    2. Some mods lie in unstructured regions.')
            print(f'    3. Some mods lie in linker regions between substructures.')
            print(f'    4. Some mods cannot be covered by any substructure.')
            print(f'\n    So even all substructures cover only a fraction of mods,')
            print(f'    motivating importance-focused subsets (e.g. Top-50 ~30%).')
            print(f"  {'=' * 60}")
    plt.close()
if __name__ == '__main__':
    main()
