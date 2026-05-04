#!/usr/bin/env python3

import os
import sys

_ad = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ad not in sys.path:
    sys.path.insert(0, _ad)
from _paths import general_tracker_root, go_analyze, tracker_features, graph_processed_root

sys.path.insert(0, general_tracker_root())
import data_hetero
import model_hetero
from torch.utils.data import DataLoader
import pickle
import argparse
import pandas as pd
import numpy as np
from Bio import SeqIO
import torch


def read_fasta_to_df_no_label(file_path):
    records = list(SeqIO.parse(file_path, 'fasta'))
    descriptions = []
    sequences = []
    labels = []
    for record in records:
        header_parts = str(record.description).split('|')
        if len(header_parts) >= 2:
            desc = '|'.join(header_parts[:2])
        else:
            desc = header_parts[0] if len(header_parts) > 0 else ''
        descriptions.append(desc)
        sequences.append(str(record.seq))
        labels.append('')
    df = pd.DataFrame({'Description': descriptions, 'Sequence': sequences, 'Label': labels})
    return df

def predict_with_model(model, data_loader, device, threshold=0.5, all_locations=None, output_csv=None):
    model.eval()
    all_predictions = []
    all_probs = []
    all_descriptions = []
    total_batches = len(data_loader)
    processed_samples = 0
    if output_csv is not None:
        columns = ['Description']
        if all_locations is not None:
            columns.extend(all_locations)
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False, mode='w')
    batch_results_buffer = []
    skipped_batches = 0
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            try:
                batch_data = batch_data.to(device)
                outputs = model(batch_data)
                probs = torch.sigmoid(outputs)
                probs_np = probs.cpu().numpy()
                batch_size = probs_np.shape[0]
                preds_binary = (probs_np >= threshold).astype(int)
                all_predictions.append(preds_binary)
                all_probs.append(probs_np)
                batch_descriptions = []
                if hasattr(batch_data, 'des'):
                    batch_descriptions = batch_data.des
                    all_descriptions.extend(batch_descriptions)
                for j in range(batch_size):
                    prob_values = probs_np[j]
                    result = {'Description': batch_descriptions[j] if j < len(batch_descriptions) else f'sample_{processed_samples + j}'}
                    if all_locations is not None:
                        for label_idx, label in enumerate(all_locations):
                            result[label] = float(prob_values[label_idx])
                    batch_results_buffer.append(result)
                processed_samples += batch_size
                if (i + 1) % 10 == 0 or i + 1 == total_batches:
                    if output_csv is not None and len(batch_results_buffer) > 0:
                        batch_df = pd.DataFrame(batch_results_buffer)
                        batch_df.to_csv(output_csv, index=False, mode='a', header=False)
                        batch_results_buffer = []
                    print(f'Batch {i + 1}/{total_batches}: processed {processed_samples} samples')
            except Exception as e:
                skipped_batches += 1
                batch_descriptions = batch_data.des if hasattr(batch_data, 'des') else [f'batch_{i}']
                print(f'Skip batch {i + 1}: error - {type(e).__name__}: {str(e)}')
                if len(batch_descriptions) > 0:
                    print(f"  Samples: {(batch_descriptions[0] if len(batch_descriptions) == 1 else f'{len(batch_descriptions)} samples')}")
                continue
    if len(all_predictions) > 0:
        all_predictions = np.vstack(all_predictions)
        all_probs = np.vstack(all_probs)
    else:
        all_predictions = np.array([])
        all_probs = np.array([])
    if skipped_batches > 0:
        print(f'\nWarning: skipped {skipped_batches} batches')
    return (all_predictions, all_probs, all_descriptions)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fasta', type=str, default=os.path.join(go_analyze('lncRNA'), 'sequences.fasta'))
    parser.add_argument('--model_path', type=str, default=os.path.join(go_analyze('lncRNA'), 'model.pth'))
    parser.add_argument('--folding_path', type=str, default=tracker_features('sequences_linearfold.pkl'))
    parser.add_argument('--kmer_path', type=str, default=tracker_features('sequences_5mer.pkl'))
    parser.add_argument('--dacc_path', type=str, default=tracker_features('sequences_dacc.pkl'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(go_analyze('lncRNA'), 'predict_outputs'))
    parser.add_argument('--graph_cache_root', type=str, default=graph_processed_root())
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--all_locations', type=str, nargs='+', default=None)
    args = parser.parse_args()
    print('=' * 80)
    print('Prediction configuration')
    print('=' * 80)
    print(f'Input FASTA: {args.input_fasta}')
    print(f'Model path: {args.model_path}')
    print(f'Output directory: {args.output_dir}')
    print(f'Batch size: {args.batch_size}')
    print(f'Threshold: {args.threshold}')
    print(f'Device: {args.device}')
    print('=' * 80)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'\nLoading input: {args.input_fasta}')
    df = read_fasta_to_df_no_label(args.input_fasta)
    print(f'Rows: {len(df)}')
    unique_descriptions = df['Description'].nunique()
    unique_sequences = df['Sequence'].nunique()
    print(f'Unique descriptions: {unique_descriptions}')
    print(f'Unique sequences: {unique_sequences}')
    if unique_descriptions < len(df):
        print(f'Warning: duplicate descriptions ({len(df) - unique_descriptions})')
        print('   Note: Description may not be unique; prefer Sequence as ID')
    if unique_sequences < len(df):
        print(f'Warning: duplicate sequences ({len(df) - unique_sequences})')
        print('   Note: identical sequences appear more than once')
    if args.all_locations is None:
        all_locations = ['Cytoplasm', 'Cytosol', 'ExtracellularVesicle', 'Membrane', 'Mitochondrion', 'Nucleolus', 'Nucleoplasm', 'Ribosome']
    else:
        all_locations = sorted(args.all_locations)
    print(f'Labels: {all_locations}')
    print(f'Num labels: {len(all_locations)}')
    print('\nBuilding tokenizer...')

    class MinimalTokenizer:

        def __init__(self, label_count):
            self.label_count = label_count
    tokenizer = MinimalTokenizer(len(all_locations))
    print(f'Tokenizer ready, num labels: {tokenizer.label_count}')
    print('\nLoading feature pickles...')
    folding_path = args.folding_path
    kmer_path = args.kmer_path
    dacc_path = args.dacc_path
    if not os.path.exists(folding_path):
        raise FileNotFoundError(f'Folding pickle not found: {folding_path}')
    if not os.path.exists(kmer_path):
        raise FileNotFoundError(f'K-mer pickle not found: {kmer_path}')
    if not os.path.exists(dacc_path):
        raise FileNotFoundError(f'DACC pickle not found: {dacc_path}')
    print(f'  - Folding: {folding_path}')
    print(f'  - K-mer: {kmer_path}')
    print(f'  - DACC: {dacc_path}')
    features_kmer = pickle.load(open(kmer_path, 'rb'))
    print(f'  - K-mer entries: {len(features_kmer)}')
    features_dacc = pickle.load(open(dacc_path, 'rb'))
    print(f'  - DACC entries: {len(features_dacc)}')
    foldings = pickle.load(open(folding_path, 'rb'))
    print(f'  - Folding entries: {len(foldings)}')
    print('\nBuilding dataset...')
    import re
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_undirected

    class PredictRNAHeteroGraphDataset(data_hetero.RNAHeteroGraphDataset):

        def process(self, df):
            data_list = []
            for i, row in enumerate(df.itertuples()):
                try:
                    des = str(row.Description)
                    seq_str = str(row.Sequence)
                    if re.findall('[^AGCT]', seq_str.upper()):
                        print(f'Skip row {i}: non-AGCT bases')
                        continue
                    if seq_str not in self.foldings:
                        print(f'Skip row {i}: sequence missing from folding dict')
                        continue
                    dot_bracket_string = self.foldings[seq_str][0]
                    label_embedded = np.zeros((1, self.tokenizer.label_count))
                    y = torch.Tensor(label_embedded).view(1, self.tokenizer.label_count)
                    graph_dict = data_hetero.build_hetero_rna_graph(seq_str, dot_bracket_string)
                    hetero_data = HeteroData()
                    hetero_data['base'].x = graph_dict['base_features']
                    if graph_dict['num_loops'] > 0:
                        hetero_data['loop'].x = graph_dict.get('loop_features', torch.zeros(graph_dict['num_loops'], 6))
                        hetero_data['loop'].base_indices = graph_dict['loop_base_indices']
                    if graph_dict['num_stems'] > 0:
                        hetero_data['stem'].x = graph_dict.get('stem_features', torch.zeros(graph_dict['num_stems'], 5))
                        hetero_data['stem'].base_indices = graph_dict['stem_base_indices']
                    hetero_data['base', 'adjacent', 'base'].edge_index = graph_dict['base_adjacent_edges']
                    hetero_data['base', 'adjacent', 'base'].edge_attr = graph_dict['base_adjacent_attr']
                    hetero_data['base', 'pair', 'base'].edge_index = graph_dict['base_pair_edges']
                    hetero_data['base', 'pair', 'base'].edge_attr = graph_dict['base_pair_attr']
                    if graph_dict['num_loops'] > 0:
                        base_to_loop = graph_dict['base_to_loop_edges']
                        loop_to_base = base_to_loop.flip(0)
                        hetero_data['base', 'belongs_to', 'loop'].edge_index = base_to_loop
                        hetero_data['loop', 'belongs_to', 'base'].edge_index = loop_to_base
                    if graph_dict['num_stems'] > 0:
                        base_to_stem = graph_dict['base_to_stem_edges']
                        stem_to_base = base_to_stem.flip(0)
                        hetero_data['base', 'belongs_to', 'stem'].edge_index = base_to_stem
                        hetero_data['stem', 'belongs_to', 'base'].edge_index = stem_to_base
                    if graph_dict['num_loops'] > 1:
                        loop_loop = to_undirected(graph_dict['loop_to_loop_edges'])
                        hetero_data['loop', 'stem_connects', 'loop'].edge_index = loop_loop
                    hetero_data.y = y
                    hetero_data.kmer = torch.tensor(self.fea_kmer[seq_str], dtype=torch.float32)
                    hetero_data.dacc = torch.tensor(self.fea_dacc[seq_str], dtype=torch.float32)
                    hetero_data.label = []
                    hetero_data.sLen = len(seq_str)
                    hetero_data.rowseq = seq_str
                    hetero_data.dot_bracket_string = dot_bracket_string
                    hetero_data.des = des
                    print(f'{i} items processed..')
                    data_list.append(hetero_data)
                except Exception as e:
                    des_str = str(row.Description)[:50] if hasattr(row, 'Description') else 'unknown'
                    print(f'Skip row {i} (Description: {des_str}...): error - {type(e).__name__}: {str(e)}')
                    continue
            torch.save(data_list, self.processed_paths[0])
    dataset = PredictRNAHeteroGraphDataset(root=args.graph_cache_root, dataset='lncRNA_GO', view='predict', df_data=df, tokenizer=tokenizer, foldings=foldings, fea_kmer=features_kmer, fea_dacc=features_dacc, isMultiLabel=True, device='cpu')
    print(f'Dataset ready, size: {len(dataset)}')
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=data_hetero.hetero_collate_func, num_workers=0)
    print(f'\nLoading model: {args.model_path}')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    num_labels = tokenizer.label_count
    hidden_dim = 128
    n_conv_layers = 2
    conv_type = 'GAT'
    for key in state_dict.keys():
        if 'hidden' in key or 'base_encoder' in key:
            if 'weight' in key:
                shape = state_dict[key].shape
                if len(shape) >= 2:
                    hidden_dim = shape[0] if shape[0] > hidden_dim else hidden_dim
                    break
    use_stem_nodes = 'stem_feat_encoder.weight' in state_dict or any(('stem' in k for k in state_dict.keys()))
    use_label_graph = 'label_gnn.label_emb' in state_dict or any(('label_gnn' in k for k in state_dict.keys()))
    print(f'Model config:')
    print(f'  - num_labels: {num_labels}')
    print(f'  - hidden_dim: {hidden_dim}')
    print(f'  - n_conv_layers: {n_conv_layers}')
    print(f'  - conv_type: {conv_type}')
    print(f'  - use_stem_nodes: {use_stem_nodes}')
    print(f'  - use_label_graph: {use_label_graph}')
    model = model_hetero.RNAHeteroModel(num_base_features=20, num_loop_features=20, num_stem_features=19, hidden_dim=hidden_dim, num_labels=num_labels, conv_type=conv_type, n_conv_layers=n_conv_layers, dropout=0.5, batch_norm=True, use_stem_nodes=use_stem_nodes, use_label_graph=use_label_graph)
    try:
        model.load_state_dict(state_dict, strict=False)
        print('Model weights loaded')
    except Exception as e:
        print(f'Warning: weight load issue: {e}')
        print('Retrying with strict=False...')
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    output_csv = os.path.join(args.output_dir, 'predictions.csv')
    print('\nRunning prediction...')
    total_batches = len(data_loader)
    print(f'Total batches: {total_batches}')
    print(f'Streaming CSV to: {output_csv}')
    predictions, probs, descriptions = predict_with_model(model, data_loader, device, threshold=args.threshold, all_locations=all_locations, output_csv=output_csv)
    print(f'\nPrediction finished, {len(predictions)} samples')
    print(f'Predictions CSV: {output_csv}')
    output_probs = os.path.join(args.output_dir, 'probabilities.npy')
    np.save(output_probs, probs)
    print(f'Probabilities .npy: {output_probs}')
    output_preds = os.path.join(args.output_dir, 'predictions_binary.npy')
    np.save(output_preds, predictions)
    print(f'Binary predictions .npy: {output_preds}')
    print('\nPrediction statistics:')
    print(f'  - Samples: {len(predictions)}')
    print(f'  - Avg predicted labels: {np.mean([np.sum(p) for p in predictions]):.2f}')
    print(f'  - Mean max probability: {np.mean([np.max(p) for p in probs]):.4f}')
    print('\nPer-label counts:')
    for i, label in enumerate(all_locations):
        count = np.sum(predictions[:, i])
        percentage = count / len(predictions) * 100
        print(f'  - {label}: {count} ({percentage:.2f}%)')
    print('\n' + '=' * 80)
    print('Done.')
    print('=' * 80)
if __name__ == '__main__':
    main()
