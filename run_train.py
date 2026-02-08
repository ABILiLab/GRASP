"""
Training script (heterogeneous graph): uses data_hetero, model_hetero; features: Kmer, DACC, LinearFold.
Configuration and flow follow run_hetero.py.
"""
import argparse
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader
from datetime import datetime

import utils
import data_hetero
import model_hetero
from metrics import *

parser = argparse.ArgumentParser(description="Train hetero model with Kmer, DACC and LinearFold features.")
parser.add_argument('--input_path', required=True, help='Training FASTA (header: >Desc|Label).')
parser.add_argument('--rna_type', default='lncRNA', choices=['lncRNA', 'mRNA'], help='RNA type for naming outputs and checkpoints (default: lncRNA).')
parser.add_argument('--feature_dir', default='./features', help='Directory to save/load folding, kmer, dacc pkl.')
parser.add_argument('--data_prepared_root', default='./data_prepared', help='Root for cached graph datasets.')
parser.add_argument('--checkpoint_folder', default='./model_saved', help='Root to save best model and tokenizer; actual path will be {checkpoint_folder}/{rna_type}.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--learningrate', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--epochs', type=int, default=200, help='Max epochs.')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patience.')
parser.add_argument('--k_folds', type=int, default=5, help='Number of CV folds.')
parser.add_argument('--kmer_k', type=int, default=5, help='Kmer length.')
parser.add_argument('--dacc_lag', type=int, default=2, help='DACC lag.')
parser.add_argument('--isMultiLabel', type=bool, default=True, help='Multi-label task.')
parser.add_argument('--isAutoThres', type=bool, default=False, help='Auto threshold.')
parser.add_argument('--device', default='cuda:0', help='Device (e.g. cuda:0 or cpu).')
parser.add_argument('--linearfold_cwd', default=None, help='CWD for LinearFold (default: reference/ under repo root).')
parser.add_argument('--ilearn_dir', default=None, help='iLearn root for DACC (default: reference/ilearn under repo root).')
parser.add_argument('--data_version', default='hetero_v1', help='Dataset version suffix for cached graphs.')
args = parser.parse_args()

_script_dir = os.path.dirname(os.path.abspath(__file__))
_reference_dir = os.path.join(_script_dir, 'reference')

input_path = args.input_path
rna_type = args.rna_type
feature_dir = args.feature_dir
data_prepared_root = os.path.abspath(args.data_prepared_root)
checkpoint_folder = os.path.join(args.checkpoint_folder, rna_type)
base_name = os.path.splitext(os.path.basename(input_path))[0]
batch_size = args.batch_size
learningrate = args.learningrate
epochs = args.epochs
patience = args.patience
k_folds = args.k_folds
kmer_k = args.kmer_k
dacc_lag = args.dacc_lag
isMultiLabel = args.isMultiLabel
isAutoThres = args.isAutoThres
device = args.device
data_version = args.data_version
linearfold_cwd = args.linearfold_cwd
if linearfold_cwd is None and os.path.isfile(os.path.join(_reference_dir, 'linearfold_v')):
    linearfold_cwd = _reference_dir
ilearn_dir = args.ilearn_dir

os.makedirs(feature_dir, exist_ok=True)
os.makedirs(checkpoint_folder, exist_ok=True)
feature_dir = os.path.abspath(feature_dir)
folding_path = os.path.join(feature_dir, f"{base_name}_linearfold.pkl")
kmer_path = os.path.join(feature_dir, f"{base_name}_{kmer_k}mer.pkl")
dacc_path = os.path.join(feature_dir, f"{base_name}_dacc.pkl")

print(f"RNA type: {rna_type}, feature base: {base_name}, checkpoint dir: {checkpoint_folder}")

# ---------- 1. Load data ----------
df = utils.read_fasta_to_df(input_path)
df['Sequence'] = df['Sequence'].str.replace('U', 'T')

# ---------- 2. Feature extraction: LinearFold, Kmer, DACC ----------
if os.path.isfile(folding_path):
    with open(folding_path, 'rb') as f:
        foldings = pickle.load(f)
    print("Loaded foldings from", folding_path)
else:
    out_fasta_dot = os.path.join(feature_dir, f"temp_{base_name}_dot")
    if linearfold_cwd:
        orig_cwd = os.getcwd()
        os.chdir(linearfold_cwd)
    try:
        utils.linear_fold(list(df['Sequence']), list(df['Description']), out_fasta_dot)
        utils.dot_fasta_to_pkl(out_fasta_dot + ".fasta", folding_path)
    finally:
        if linearfold_cwd:
            os.chdir(orig_cwd)
    with open(folding_path, 'rb') as f:
        foldings = pickle.load(f)
    print("LinearFold done, saved to", folding_path)

if os.path.isfile(kmer_path):
    with open(kmer_path, 'rb') as f:
        features_kmer = pickle.load(f)
    print("Loaded Kmer from", kmer_path)
else:
    kw = {'order': 'ACGT'}
    features_kmer = utils.Kmer(input_path, k=kmer_k, type="RNA", upto=False, normalize=True, **kw)
    with open(kmer_path, 'wb') as f:
        pickle.dump(features_kmer, f)
    print("Kmer done, saved to", kmer_path)

if os.path.isfile(dacc_path):
    with open(dacc_path, 'rb') as f:
        features_dacc = pickle.load(f)
    print("Loaded DACC from", dacc_path)
else:
    features_dacc = utils.compute_DACC(input_path, lag=dacc_lag, seq_type='RNA', ilearn_base_dir=ilearn_dir)
    with open(dacc_path, 'wb') as f:
        pickle.dump(features_dacc, f)
    print("DACC done, saved to", dacc_path)

# ---------- 3. Labels and split ----------
all_locations = set()
for i in df['Label']:
    all_locations.update(i.split(','))
all_locations = sorted(list(all_locations))
print("All locations:", all_locations)

k_folds_idx = utils.split_dataset_ensure_label(df, "Label", all_locations, k=k_folds, min_samples=10, random_state=41)
labels_list = [locs.split(",") for locs in list(df["Label"])]

# Tokenizer (data_hetero uses data_2.SequenceTokenizer)
tokenizer = data_hetero.SequenceTokenizer(df['Sequence'], labels_list, isMultiLabel=isMultiLabel)
print("Tokenizer classes:", tokenizer.mlb.classes_)

tokenizer_path = os.path.join(checkpoint_folder, "tokenizer.pkl")
tokenizer.save_tokenizer(tokenizer_path)
mlb_classes = list(tokenizer.mlb.classes_)
with open(os.path.join(checkpoint_folder, "mlb_classes.txt"), 'w') as f:
    f.write('\n'.join(mlb_classes))
print(f"Tokenizer and mlb_classes saved to {checkpoint_folder}")

# ---------- 4. Threshold ----------
if isMultiLabel:
    if isAutoThres:
        ref_df = utils.read_fasta_to_df(input_path)
        labels_list_ref = [locs.split(",") for locs in list(ref_df["Label"])]
        all_labels = [label.strip() for sublist in labels_list_ref for label in sublist]
        label_counter = Counter(all_labels)
        total_samples = len(ref_df)
        mean_freq = np.mean(list(label_counter.values())) / total_samples
        thresholds = {
            label: float(np.clip(0.5 + 0.8 * ((count / total_samples) - mean_freq), 0.1, 0.9))
            for label, count in label_counter.items()
        }
        thres = np.array([thresholds[lab] for lab in list(tokenizer.mlb.classes_)])
    else:
        thres = 0.5
else:
    thres = 0.5

# ---------- 5. Training config (aligned with run_hetero) ----------
model_params = {
    'num_base_features': 20,
    'num_loop_features': 20,
    'num_stem_features': 19,
    'hidden_dim': 128,
    'num_labels': None,
    'conv_type': "GAT",
    'n_conv_layers': 2,
    'dropout': 0.5,
    'batch_norm': True,
    'use_stem_nodes': True
}
optimizer_params = {
    'optimizer': 'AdamW',
    'learning_rate': learningrate,
    'weight_decay': 0.0,
}
scheduler_params = {
    'scheduler': 'CosineAnnealingWarmRestarts',
    'T_0': 10,
    'T_mult': 2,
    'eta_min': 1e-6
}
lambda_corr = 0.3

# ---------- 6. Training loop ----------
print(f"Start training at {datetime.now()}...")
for fold_idx, (train_idx, valid_idx) in enumerate(k_folds_idx):
    train_df = df.iloc[train_idx]
    valid_df = df.iloc[valid_idx]
    ds_name = f"{base_name}_hetero_fold_{fold_idx}_{data_version}"

    train_dataset = data_hetero.RNAHeteroGraphDataset(
        root=data_prepared_root,
        dataset=ds_name,
        view='train',
        df_data=train_df,
        tokenizer=tokenizer,
        foldings=foldings,
        fea_kmer=features_kmer,
        fea_dacc=features_dacc,
        isMultiLabel=isMultiLabel,
        device="cpu"
    )
    if hasattr(train_dataset, 'data') and hasattr(train_dataset.data, 'y'):
        all_y = train_dataset.data.y.cpu().numpy()
    else:
        train_labels = [locs.split(",") for locs in train_df['Label']]
        all_y = tokenizer.mlb.transform(train_labels)
    R = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_hetero.hetero_collate_func,
        num_workers=0
    )

    valid_dataset = data_hetero.RNAHeteroGraphDataset(
        root=data_prepared_root,
        dataset=ds_name,
        view='valid',
        df_data=valid_df,
        tokenizer=tokenizer,
        foldings=foldings,
        fea_kmer=features_kmer,
        fea_dacc=features_dacc,
        isMultiLabel=isMultiLabel,
        device="cpu"
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=data_hetero.hetero_collate_func,
        num_workers=0
    )

    model_params['num_labels'] = tokenizer.label_count
    model = model_hetero.RNAHeteroModel(
        num_base_features=model_params['num_base_features'],
        num_loop_features=model_params['num_loop_features'],
        num_stem_features=model_params['num_stem_features'],
        hidden_dim=model_params['hidden_dim'],
        num_labels=model_params['num_labels'],
        conv_type=model_params['conv_type'],
        n_conv_layers=model_params['n_conv_layers'],
        dropout=model_params['dropout'],
        batch_norm=model_params['batch_norm'],
        use_stem_nodes=model_params['use_stem_nodes'],
        use_label_graph=False
    )
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_params['learning_rate'],
        weight_decay=optimizer_params.get('weight_decay', 0.0)
    )
    criterion = model_hetero.AsymmetricLoss(
        gamma_pos=0.4,
        gamma_neg=0.8,
        clip=0.05,
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_params['T_0'],
        T_mult=scheduler_params['T_mult'],
        eta_min=scheduler_params['eta_min']
    )

    best_model = model_hetero.train_valid(
        model, train_loader, valid_loader, epochs, patience,
        optimizer, scheduler, criterion, R, lambda_corr, thres,
        checkpoint_folder, isMultiLabel=isMultiLabel, device=device
    )
    model_hetero.valid_step(best_model, valid_loader, criterion, thres, isMultiLabel, device)

    # Save with fixed name for run_predict
    fixed_name = "model_4lncRNA.pth" if rna_type == "lncRNA" else "model_4mRNA.pth"
    torch.save(best_model.state_dict(), os.path.join(checkpoint_folder, fixed_name))
    print(f"Also saved as {checkpoint_folder}/{fixed_name}")

tokenizer.save_tokenizer(tokenizer_path)
print(f"Training finished at {datetime.now()}.")
