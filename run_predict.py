"""
Prediction script (heterogeneous graph): uses data_hetero, model_hetero; config aligned with run_hetero/run_train.
"""
import argparse
import os
import pickle
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
import data_hetero
import model_hetero

parser = argparse.ArgumentParser(description="Predict with hetero model (Kmer, DACC, LinearFold).")
parser.add_argument('--input_path', required=True, help='Input FASTA to predict (unlabeled; header can be >id only).')
parser.add_argument('--output_path', required=True, help='Output CSV path (prob + binary rows).')
parser.add_argument('--rna_type', default='lncRNA', choices=['lncRNA', 'mRNA'], help='RNA type: use checkpoint under {checkpoint_folder}/{rna_type}/ (default: lncRNA).')
parser.add_argument('--checkpoint_folder', default='./model_saved', help='Root folder for checkpoints; actual path is {checkpoint_folder}/{rna_type}/.')
parser.add_argument('--model_path', default=None, help='Path to model .pth (default: auto under checkpoint by rna_type).')
parser.add_argument('--feature_dir', default='./features', help='Directory to save/load folding, kmer, dacc pkl.')
parser.add_argument('--data_prepared_root', default='./data_prepared', help='Root for cached graph dataset.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--kmer_k', type=int, default=5, help='Kmer length (must match training).')
parser.add_argument('--dacc_lag', type=int, default=2, help='DACC lag (must match training).')
parser.add_argument('--isMultiLabel', type=bool, default=True, help='Multi-label task.')
parser.add_argument('--isAutoThres', type=bool, default=False, help='Use auto threshold (need ref fasta).')
parser.add_argument('--thres', type=float, default=0.5, help='Fixed threshold when isAutoThres=False.')
parser.add_argument('--device', default='cuda:0', help='Device (e.g. cuda:0 or cpu).')
parser.add_argument('--linearfold_cwd', default=None, help='CWD for LinearFold (default: reference/ under repo root).')
parser.add_argument('--ilearn_dir', default=None, help='iLearn root for DACC (default: reference/ilearn under repo root).')
parser.add_argument('--data_version', default='hetero_v1', help='Dataset version suffix for cached graph.')
args = parser.parse_args()

_script_dir = os.path.dirname(os.path.abspath(__file__))
_reference_dir = os.path.join(_script_dir, 'reference')

input_path = args.input_path
output_path = args.output_path
rna_type = args.rna_type
checkpoint_folder = os.path.join(args.checkpoint_folder, rna_type)
model_path = args.model_path
feature_dir = args.feature_dir
data_prepared_root = os.path.abspath(args.data_prepared_root)
base_name = os.path.splitext(os.path.basename(input_path))[0]
batch_size = args.batch_size
kmer_k = args.kmer_k
dacc_lag = args.dacc_lag
isMultiLabel = args.isMultiLabel
isAutoThres = args.isAutoThres
thres = args.thres
device = args.device
data_version = args.data_version
linearfold_cwd = args.linearfold_cwd
if linearfold_cwd is None and os.path.isfile(os.path.join(_reference_dir, 'linearfold_v')):
    linearfold_cwd = _reference_dir
ilearn_dir = args.ilearn_dir

os.makedirs(feature_dir, exist_ok=True)
feature_dir = os.path.abspath(feature_dir)
print("RNA type:", rna_type, "| Checkpoint dir:", checkpoint_folder)
folding_path = os.path.join(feature_dir, f"{base_name}_linearfold.pkl")
kmer_path = os.path.join(feature_dir, f"{base_name}_{kmer_k}mer.pkl")
dacc_path = os.path.join(feature_dir, f"{base_name}_dacc.pkl")

# ---------- 1. Load test data ----------
test_df = utils.read_fasta_to_df(input_path)
test_df['Sequence'] = test_df['Sequence'].str.replace('U', 'T')
test_df['Label'] = ''

# ---------- 2. Features: LinearFold, Kmer, DACC ----------
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
        utils.linear_fold(list(test_df['Sequence']), list(test_df['Description']), out_fasta_dot)
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
else:
    kw = {'order': 'ACGT'}
    features_kmer = utils.Kmer(input_path, k=kmer_k, type="RNA", upto=False, normalize=True, **kw)
    with open(kmer_path, 'wb') as f:
        pickle.dump(features_kmer, f)

if os.path.isfile(dacc_path):
    with open(dacc_path, 'rb') as f:
        features_dacc = pickle.load(f)
else:
    features_dacc = utils.compute_DACC(input_path, lag=dacc_lag, seq_type='RNA', ilearn_base_dir=ilearn_dir)
    with open(dacc_path, 'wb') as f:
        pickle.dump(features_dacc, f)

# ---------- 3. Load tokenizer (same as data_hetero, from data_2) ----------
tokenizer = data_hetero.load_tokenizer(os.path.join(checkpoint_folder, "tokenizer.pkl"))
mlb_classes = list(tokenizer.mlb.classes_)
print("Labels (mlb_classes):", mlb_classes)

if isMultiLabel and isAutoThres:
    from collections import Counter
    ref_df = utils.read_fasta_to_df(input_path)
    labels_list_ref = [locs.split(",") for locs in list(ref_df["Label"])]
    all_labels = [label.strip() for sublist in labels_list_ref for label in sublist]
    label_counter = Counter(all_labels)
    total_samples = len(ref_df)
    mean_freq = np.mean(list(label_counter.values())) / total_samples
    thresholds = {
        label: float(np.clip(0.5 + 0.7 * ((count / total_samples) - mean_freq), 0.1, 0.9))
        for label, count in label_counter.items()
    }
    thres = np.array([thresholds[lab] for lab in mlb_classes])
elif isMultiLabel and isinstance(thres, float):
    thres = float(thres)
print("Threshold:", thres)

# ---------- 4. Test dataset and DataLoader (hetero graph) ----------
ds_name = f"{base_name}_predict_{data_version}"
test_dataset = data_hetero.RNAHeteroGraphDataset(
    root=data_prepared_root,
    dataset=ds_name,
    view='valid',
    df_data=test_df,
    tokenizer=tokenizer,
    foldings=foldings,
    fea_kmer=features_kmer,
    fea_dacc=features_dacc,
    isMultiLabel=isMultiLabel,
    device="cpu"
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=data_hetero.hetero_collate_func,
    num_workers=0
)

# ---------- 5. Load model and predict (model params aligned with run_hetero) ----------
if model_path is None:
    model_path = os.path.join(checkpoint_folder, "lncRNA_model_best_neg1.pth" if rna_type == "lncRNA" else "model_4mRNA.pth")
print("RNA type:", rna_type, "| Loading model from", model_path)


model_params = {
    'num_base_features': 20,
    'num_loop_features': 20,
    'num_stem_features': 19,
    'hidden_dim': 128,
    'num_labels': tokenizer.label_count,
    'conv_type': "GAT",
    'n_conv_layers': 2,
    'dropout': 0.5,
    'batch_norm': True,
    'use_stem_nodes': True
}
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
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

all_prob = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)  # logits
        probs = torch.sigmoid(out)
        all_prob.append(probs.cpu().numpy())
res_prob = np.vstack(all_prob)
if hasattr(thres, '__len__'):
    res_binary = (res_prob >= thres).astype(np.float32)
else:
    res_binary = (res_prob >= thres).astype(np.float32)

# ---------- 6. Write results ----------
descriptions = list(test_df['Description'])
os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ["Seq_ID"] + list(mlb_classes)
    writer.writerow(["Type"] + header)
    for i in range(len(res_prob)):
        sid = descriptions[i] if i < len(descriptions) else ""
        writer.writerow(["Prob"] + [sid] + res_prob[i].tolist())
    for i in range(len(res_binary)):
        sid = descriptions[i] if i < len(descriptions) else ""
        writer.writerow(["Binary"] + [sid] + res_binary[i].tolist())
print("Predictions saved to", output_path)

# ---------- 7. Remove generated feature files after prediction ----------
# for path in [folding_path, kmer_path, dacc_path]:
#     if os.path.isfile(path):
#         try:
#             os.remove(path)
#             print("Removed", path)
#         except OSError as e:
#             print("Could not remove", path, ":", e)
