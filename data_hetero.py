import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch,os
from torch_geometric.data import Data, InMemoryDataset, HeteroData
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import networkx as nx
import logging,pickle
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import re
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_undirected


class SequenceTokenizer:
    def __init__(self, sequences, labels, isMultiLabel, k=3):
        print('Tokenizing the data...')

        # Padding sequences with '-'
        padded_sequences = [('-' * (k // 2)) + seq + ('-' * (k // 2)) for seq in sequences]
        
        # Sub-sequences generation
        sub_sequences = [
            [seq[j - k // 2:j + k // 2 + 1] for j in range(k // 2, len(seq) - k // 2)]
            for seq in padded_sequences
        ]
        
        # Token-to-ID and ID-to-Token dictionaries
        token_count = 3
        id2token = ["[MASK]", "[PAD]", "[CLS]"]
        token2id = {"[MASK]": 0, "[PAD]": 1, "[CLS]": 2}
        
        # Update token dictionaries with sub-sequences
        for sub_seq in tqdm(sub_sequences):
            for token in sub_seq:
                if token not in token2id:
                    token2id[token] = token_count
                    id2token.append(token)
                    token_count += 1
        
        # Save token dictionaries
        self.id2token = id2token
        self.token2id = token2id
        self.token_count = token_count

        # Label-to-ID and ID-to-Label dictionaries
        label_count = 0
        id2label = []
        label2id = {}
        if isMultiLabel:
            for label_list in labels:
                for label in label_list:
                    if label not in label2id:
                        label2id[label] = label_count
                        id2label.append(label)
                        label_count += 1
        else:
            for label in labels:
                if label not in label2id:
                    label2id[label] = label_count
                    id2label.append(label)
                    label_count += 1

        # Save label dictionaries
        self.id2label = id2label
        self.label2id = label2id
        self.label_count = label_count

        # MultiLabelBinarizer for labels
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(labels)

        # OneHotEncoder for tokens
        self.ohe = OneHotEncoder()
        self.ohe.fit([[i] for i in range(self.token_count)])

    def tokenize_sub_sequences(self, sequence, embedding_dim):
        # Convert tokens in the sequence to their corresponding embeddings
        token_embeddings = F.adaptive_avg_pool1d(
            torch.tensor(
                self.ohe.transform([
                    [self.token2id.get(token, self.token2id["[MASK]"])]
                    for token in sequence
                ]).toarray(),
                dtype=torch.float32
            ).transpose(-1, -2),
            output_size=embedding_dim
        ).transpose(-1, -2).unsqueeze(0)

        return token_embeddings  # Shape: (1, embedding_dim, num_tokens)
    
    def save_tokenizer(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def get_base_pair_map(dot_bracket):
    stack = []
    pair_map = {}
    for i, c in enumerate(dot_bracket):
        if c == '(':
            stack.append(i)
        elif c == ')':
            j = stack.pop()
            pair_map[i] = j
            pair_map[j] = i
    return pair_map

def sinusoidal_position_encoding(num_nodes, pos_encoding_dim):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / pos_encoding_dim) for j in range(pos_encoding_dim)]
        for pos in range(num_nodes)
    ])
    # print("one position_enc!")
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) 
    return torch.tensor(position_enc, dtype=torch.float)

# Physicochemical properties (base -> molecular weight, dipole, H-bond donor/acceptor counts)
BASE_PHYSICOCHEM = {
    'A': [135.13, 3.0, 1, 2],
    'T': [112.09, 4.0, 1, 2],
    'C': [111.10, 6.5, 2, 1],
    'G': [151.13, 7.0, 2, 3]
}

BASES = ['A', 'T', 'C', 'G']
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}

# Base pair -> H-bond count
PAIR_HBOND = {('A', 'T'): 2, ('T', 'A'): 2,
              ('G', 'C'): 3, ('C', 'G'): 3,
              ('G', 'T'): 2, ('T', 'G'): 2}

word_to_ix = {"X": [0,0,0,0,0], "A": [0,1,0,0,0], "G": [0,0,1,0,0], "C": [0,0,0,1,0], "T": [0,0,0,0,1], "U": [0,0,0,0,1]}
N_to_EIIP = {"<PAD>": 0, "A": 0.1260, "G": 0.0806, "C": 0.1340, "T": 0.1335, "U": 0.1335}
N_to_NCP = {"<PAD>": [0,0,0], "A": [1,1,1], "G": [1,0,0], "C": [0,1,0], "T": [0,0,1], "U": [0,0,1]}

def ENAC_per_base(seq, window=5, alphabet="ACGT"):
    """Generate ENAC window frequency features for each position in the sequence."""
    L = len(seq)
    encodings = torch.zeros((L, len(alphabet)))
    for i in range(L):
        if i + window <= L:  # only compute for full windows
            window_seq = seq[i:i+window]
            count = Counter(window_seq)
            for j, aa in enumerate(alphabet):
                encodings[i, j] = count.get(aa, 0) / window
    return encodings

def prepare_sequence(seq,to_ix, pos_dim=10):
    idxs=[]
    pos_encode = sinusoidal_position_encoding(len(seq), pos_dim)
    enac = ENAC_per_base(seq, window=10)
    for j,char in enumerate(seq):
        ANF=[seq[0:j+1].count(seq[j])/(j+1)]  # accumulated nucleotide frequency
        
        # subidx=to_ix[char]+[N_to_EIIP[char]]+N_to_NCP[char]+ANF -> 20-dim feature
        subidx=to_ix[char]+ANF
        idxs.append(subidx)
    idx = torch.tensor(idxs, dtype=torch.float)
    return torch.concat([idx,pos_encode, enac],axis=1)


def build_hetero_rna_graph(sequence, dot_bracket):
    """
    Build heterogeneous graph with base, loop, stem node types.

    Node types:
    - base: nucleotide nodes
    - loop: loop region nodes
    - stem: stem region nodes (optional)

    Edge types:
    - ('base', 'adjacent', 'base'): adjacent bases
    - ('base', 'pair', 'base'): base pairing
    - ('base', 'belongs_to', 'loop'): base belongs to loop
    - ('base', 'belongs_to', 'stem'): base belongs to stem
    - ('loop', 'stem_connects', 'loop'): loops connected via stem
    """
    pair_map = get_base_pair_map(dot_bracket)
    loop_id_counter = [0]
    stem_id_counter = [0]
    all_pos = sinusoidal_position_encoding(len(sequence), 6).detach().cpu().numpy()
    
    # Node features
    base_features = prepare_sequence(sequence, word_to_ix)  # [num_bases, feat_dim]
    
    # Edge index lists
    base_adjacent_edges = []   # base -> base (adjacent)
    base_pair_edges = []       # base -> base (pair)
    base_to_loop_edges = []    # base -> loop (belongs_to)
    base_to_stem_edges = []    # base -> stem (belongs_to)
    loop_to_loop_edges = []    # loop -> loop (via stem)
    loop_stem_loop_map = {}    # (loop_i, loop_j) -> stem_id
    
    # Index maps: base indices per loop/stem
    loop_base_map = {}  # loop_id -> [base_indices]
    stem_base_map = {}  # stem_id -> [base_indices]

    # Structure prior features (low-dim, no label info)
    # loop: [loop_len_norm, adjacent_stem_count_norm, avg_pair_dist_norm, is_hairpin, is_internal, is_multiloop]
    # stem: [stem_len_norm, gc_ratio, avg_pair_dist_norm, near_5prime, near_3prime]
    loop_feat_map = {}  # loop_id -> np.array([6])
    stem_feat_map = {}  # stem_id -> np.array([5])
    
    def new_loop_id():
        loop_id = loop_id_counter[0]
        loop_id_counter[0] += 1
        return loop_id
    
    def new_stem_id():
        stem_id = stem_id_counter[0]
        stem_id_counter[0] += 1
        return stem_id
    
    def parse_loop(start, end, parent_loop_id=None):
        loop_seq = []
        loop_pos = []
        child_stems = []
        L = len(sequence)
        
        # Process loop region
        has_parent_pair = False
        parent_pair_dist = None
        if start - 1 > 0 and end + 1 < len(sequence):
            if dot_bracket[start - 1] == '(' and dot_bracket[end + 1] == ')' and pair_map[start - 1] == end + 1:
                loop_seq.extend([sequence[start - 1], sequence[end + 1]])
                loop_pos.extend([start - 1, end + 1])
                has_parent_pair = True
                parent_pair_dist = abs((start - 1) - (end + 1))
        
        i = start
        while i <= end:
            if dot_bracket[i] == '(':
                j = pair_map[i]
                stem = [(i, j)]
                i1, j1 = i + 1, j - 1
                while i1 < j1 and dot_bracket[i1] == '(' and pair_map[i1] == j1:
                    stem.append((i1, j1))
                    i1 += 1
                    j1 -= 1
                child_stems.append((i, j, stem))  # i,j = stem start/end indices; stem = list of base pairs
                loop_seq.extend([sequence[i], sequence[j]])
                loop_pos.extend([i, j])
                i = j + 1
            else:
                loop_seq.append(sequence[i])
                loop_pos.append(i)
                i += 1
        
        if len(loop_pos) == 0:
            return None
        
        loop_id = new_loop_id()
        loop_base_map[loop_id] = loop_pos

        # -------- Loop structure prior features --------
        child_cnt = len(child_stems)
        # Simplified: 0=hairpin, 1=internal, >=2=multiloop
        is_hairpin = 1.0 if child_cnt == 0 else 0.0
        is_internal = 1.0 if child_cnt == 1 else 0.0
        is_multiloop = 1.0 if child_cnt >= 2 else 0.0

        adjacent_stem_cnt = child_cnt + (1 if parent_loop_id is not None else 0)

        pair_dists = []
        if parent_pair_dist is not None:
            pair_dists.append(float(parent_pair_dist))
        for _, __, stem_pairs in child_stems:
            for a, b in stem_pairs:
                pair_dists.append(float(abs(a - b)))
        avg_pair_dist = float(np.mean(pair_dists)) if len(pair_dists) else 0.0

        loop_len = len(loop_pos)
        loop_len_norm = float(loop_len / max(1, L))
        adjacent_stem_cnt_norm = float(adjacent_stem_cnt / 10.0)  # empirical normalization
        avg_pair_dist_norm = float(avg_pair_dist / max(1, L))

        loop_feat_map[loop_id] = np.array(
            [loop_len_norm, adjacent_stem_cnt_norm, avg_pair_dist_norm,
             is_hairpin, is_internal, is_multiloop],
            dtype=np.float32
        )
        
        # Recursively process child loops and stems
        for _, _, stem_pairs in child_stems:
            i1 = stem_pairs[-1][0] + 1
            j1 = stem_pairs[-1][1] - 1
            inner_loop_id = parse_loop(i1, j1, loop_id)
            
            if inner_loop_id:
                # Create stem node: store base indices only
                stem_id = new_stem_id()
                stem_pos = [p for pair in stem_pairs for p in pair]
                stem_base_map[stem_id] = stem_pos

                # -------- Stem structure prior features --------
                # stem_length = number of base pairs
                stem_len = len(stem_pairs)
                stem_len_norm = float(stem_len / max(1, L))
                # GC ratio (from bases in stem_pos)
                if len(stem_pos) > 0:
                    gc = 0
                    for p in stem_pos:
                        c = sequence[p]
                        if c == 'G' or c == 'C':
                            gc += 1
                    gc_ratio = float(gc / len(stem_pos))
                else:
                    gc_ratio = 0.0
                # avg pairing distance
                stem_pair_dists = [float(abs(a - b)) for a, b in stem_pairs] if stem_pairs else []
                stem_avg_pair_dist = float(np.mean(stem_pair_dists)) if len(stem_pair_dists) else 0.0
                stem_avg_pair_dist_norm = float(stem_avg_pair_dist / max(1, L))
                # 5'/3' proximity (simple 10% threshold)
                if len(stem_pos) > 0:
                    min_pos = min(stem_pos)
                    max_pos = max(stem_pos)
                    near_5 = 1.0 if min_pos <= int(0.1 * L) else 0.0
                    near_3 = 1.0 if max_pos >= int(0.9 * (L - 1)) else 0.0
                else:
                    near_5 = 0.0
                    near_3 = 0.0

                stem_feat_map[stem_id] = np.array(
                    [stem_len_norm, gc_ratio, stem_avg_pair_dist_norm, near_5, near_3],
                    dtype=np.float32
                )
                
                # Loops connected via stem
                loop_to_loop_edges.append([loop_id, inner_loop_id])
                loop_stem_loop_map[(loop_id, inner_loop_id)] = stem_id
        
        return loop_id
    
    # Parse outermost loop
    root_loop_id = parse_loop(0, len(dot_bracket) - 1)
    
    # Build base-to-base edges (adjacent and pair)
    for i in range(len(sequence)):
        # Adjacent edges
        if i > 0:
            base_adjacent_edges.append([i-1, i])
        
        # Pair edges
        if i in pair_map:
            j = pair_map[i]
            if i < j:  # avoid duplicate
                base_pair_edges.append([i, j])
                base_pair_edges.append([j, i])  # bidirectional
    
    # Build base -> loop and base -> stem edges
    for loop_id, base_indices in loop_base_map.items():
        for base_idx in base_indices:
            base_to_loop_edges.append([base_idx, loop_id])
    
    for stem_id, base_indices in stem_base_map.items():
        for base_idx in base_indices:
            base_to_stem_edges.append([base_idx, stem_id])
    
    # Convert to tensor (handle empty edges)
    if len(base_adjacent_edges) > 0:
        base_adjacent_edges = torch.from_numpy(np.array(base_adjacent_edges, dtype=np.int64)).t().contiguous()
    else:
        base_adjacent_edges = torch.empty((2, 0), dtype=torch.long)
    
    if len(base_pair_edges) > 0:
        base_pair_edges = torch.from_numpy(np.array(base_pair_edges, dtype=np.int64)).t().contiguous()
    else:
        base_pair_edges = torch.empty((2, 0), dtype=torch.long)
    
    if len(base_to_loop_edges) > 0:
        base_to_loop_edges = torch.from_numpy(np.array(base_to_loop_edges, dtype=np.int64)).t().contiguous()
    else:
        base_to_loop_edges = torch.empty((2, 0), dtype=torch.long)
    
    if len(base_to_stem_edges) > 0:
        base_to_stem_edges = torch.from_numpy(np.array(base_to_stem_edges, dtype=np.int64)).t().contiguous()
    else:
        base_to_stem_edges = torch.empty((2, 0), dtype=torch.long)
    
    if len(loop_to_loop_edges) > 0:
        loop_to_loop_edges = torch.from_numpy(np.array(loop_to_loop_edges, dtype=np.int64)).t().contiguous()
    else:
        loop_to_loop_edges = torch.empty((2, 0), dtype=torch.long)
    
    # Build edge attributes (handle empty edges)
    if base_adjacent_edges.size(1) > 0:
        base_adjacent_attr = torch.ones(base_adjacent_edges.size(1), 2) * torch.tensor([[0., 1.]])
    else:
        base_adjacent_attr = torch.empty((0, 2), dtype=torch.float32)
    
    if base_pair_edges.size(1) > 0:
        base_pair_attr = torch.ones(base_pair_edges.size(1), 2) * torch.tensor([[1., 0.]])
    else:
        base_pair_attr = torch.empty((0, 2), dtype=torch.float32)
    
    # Convert base index lists to tensor for use in model; keep order by loop_id/stem_id for consistency
    num_loops = len(loop_base_map)
    num_stems = len(stem_base_map)

    loop_base_indices = [None] * num_loops
    loop_features = [None] * num_loops
    for lid, indices in loop_base_map.items():
        loop_base_indices[lid] = torch.tensor(indices, dtype=torch.long)
        loop_features[lid] = loop_feat_map.get(lid, np.zeros(6, dtype=np.float32))

    stem_base_indices = [None] * num_stems
    stem_features = [None] * num_stems
    for sid, indices in stem_base_map.items():
        stem_base_indices[sid] = torch.tensor(indices, dtype=torch.long)
        stem_features[sid] = stem_feat_map.get(sid, np.zeros(5, dtype=np.float32))

    loop_features = torch.tensor(np.stack(loop_features, axis=0), dtype=torch.float32) if num_loops > 0 else torch.zeros((0, 6), dtype=torch.float32)
    stem_features = torch.tensor(np.stack(stem_features, axis=0), dtype=torch.float32) if num_stems > 0 else torch.zeros((0, 5), dtype=torch.float32)
    
    return {
        'base_features': base_features,
        'loop_base_indices': loop_base_indices,
        'stem_base_indices': stem_base_indices,
        'loop_features': loop_features,
        'stem_features': stem_features,
        'base_adjacent_edges': base_adjacent_edges,
        'base_pair_edges': base_pair_edges,
        'base_to_loop_edges': base_to_loop_edges,
        'base_to_stem_edges': base_to_stem_edges,
        'loop_to_loop_edges': loop_to_loop_edges,
        'base_adjacent_attr': base_adjacent_attr,
        'base_pair_attr': base_pair_attr,
        'num_bases': len(sequence),
        'num_loops': num_loops,
        'num_stems': num_stems,
    }


class RNAHeteroGraphDataset(InMemoryDataset):
    """RNA heterogeneous graph dataset. tokenizer must be data_2.SequenceTokenizer, shared with run_train/run_predict."""
    def __init__(self, root='data', dataset='g1', view='train', 
                 df_data=None, tokenizer=None, foldings=None, fea_kmer=None, fea_dacc=None, 
                 transform=None, pre_transform=None, isMultiLabel=True, device="cuda"):
        super(RNAHeteroGraphDataset, self).__init__(root, transform, pre_transform)
        
        self.dataset = dataset
        self.view = view
        self.tokenizer = tokenizer
        self.foldings = foldings
        self.fea_kmer = fea_kmer
        self.fea_dacc = fea_dacc
        self.isMultiLabel = isMultiLabel
        self.device = device
        
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            loaded_data = torch.load(self.processed_paths[0], map_location=device)
            # Check loaded data format
            if isinstance(loaded_data, (list, tuple)):
                # New format: list of samples
                self.data_list = loaded_data
                self.slices = {'len': len(loaded_data)}
            else:
                # Old format: (data, slices)
                self.data, self.slices = loaded_data
                self.data_list = None
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(df_data)
            loaded_data = torch.load(self.processed_paths[0], map_location=device)
            if isinstance(loaded_data, (list, tuple)):
                self.data_list = loaded_data
                self.slices = {'len': len(loaded_data)}
            else:
                self.data, self.slices = loaded_data
                self.data_list = None
    
    def __len__(self):
        """Return dataset size."""
        if self.data_list is not None:
            return len(self.data_list)
        # Compatible with old format
        if 'y' in self.slices:
            y_slice = self.slices['y']
            if isinstance(y_slice, torch.Tensor):
                return len(y_slice) - 1
            elif isinstance(y_slice, (list, tuple)):
                return len(y_slice) - 1
        
        # If no y, infer from base nodes
        if 'base' in self.slices:
            base_slice = self.slices['base']
            # base may be nested dict {'x': tensor([...])}
            if isinstance(base_slice, dict) and 'x' in base_slice:
                if isinstance(base_slice['x'], torch.Tensor):
                    return len(base_slice['x']) - 1
            elif isinstance(base_slice, torch.Tensor):
                return len(base_slice) - 1
            elif isinstance(base_slice, (list, tuple)):
                return len(base_slice) - 1
        
        # If nothing found, raise to aid debugging
        raise ValueError(f"Invalid slices format: {type(self.slices)}. Keys: {list(self.slices.keys()) if hasattr(self.slices, 'keys') else 'N/A'}")
    
    def __getitem__(self, idx):
        """Get a single sample."""
        if self.data_list is not None:
            # New format: return from list
            return self.data_list[idx]
        else:
            # Old format: use InMemoryDataset default
            return super().__getitem__(idx)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [self.dataset + '_hetero_' + self.view + '.pt']
    
    def download(self):
        pass
    
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def collate(self, data_list):
        """Similar to data_2.py Batch.from_data_list; for hetero graph we create slices for all node/edge types."""
        from torch_geometric.data import Batch
        
        # Use Batch.from_data_list for hetero graph
        batch_data = Batch.from_data_list(data_list)
        
        # Explicitly merge y across samples
        batch_data.y = torch.cat([data.y for data in data_list], dim=0)
        
        # Handle other attributes
        batch_data.kmer = torch.stack([data.kmer for data in data_list], dim=0)
        batch_data.dacc = torch.stack([data.dacc for data in data_list], dim=0)
        batch_data.des = [data.des for data in data_list]
        batch_data.seqLens = [data.sLen for data in data_list]
        
        # Build slices dict: must include all node and edge types in batch_data (for separate())
        slices = {}
        
        # y slices (for dataset size)
        slices['y'] = torch.arange(len(data_list) + 1)
        
        # Slices for all node types (must match batch_data)
        for node_type in batch_data.node_types:
            # Count nodes of this type per graph
            num_nodes_list = []
            for data in data_list:
                if node_type in data.node_types:
                    num_nodes_list.append(data[node_type].x.size(0))
                else:
                    num_nodes_list.append(0)
            
            # Cumulative slices
            node_slices = torch.tensor([0] + [sum(num_nodes_list[:i+1]) for i in range(len(num_nodes_list))])
            slices[node_type] = {'x': node_slices}
        
        # Slices for all edge types
        for edge_type in batch_data.edge_types:
            # Count edges of this type per graph
            edge_counts = []
            for data in data_list:
                if edge_type in data.edge_types:
                    edge_counts.append(data[edge_type].edge_index.size(1))
                else:
                    edge_counts.append(0)
            
            # Edge slices
            edge_slices = {'edge_index': torch.tensor([0] + [sum(edge_counts[:i+1]) for i in range(len(edge_counts))])}
            
            # If edges have edge_attr, slice it too
            if hasattr(batch_data[edge_type], 'edge_attr'):
                edge_slices['edge_attr'] = edge_slices['edge_index']  # same count as edge_index
            
            slices[edge_type] = edge_slices
        
        slices['kmer'] = torch.arange(len(data_list) + 1)
        slices['dacc'] = torch.arange(len(data_list) + 1)
        slices['des'] = torch.arange(len(data_list) + 1)
        slices['seqLens'] = torch.arange(len(data_list) + 1)

        
        return batch_data, slices
    
    def process(self, df):
        data_list = []
        for i, row in enumerate(df.itertuples()):
            des = str(row.Description)
            seq_str = str(row.Sequence)
            
            if re.findall(r'[^AGCT]', seq_str.upper()):
                print("pass 1 records....")
                continue
            if seq_str not in self.foldings:
                print("pass 1 records....")
                continue
            
            dot_bracket_string = self.foldings[seq_str][0]
            
            # Label encoding
            if self.isMultiLabel:
                locations = str(row.Label).split(',')
                label_embedded = self.tokenizer.mlb.transform([locations])
            else:
                label_id = self.tokenizer.label2id[str(row.Label)]
                label_embedded = np.zeros(self.tokenizer.label_count)
                label_embedded[label_id] = 1
            
            y = torch.Tensor(label_embedded).view(1, self.tokenizer.label_count)
            
            # Build hetero graph
            graph_dict = build_hetero_rna_graph(seq_str, dot_bracket_string)
            
            # Create HeteroData
            hetero_data = HeteroData()
            
            # Node features: assign base features directly
            hetero_data['base'].x = graph_dict['base_features']
            
            # Loop/stem nodes: structure prior features (low-dim, no label info). base_indices kept for interpretability.
            if graph_dict['num_loops'] > 0:
                hetero_data['loop'].x = graph_dict.get('loop_features', torch.zeros(graph_dict['num_loops'], 6))
                # Store base indices per loop (for future use)
                hetero_data['loop'].base_indices = graph_dict['loop_base_indices']
            if graph_dict['num_stems'] > 0:
                hetero_data['stem'].x = graph_dict.get('stem_features', torch.zeros(graph_dict['num_stems'], 5))
                # Store base indices per stem (for future use)
                hetero_data['stem'].base_indices = graph_dict['stem_base_indices']
            
            # Edge indices and attributes: assign directly (errors will surface if mismatched)
            hetero_data['base', 'adjacent', 'base'].edge_index = graph_dict['base_adjacent_edges']
            hetero_data['base', 'adjacent', 'base'].edge_attr = graph_dict['base_adjacent_attr']
            
            hetero_data['base', 'pair', 'base'].edge_index = graph_dict['base_pair_edges']
            hetero_data['base', 'pair', 'base'].edge_attr = graph_dict['base_pair_attr']
            
            if graph_dict['num_loops'] > 0:
                # Bidirectional edges: base -> loop and loop -> base
                base_to_loop = graph_dict['base_to_loop_edges']
                loop_to_base = base_to_loop.flip(0)  # reverse direction
                hetero_data['base', 'belongs_to', 'loop'].edge_index = base_to_loop
                hetero_data['loop', 'belongs_to', 'base'].edge_index = loop_to_base
            
            if graph_dict['num_stems'] > 0:
                # Bidirectional edges: base -> stem and stem -> base
                base_to_stem = graph_dict['base_to_stem_edges']
                stem_to_base = base_to_stem.flip(0)  # reverse direction
                hetero_data['base', 'belongs_to', 'stem'].edge_index = base_to_stem
                hetero_data['stem', 'belongs_to', 'base'].edge_index = stem_to_base
            
            if graph_dict['num_loops'] > 1:  # need at least 2 loops for loop->loop edges
                loop_loop = to_undirected(graph_dict['loop_to_loop_edges'])
                hetero_data['loop', 'stem_connects', 'loop'].edge_index = loop_loop
            
            # Metadata
            hetero_data.y = y
            hetero_data.kmer = torch.tensor(self.fea_kmer[seq_str], dtype=torch.float32)
            hetero_data.dacc = torch.tensor(self.fea_dacc[seq_str], dtype=torch.float32)
            hetero_data.label = str(row.Label).split(',')
            hetero_data.sLen = len(seq_str)
            hetero_data.rowseq = seq_str
            hetero_data.dot_bracket_string = dot_bracket_string
            hetero_data.des = des
            
            data_list.append(hetero_data)
        
        # For InMemoryDataset, save list of single samples (not batched); __getitem__ returns one sample, DataLoader batches
        torch.save(data_list, self.processed_paths[0])


def hetero_collate_func(batch):
    """Hetero batch collate: manual batching to avoid Batch.from_data_list issues."""
    from torch_geometric.data import HeteroData
    
    if len(batch) == 0:
        raise ValueError("Empty batch")
    
    # Create new HeteroData as batch
    batch_data = HeteroData()
    
    # Step 1: collect all node and edge types
    all_node_types = set()
    all_edge_types = set()
    for data in batch:
        all_node_types.update(data.node_types)
        all_edge_types.update(data.edge_types)
    
    # Step 2: cumulative offsets per sample per node type
    node_type_offsets = {}  # {node_type: [offset_0, offset_1, ..., offset_n]}
    node_type_counts = {}   # {node_type: [count_0, count_1, ...]} for validation
    for node_type in all_node_types:
        offsets = [0]
        counts = []
        for data in batch:
            if node_type in data.node_types:
                num_nodes = data[node_type].x.size(0)
                counts.append(num_nodes)
                offsets.append(offsets[-1] + num_nodes)
            else:
                counts.append(0)
                offsets.append(offsets[-1])
        node_type_offsets[node_type] = offsets
        node_type_counts[node_type] = counts
    
    # Step 3: merge node features and batch indices; track base node offset per sample
    base_offsets = [0]
    for i, data in enumerate(batch):
        if 'base' in data.node_types:
            base_offsets.append(base_offsets[-1] + data['base'].x.size(0))
        else:
            base_offsets.append(base_offsets[-1])
    
    for node_type in all_node_types:
        node_features_list = []
        batch_indices = []
        base_indices_list = []  # base_indices for loop/stem nodes
        
        for i, data in enumerate(batch):
            if node_type in data.node_types:
                num_nodes = data[node_type].x.size(0)
                node_features_list.append(data[node_type].x)
                batch_indices.append(torch.full((num_nodes,), i, dtype=torch.long))
                
                # Handle base_indices (loop and stem only)
                if node_type in ['loop', 'stem']:
                    base_indices = getattr(data[node_type], 'base_indices', None)
                    if base_indices is not None and len(base_indices) > 0:
                        # Adjust base indices by current sample's base offset
                        adjusted_indices = []
                        for idx_tensor in base_indices:
                            adjusted_idx = idx_tensor + base_offsets[i]
                            adjusted_indices.append(adjusted_idx.to(data[node_type].x.device))
                        base_indices_list.extend(adjusted_indices)
                    else:
                        # If no base_indices, create empty tensor per node
                        device = data[node_type].x.device
                        base_indices_list.extend([torch.tensor([], dtype=torch.long, device=device)] * num_nodes)
        
        if node_features_list:
            batch_data[node_type].x = torch.cat(node_features_list, dim=0)
            batch_data[node_type].batch = torch.cat(batch_indices, dim=0)
            # Store adjusted base_indices
            if node_type in ['loop', 'stem']:
                batch_data[node_type].base_indices = base_indices_list
        else:
            # If a node type is missing in all samples, create empty features (edge types may still reference it)
            pass
    
    # Step 4: merge edge indices and attributes (apply offsets)
    for edge_type in all_edge_types:
        src_type, _, dst_type = edge_type
        src_offsets = node_type_offsets[src_type]
        dst_offsets = node_type_offsets[dst_type]
        src_counts = node_type_counts[src_type]
        dst_counts = node_type_counts[dst_type]
        
        all_edges = []
        all_attrs = []
        
        for i, data in enumerate(batch):
            if edge_type in data.edge_types:
                edges = data[edge_type].edge_index.clone()
                
                # Validate edge indices in range
                if edges.size(1) > 0:
                    max_src_idx = edges[0].max().item() if edges[0].numel() > 0 else -1
                    max_dst_idx = edges[1].max().item() if edges[1].numel() > 0 else -1
                    
                    # Check source indices in range
                    if max_src_idx >= src_counts[i]:
                        raise ValueError(
                            f"Sample {i} edge type {edge_type}: source index {max_src_idx} "
                            f"out of range for {src_type} count {src_counts[i]}"
                        )
                    # Check target indices in range
                    if max_dst_idx >= dst_counts[i]:
                        raise ValueError(
                            f"Sample {i} edge type {edge_type}: target index {max_dst_idx} "
                            f"out of range for {dst_type} count {dst_counts[i]}"
                        )
                
                # Apply offsets
                edges[0] += src_offsets[i]  # source
                edges[1] += dst_offsets[i]  # target
                all_edges.append(edges)
                
                # Handle edge attributes
                if hasattr(data[edge_type], 'edge_attr'):
                    all_attrs.append(data[edge_type].edge_attr)
        
        if all_edges:
            batch_data[edge_type].edge_index = torch.cat(all_edges, dim=1)
            if all_attrs:
                batch_data[edge_type].edge_attr = torch.cat(all_attrs, dim=0)
    
    # Step 5: global attributes (y, kmer, dacc, etc.)
    ys = []
    kmers = []
    daccs = []
    des_list = []
    seqLens_list = []
    rowseq_list = []
    dot_bracket_string_list = []
    
    for data in batch:
        # Handle y
        if hasattr(data, 'y') and data.y is not None:
            ys.append(data.y)
        elif hasattr(data, 'stores'):
            for store in data.stores:
                if 'y' in store:
                    ys.append(store['y'])
                    break
        else:
            raise ValueError(f"Missing 'y' in batch sample")
        
        # Handle other attributes
        if hasattr(data, 'kmer'):
            kmers.append(data.kmer)
        if hasattr(data, 'dacc'):
            daccs.append(data.dacc)
        if hasattr(data, 'des'):
            des_list.append(data.des)
        elif hasattr(data, 'label'):
            des_list.append(data.label)
        if hasattr(data, 'seqLens'):
            seqLens_list.append(data.seqLens)
        elif hasattr(data, 'sLen'):
            seqLens_list.append(data.sLen)
        # Handle sequence and dot_bracket
        if hasattr(data, 'rowseq'):
            rowseq_list.append(data.rowseq)
        if hasattr(data, 'dot_bracket_string'):
            dot_bracket_string_list.append(data.dot_bracket_string)
    
    batch_data.y = torch.cat(ys, dim=0)
    batch_data.kmer = torch.stack(kmers, dim=0)
    batch_data.dacc = torch.stack(daccs, dim=0)
    batch_data.des = des_list
    batch_data.seqLens = seqLens_list
    batch_data.rowseq = rowseq_list  # store as list
    batch_data.dot_bracket_string = dot_bracket_string_list  # store as list
    
    return batch_data

