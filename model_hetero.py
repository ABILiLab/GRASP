import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, GINEConv, Set2Set, global_mean_pool,
    global_add_pool, HeteroConv, SAGEConv, Linear
)
from torch_geometric.data import HeteroData
from datetime import datetime
from metrics import *
import utils
import torch_scatter


# ================================
# Reusable training and validation helpers
# ================================
def train_valid(model, train_loader, valid_loader, epochs, patience,
                optimizer, scheduler, criterion, R, lambda_corr, thres,
                checkpoint_folder, isMultiLabel, device):

    import copy
    best_val_auc = 0.0
    best_model = None
    no_improvement_count = 0
    print(f"Best model save policy: best_val_auc; early stopping: best_val_auc")
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        train_loss = train_step(
            model, train_loader, optimizer, scheduler, criterion,
            R, lambda_corr, device
        )
        valid_loss, val_auc = valid_step(
            model, valid_loader, criterion, thres, isMultiLabel,
            device, epoch
        )

        scheduler.step()
        print("learning rate:", optimizer.param_groups[0]['lr'])
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

        # ---- Update best by AUC ----
        if val_auc > best_val_auc:
            print(f"New best AUC: {val_auc:.4f}")
            best_val_auc = val_auc
            best_model = copy.deepcopy(model)  # deep copy so training does not modify it
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # ---- Early stopping based on AUC ----
        if no_improvement_count >= patience:
            print(f"Early stopping at epoch {epoch+1}, "
                f"no AUC improvement for {patience} epochs.")
            break

    # ---- Save best model ----
    best_model = best_model.to(device)
    save_path = f"{checkpoint_folder}/model_best_auc_{best_val_auc:.4f}_asl_pos04_neg08_noR.pth"
    torch.save(best_model.state_dict(), save_path)
    print("Best model saved at:", save_path)

    return best_model


def train_step(model, train_loader, optimizer, scheduler,
               criterion, R, lambda_corr, device):

    print('Training...')
    model.train()
    counter = 0
    train_loss = 0

    for data in train_loader:
        counter += 1
        data = data.to(device)
        targets = data.y
        outputs = model(data)

        base_loss = criterion(outputs, targets)
        
        # Add label correlation matrix R as regularizer with weight lambda_corr
        if R is not None:
            # Predicted probabilities
            probs = torch.sigmoid(outputs)  # [B, L]
            # Correlation loss via utils.corr_loss_from_probs
            corr = utils.corr_loss_from_probs(probs, R)
            loss = base_loss + lambda_corr * corr
        else:
            loss = base_loss

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    train_total_loss = train_loss / counter
    return train_total_loss


def valid_step(model, val_loader, criterion,
               thres=0.5, isMultiLabel=True, device="cuda", epoch=0):

    print('----------------------Validating---------------------------')
    model.eval()

    counter = 0
    val_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            counter += 1
            data = data.to(device)
            targets = data.y.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)
            val_loss += loss.item()

            probs = torch.sigmoid(outputs)
            
            all_predictions.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())

    val_total_loss = val_loss / counter
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    metrics = evaluate_all_metrics(all_targets, all_predictions,
                                   isMultiLabel, thres)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return val_total_loss, metrics['Average AUC']

# label_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv  # small label graph -> GAT/GCN ok

class LabelGNN(nn.Module):
    def __init__(self, label_count=8, label_dim=128, conv_type='GAT',
                 n_layers=2, dropout=0.1, use_prior_adj=None):
        """
        use_prior_adj: None or tensor(edge_index) or 'fully_connected'
        """
        super().__init__()
        self.label_count = label_count
        self.label_dim = label_dim
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        # Learnable initial label embedding
        self.label_emb = nn.Parameter(torch.randn(label_count, label_dim) * 0.01)

        # Build small GNN
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            if conv_type == 'GAT':
                self.convs.append(GATConv(label_dim, label_dim//1, heads=1, concat=False))
            else:
                self.convs.append(GCNConv(label_dim, label_dim))
        # Optional: prepare edge_index externally for learnable adj/attention
        self.use_prior_adj = use_prior_adj  # None, edge_index tensor, or 'fully'

    def forward(self, edge_index=None):
        """
        Returns label embeddings [L, D].
        edge_index: optional (2, E) for label graph edges.
        If None and use_prior_adj == 'fully', builds fully-connected edge_index.
        """
        x = self.label_emb  # [L, D]
        if edge_index is None:
            if self.use_prior_adj is None:
                # treat as fully connected via GAT? we can build full edge_index
                src = []
                dst = []
                L = self.label_count
                for i in range(L):
                    for j in range(L):
                        if i != j:
                            src.append(i); dst.append(j)
                edge_index = torch.tensor([src, dst], dtype=torch.long, device=x.device)
            elif isinstance(self.use_prior_adj, torch.Tensor):
                edge_index = self.use_prior_adj.to(x.device)
            else:
                # fallback
                edge_index = None

        # apply GNN layers
        if edge_index is not None:
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
        else:
            # no edges -> just a linear transform (identity)
            x = x
        return x  # [L, D]



# ========================================================
#           RNA heterogeneous graph model
# ========================================================
class RNAHeteroModel(nn.Module):
    """
    RNA hierarchical model based on heterogeneous graph.

    Node types:
    - base: nucleotide nodes
    - loop: loop region nodes
    - stem: stem region nodes

    Edge types:
    - (base, adjacent, base): adjacent bases
    - (base, pair, base): base pairing
    - (base, belongs_to, loop)
    - (base, belongs_to, stem)
    - (loop, stem_connects, loop)
    """

    def __init__(self, num_base_features=30, num_loop_features=19,
                 num_stem_features=19, hidden_dim=128,
                 num_labels=10, conv_type="GAT",
                 n_conv_layers=4, dropout=0.2,
                 batch_norm=False, use_stem_nodes=True, use_label_graph=False):

        super(RNAHeteroModel, self).__init__()

        self.num_base_features = num_base_features
        # Loop/stem structure prior feature dim (low-dim, no label info). Forward pads/truncates to this for compatibility.
        self.num_loop_features = num_loop_features
        self.num_stem_features = num_stem_features

        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.conv_type = conv_type
        self.n_conv_layers = n_conv_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.use_stem_nodes = use_stem_nodes
        self.use_label_graph = use_label_graph

        # ---- Node feature encoders ----
        self.base_encoder = nn.Linear(num_base_features, hidden_dim)
        
        # Loop and stem use learnable initial features (not aggregated from base)
        self.loop_init_feature = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
        self.loop_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.loop_feat_encoder = nn.Linear(self.num_loop_features, hidden_dim)  # structure prior projection
        
        if use_stem_nodes:
            self.stem_init_feature = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)
            self.stem_encoder = nn.Linear(hidden_dim, hidden_dim)
            self.stem_feat_encoder = nn.Linear(self.num_stem_features, hidden_dim)  # structure prior projection

        # ---- Edge feature encoders ----
        self.edge_encoder_adjacent = nn.Linear(2, hidden_dim)
        self.edge_encoder_pair = nn.Linear(2, hidden_dim)

        # ---- Hetero graph convolution ----
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleDict()

        for i in range(n_conv_layers):
            conv_dict = {}

            # base → base (adjacent)
            conv_dict[('base', 'adjacent', 'base')] = self.get_conv_layer(
                hidden_dim, hidden_dim, conv_type, has_edge_attr=True,
                add_self_loops=True
            )

            # base → base (pair)
            conv_dict[('base', 'pair', 'base')] = self.get_conv_layer(
                hidden_dim, hidden_dim, conv_type, has_edge_attr=True,
                add_self_loops=True
            )

            # base ↔ loop
            conv_dict[('base', 'belongs_to', 'loop')] = self.get_conv_layer(hidden_dim, hidden_dim, conv_type, has_edge_attr=False, add_self_loops=False)

            conv_dict[('loop', 'belongs_to', 'base')] = self.get_conv_layer(hidden_dim, hidden_dim, conv_type, has_edge_attr=False, add_self_loops=False)

            # base ↔ stem
            if use_stem_nodes:
                conv_dict[('base', 'belongs_to', 'stem')] = self.get_conv_layer(hidden_dim, hidden_dim, conv_type, has_edge_attr=False, add_self_loops=False)

                conv_dict[('stem', 'belongs_to', 'base')] = self.get_conv_layer(hidden_dim, hidden_dim, conv_type, has_edge_attr=False, add_self_loops=False)

            # loop → loop
            conv_dict[('loop', 'stem_connects', 'loop')] = self.get_conv_layer(hidden_dim, hidden_dim, conv_type, has_edge_attr=False, add_self_loops=True)

            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

            if batch_norm:
                self.batch_norms[f'base_{i}'] = nn.LayerNorm(hidden_dim)
                self.batch_norms[f'loop_{i}'] = nn.LayerNorm(hidden_dim)
                if use_stem_nodes:
                    self.batch_norms[f'stem_{i}'] = nn.LayerNorm(hidden_dim)

        # ---- Graph-level pooling ----
        self.pooling_base = Set2Set(hidden_dim, processing_steps=2)
        self.pooling_loop = Set2Set(hidden_dim, processing_steps=2)
        if use_stem_nodes:
            self.pooling_stem = Set2Set(hidden_dim, processing_steps=2)

        self.dropout_layer = nn.Dropout(dropout)

        # ---- Extra sequence features (kmer, dacc) ----
        self.kmer_encoder = nn.Linear(1024, hidden_dim * 2)
        self.dacc_encoder = nn.Linear(72, hidden_dim * 2)

        # ---- Final classifier ----
        if use_stem_nodes:
            mlp_input_dim = hidden_dim * 10
        else:
            mlp_input_dim = hidden_dim * 8

        self.mlp_1 = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

        )

        self.mlp_2 = nn.Linear(hidden_dim, num_labels)

        if self.use_label_graph:
            self.label_gnn = LabelGNN(label_count=num_labels, label_dim=self.hidden_dim, conv_type='GAT', n_layers=2, dropout=0.1)

        self.sigmoid = nn.Sigmoid()

        self.cc_head = CCHead(
            input_dim=hidden_dim,
            num_labels=num_labels,
            hidden_dim=None,   # e.g. 64 or 128
            dropout=0.1
        )

    # ========================================================
    def get_conv_layer(self, in_channels, out_channels,
                       conv_type="GAT", has_edge_attr=False,
                       add_self_loops=True):

        if conv_type == "GCN":
            return GCNConv(
                in_channels, out_channels,
                add_self_loops=add_self_loops
            )

        elif conv_type == "GAT":
            return GATConv(
                in_channels, out_channels,
                heads=1, concat=False,
                add_self_loops=add_self_loops
            )

        elif conv_type == "GIN":
            return GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels)
                )
            )

        elif conv_type == "GINE":
            net = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            try:
                return GINEConv(net, add_self_loops=add_self_loops)
            except TypeError:
                return GINEConv(net)

        elif conv_type == "SAGE":
            return SAGEConv(in_channels, out_channels,
                            add_self_loops=add_self_loops)

        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

    # ========================================================
    def forward(self, hetero_data):
        def _pad_or_truncate(x, target_dim: int):
            if x is None:
                return None
            if x.dim() != 2:
                x = x.view(x.size(0), -1)
            d = x.size(1)
            if d == target_dim:
                return x
            if d < target_dim:
                pad = torch.zeros((x.size(0), target_dim - d), device=x.device, dtype=x.dtype)
                return torch.cat([x, pad], dim=1)
            return x[:, :target_dim]

        # ---- Node features ----
        x_base = self.base_encoder(hetero_data['base'].x)

        # Loop nodes: learnable initial features
        if 'loop' in hetero_data.node_types:
            num_loops = hetero_data['loop'].x.size(0)
            loop_init = self.loop_init_feature.expand(num_loops, -1)  # [num_loops, hidden_dim]
            x_loop = self.loop_encoder(loop_init)  # [num_loops, hidden_dim]
            loop_feat = _pad_or_truncate(hetero_data['loop'].x, self.num_loop_features)
            x_loop = x_loop + self.loop_feat_encoder(loop_feat)
        else:
            # Edge case: no loop nodes
            x_loop = None

        # Stem nodes: learnable initial features
        if self.use_stem_nodes and 'stem' in hetero_data.node_types:
            num_stems = hetero_data['stem'].x.size(0)
            stem_init = self.stem_init_feature.expand(num_stems, -1)  # [num_stems, hidden_dim]
            x_stem = self.stem_encoder(stem_init)  # [num_stems, hidden_dim]
            stem_feat = _pad_or_truncate(hetero_data['stem'].x, self.num_stem_features)
            x_stem = x_stem + self.stem_feat_encoder(stem_feat)
        else:
            x_stem = None

        # ---- Edge attributes ----
        edge_dict = {}

        edge_dict[('base', 'adjacent', 'base')] = self.edge_encoder_adjacent(hetero_data[('base', 'adjacent', 'base')].edge_attr)

        edge_dict[('base', 'pair', 'base')] = self.edge_encoder_pair( hetero_data[('base', 'pair', 'base')].edge_attr)

        # ---- x_dict ----
        x_dict = {'base': x_base}
        if x_loop is not None:
            x_dict['loop'] = x_loop
        if x_stem is not None:
            x_dict['stem'] = x_stem

        # ---- edge_index_dict ----
        edge_index_dict = {}
        for edge_type in hetero_data.edge_types:
            edge_index_dict[edge_type] = hetero_data[edge_type].edge_index

        # ---- Multi-layer convolution ----
        for i, conv in enumerate(self.convs):

            out_dict = conv(x_dict, edge_index_dict)

            # BN and activation
            for node_type in out_dict:
                out_dict[node_type] = self.batch_norms[f'{node_type}_{i}'](out_dict[node_type])
                out_dict[node_type] = F.relu(out_dict[node_type])
                out_dict[node_type] = self.dropout_layer(out_dict[node_type])

            # Residual connection
            for node_type in x_dict:
                out_dict[node_type] = out_dict[node_type] + x_dict[node_type]

            x_dict = out_dict

        # ---- Graph pooling ----
        batch_base = getattr(hetero_data['base'], 'batch', None)
        # if batch_base is None:
        #     batch_base = torch.zeros(
        #         x_dict['base'].size(0),
        #         dtype=torch.long,
        #         device=x_dict['base'].device
        #     )
        base_pool = self.pooling_base(x_dict['base'], batch_base)
        base_pool = self.dropout_layer(base_pool)

        # loop
        if 'loop' in x_dict:
            batch_loop = getattr(hetero_data['loop'], 'batch', None)
            loop_pool = self.pooling_loop(x_dict['loop'], batch_loop)
            loop_pool = self.dropout_layer(loop_pool)
        else:
            loop_pool = None
        

        # stem
        if self.use_stem_nodes and 'stem' in x_dict:
            batch_stem = getattr(hetero_data['stem'], 'batch', None)
            stem_pool = self.pooling_stem(x_dict['stem'], batch_stem)
            stem_pool = self.dropout_layer(stem_pool)
        else:
            stem_pool = None

        # ---- K-mer / DACC ----
        x_kmer = self.kmer_encoder(hetero_data.kmer)
        x_dacc = self.dacc_encoder(hetero_data.dacc)

        # ---- Feature fusion ----
        if self.use_stem_nodes and stem_pool is not None:
            g = torch.cat(
                [base_pool, loop_pool, stem_pool, x_kmer, x_dacc], dim=1
            )
        else:
            g = torch.cat(
                [base_pool, loop_pool, x_kmer, x_dacc], dim=1
            )

        # ---- MLP ----
        g = self.mlp_1(g)

        if self.use_label_graph:
            # Simple dot-product matching: logits[b,l] = <g[b], label_embs[l]>
            label_embs = self.label_gnn(edge_index=None)   # [L, D]; optionally pass label_edge_index (2,E)
            g = torch.matmul(g, label_embs.t())   # shape [B, L]
        else:
            g = self.mlp_2(g)
        return g

        # g = self.sigmoid(g)
        # return g

class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
        eps=1e-8
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits: [B, L] (raw logits)
        targets: [B, L] in {0,1}
        """
        probs = torch.sigmoid(logits)

        # ---- Asymmetric Clipping ----
        if self.clip is not None and self.clip > 0:
            probs = torch.clamp(probs, min=self.clip, max=1 - self.clip)

        pos_loss = targets * torch.log(probs + self.eps)
        neg_loss = (1 - targets) * torch.log(1 - probs + self.eps)

        # ---- Asymmetric Focusing ----
        pos_weight = torch.pow(1 - probs, self.gamma_pos)
        neg_weight = torch.pow(probs, self.gamma_neg)

        loss = - (pos_weight * pos_loss + neg_weight * neg_loss)
        return loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCHead(nn.Module):

    def __init__(self, input_dim, num_labels, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels

        # -------- Random label order for chain --------
        order = torch.randperm(num_labels)
        self.register_buffer("order", order)
        self.register_buffer("inv_order", torch.argsort(order))

        # -------- One classifier per label (chain) --------
        self.classifiers = nn.ModuleList()

        for i in range(num_labels):
            in_dim = input_dim + i  # concat previous i label predictions
            if hidden_dim is None:
                self.classifiers.append(nn.Linear(in_dim, 1))
            else:
                self.classifiers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1)
                    )
                )

    def forward(self, g):

        B = g.size(0)
        prev_probs = []      # sigmoid(logit) as conditioning features
        logits_chain = []    # logits in chain order

        for i, clf in enumerate(self.classifiers):
            if i == 0:
                x = g
            else:
                cond = torch.cat(prev_probs, dim=1)  # [B, i]
                x = torch.cat([g, cond], dim=1)

            logit = clf(x)                 # [B, 1]
            prob = torch.sigmoid(logit)    # no detach, allow gradient flow

            logits_chain.append(logit)
            prev_probs.append(prob)

        # [B, L] (chain order)
        logits_chain = torch.cat(logits_chain, dim=1)

        # -------- Restore original label order --------
        logits = logits_chain[:, self.inv_order]
        return logits
