

import os
import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
import urllib.request
import zipfile
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Helper: dataset loader
# ---------------------------

def row_normalize(mx):
    """Row-normalize sparse matrix (scipy)"""
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.power(rowsum, -1.0)
    r_inv[np.isinf(r_inv)] = 0.0
    R = sp.diags(r_inv)
    return R.dot(mx)

def sparse_to_torch_sparse(adj):
    """Convert scipy coo_matrix to torch.sparse.FloatTensor"""
    adj_coo = adj.tocoo()
    indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64))
    values = torch.FloatTensor(adj_coo.data.astype(np.float32))
    shape = torch.Size(adj_coo.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def try_load_with_pyg(dataset_root, dataset_name):
    """Try to load dataset using torch_geometric.Planetoid (preferred)."""
    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_scipy_sparse_matrix
        print("[INFO] torch_geometric available: using Planetoid to download/load dataset.")
        root = os.path.join(dataset_root, "pyg_data")
        ds = Planetoid(root=root, name=dataset_name)
        data = ds[0]
        # features, labels
        features = torch.FloatTensor(data.x.numpy())
        labels = torch.LongTensor(data.y.numpy())
        # adjacency
        adj_sp = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocoo()
        adj_sp = adj_sp + sp.eye(adj_sp.shape[0])
        rowsum = np.array(adj_sp.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_norm = D_inv_sqrt.dot(adj_sp).dot(D_inv_sqrt).tocoo()
        adj_torch = sparse_to_torch_sparse(adj_norm)
        # splits: use masks included in PyG dataset object
        idx_train = np.nonzero(data.train_mask.numpy())[0].astype(np.int64)
        idx_val = np.nonzero(data.val_mask.numpy())[0].astype(np.int64)
        idx_test = np.nonzero(data.test_mask.numpy())[0].astype(np.int64)
        # ensure features are dense float tensor
        features = torch.FloatTensor(np.array(sp.csr_matrix(features.numpy()).todense(), dtype=np.float32))
        return adj_torch, features, labels, idx_train, idx_val, idx_test
    except Exception as e:
        print("[WARN] torch_geometric not usable:", str(e))
        return None

# Fallback: download raw planetoid files and construct dataset
PLANETOID_BASE = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data"

def download_raw_planetoid(dataset_root, dataset):
    """
    Download raw files (.content, .cites) from planetoid repo and save to data/<dataset>/
    Returns folder path.
    """
    folder = os.path.join(dataset_root, dataset)
    os.makedirs(folder, exist_ok=True)
    files = [f"{dataset}.content", f"{dataset}.cites"]
    for fname in files:
        url = f"{PLANETOID_BASE}/{fname}"
        out_path = os.path.join(folder, fname)
        if not os.path.exists(out_path):
            print(f"[INFO] Downloading {url} -> {out_path}")
            try:
                urllib.request.urlretrieve(url, out_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download {url}: {e}")
        else:
            print(f"[INFO] Raw file already exists: {out_path}")
    return folder

def build_npz_from_raw(folder, dataset):
    """
    Build a minimal .npz dataset consistent with the loader used earlier.
    Use the common Planetoid split heuristics (train first 140, val 200:700, test last 1000).
    Note: For exact reproduction of the paper's reported numbers, prefer the author's provided splits
    or use torch_geometric Planetoid which uses canonical splits.
    """
    content_path = os.path.join(folder, f"{dataset}.content")
    cites_path = os.path.join(folder, f"{dataset}.cites")
    if not (os.path.exists(content_path) and os.path.exists(cites_path)):
        raise FileNotFoundError("Raw Planetoid files missing.")

    # Parse content
    idx_features_labels = []
    with open(content_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            node_id = parts[0]
            feats = list(map(int, parts[1:-1]))
            label = parts[-1]
            idx_features_labels.append((node_id, feats, label))

    idx_map = {entry[0]: i for i, entry in enumerate(idx_features_labels)}
    features = np.array([entry[1] for entry in idx_features_labels], dtype=np.float32)
    labels_raw = [entry[2] for entry in idx_features_labels]
    unique_labels = sorted(list(set(labels_raw)))
    label_map = {c: i for i, c in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels_raw], dtype=np.int64)

    # Parse cites -> build adjacency
    edges = []
    with open(cites_path, 'r', encoding='utf-8') as f:
        for line in f:
            s, t = line.strip().split()
            if s in idx_map and t in idx_map:
                edges.append((idx_map[s], idx_map[t]))

    # make undirected
    edges_all = edges + [(j, i) for (i, j) in edges]
    if len(edges_all) == 0:
        raise RuntimeError("No edges found when building adjacency.")
    row = np.array([e[0] for e in edges_all], dtype=np.int32)
    col = np.array([e[1] for e in edges_all], dtype=np.int32)
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(features.shape[0], features.shape[0]))

    # create splits (common Planetoid split)
    n = features.shape[0]
    idx_train = np.arange(140, dtype=np.int64)
    idx_val = np.arange(200, 700, dtype=np.int64) if n >= 700 else np.arange(140, 140 + min(500, n-140), dtype=np.int64)
    if n >= 2708:
        idx_test = np.arange(1708, 2708, dtype=np.int64)
    else:
        # last 1000 as test if possible
        idx_test = np.arange(max(0, n - 1000), n, dtype=np.int64)

    # Save .npz (minimal)
    out = {
        'adj_data': adj.data, 'adj_indices': adj.indices, 'adj_indptr': adj.indptr, 'adj_shape': adj.shape,
        'feat_data': sp.csr_matrix(features).data, 'feat_indices': sp.csr_matrix(features).indices, 'feat_indptr': sp.csr_matrix(features).indptr, 'feat_shape': sp.csr_matrix(features).shape,
        'labels': labels,
        'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test
    }
    npz_path = os.path.join(folder, f"{dataset}.npz")
    np.savez(npz_path, **out)
    print(f"[INFO] Saved {npz_path}")
    return npz_path

def load_npz_dataset(dataset_root, dataset):
    """Load dataset from data/<dataset>/<dataset>.npz (the format created above)."""
    npz_path = os.path.join(dataset_root, dataset, f"{dataset}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No .npz found at {npz_path}")
    d = np.load(npz_path, allow_pickle=True)
    adj = sp.csr_matrix((d['adj_data'], d['adj_indices'], d['adj_indptr']), shape=tuple(d['adj_shape']))
    features = sp.csr_matrix((d['feat_data'], d['feat_indices'], d['feat_indptr']), shape=tuple(d['feat_shape']))
    labels = d['labels'].astype(np.int64)
    idx_train = d['idx_train'].astype(np.int64)
    idx_val = d['idx_val'].astype(np.int64)
    idx_test = d['idx_test'].astype(np.int64)

    # add self loops and normalize adjacency (renormalization trick)
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = D_inv_sqrt.dot(adj).dot(D_inv_sqrt).tocoo()
    adj_torch = sparse_to_torch_sparse(adj_norm)

    features = row_normalize(features).astype(np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    return adj_torch, features, labels, np.array(idx_train), np.array(idx_val), np.array(idx_test)

def load_dataset_auto(dataset_root, dataset_name):
    """Main loader: try PyG first, else download raw and build npz and load it."""
    # try PyG
    pyg_res = try_load_with_pyg(dataset_root, dataset_name)
    if pyg_res is not None:
        return pyg_res
    # else: try raw download + build
    try:
        folder = download_raw_planetoid(dataset_root, dataset_name)
        npz_path = build_npz_from_raw(folder, dataset_name)
        return load_npz_dataset(dataset_root, dataset_name)
    except Exception as e:
        raise RuntimeError("Failed to auto-load dataset. Either install torch_geometric or ensure network access to download raw files. Error: " + str(e))

# ---------------------------
# Model: Graph Convolution + 2-layer GCN
# ---------------------------

class GraphConvolution(nn.Module):
    """Simple graph convolution layer using precomputed normalized adjacency (sparse tensor)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: dense (N x F)
        support = torch.matmul(x, self.weight)  # N x out
        out = torch.sparse.mm(adj, support)     # sparse @ dense -> dense
        if self.bias is not None:
            out = out + self.bias
        return out

class TwoLayerGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

# ---------------------------
# Training / eval
# ---------------------------

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    return correct.sum() / len(labels)

def train_model(adj, features, labels, idx_train, idx_val, idx_test, device='cpu',
                lr=0.01, weight_decay=5e-4, hidden=16, dropout=0.5, max_epochs=200, patience=10, seed=42):
    # reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    nfeat = features.shape[1]
    nclass = int(labels.max().item()) + 1

    model = TwoLayerGCN(nfeat=nfeat, nhid=hidden, nclass=nclass, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()

    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)

    best_val_loss = float('inf')
    best_epoch = -1
    best_st = None

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(features, adj)
        loss_train = loss_fn(out[idx_train], labels[idx_train])
        acc_train = accuracy(out[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_val = model(features, adj)
            loss_val = loss_fn(out_val[idx_val], labels[idx_val]).item()
            acc_val = accuracy(out_val[idx_val], labels[idx_val]).item()

        print(f"Epoch {epoch:03d} | Train loss {loss_train.item():.4f} | Train acc {acc_train.item():.4f} "
              f"| Val loss {loss_val:.4f} | Val acc {acc_val:.4f}")

        # early stopping on val loss
        if loss_val < best_val_loss - 1e-6:
            best_val_loss = loss_val
            best_epoch = epoch
            best_st = {k: v.cpu() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            print(f"[INFO] Early stopping at epoch {epoch} (no val loss improvement for {patience} epochs).")
            break

    total_time = time.time() - t0
    if best_st is not None:
        model.load_state_dict(best_st)

    model.eval()
    with torch.no_grad():
        out_final = model(features, adj)
        test_loss = loss_fn(out_final[idx_test], labels[idx_test]).item()
        test_acc = accuracy(out_final[idx_test], labels[idx_test]).item()
        val_acc = accuracy(out_final[idx_val], labels[idx_val]).item()

    return {
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'val_acc': float(val_acc),
        'best_epoch': best_epoch,
        'time_s': total_time
    }

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed'],
                        help='Dataset name (cora/citeseer/pubmed)')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root folder')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=== Reproducing Kipf & Welling (2017) two-layer GCN ===")
    print(f"Dataset: {args.dataset} | device: {args.device}")

    try:
        adj, features, labels, idx_train, idx_val, idx_test = load_dataset_auto(args.data_root, args.dataset)
    except Exception as e:
        print("[ERROR] Failed to load dataset automatically:", e)
        print("Options: (1) install torch_geometric; (2) ensure network access to download raw planetoid files; (3) prepare data/<dataset>/<dataset>.npz manually")
        sys.exit(1)

    print("[INFO] Data loaded. N =", features.shape[0], "D =", features.shape[1], "classes =", int(labels.max().item())+1)
    # Hyperparameters from paper
    results = train_model(adj, features, labels, idx_train, idx_val, idx_test,
                          device=args.device, lr=0.01, weight_decay=5e-4, hidden=16, dropout=0.5,
                          max_epochs=200, patience=10, seed=args.seed)

    print("=== Final results ===")
    print(f"Dataset: {args.dataset}")
    print(f"Test accuracy: {results['test_acc']*100:.2f}%")
    print(f"Test loss: {results['test_loss']:.4f}")
    print(f"Val accuracy (best model): {results['val_acc']*100:.2f}%")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Training wall-clock (s): {results['time_s']:.2f}")
    
if __name__ == '__main__':
    main()
