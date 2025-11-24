

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling
import matplotlib.pyplot as plt

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
hidden_dim = 128
emb_dim = 64
dropout = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 200

# load dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]
data = train_test_split_edges(data)  # prepare link-pred splits
print("Dataset:", dataset)
print("Num nodes:", data.num_nodes, "Num node features:", dataset.num_node_features)

# sklearn fallback: try import; if not present, use numpy auc
try:
    from sklearn.metrics import roc_auc_score, accuracy_score
    SKLEARN_AVAILABLE = True
    print("sklearn available")
except Exception as e:
    SKLEARN_AVAILABLE = False
    print("sklearn not available, using numpy fallback for AUC/accuracy.")

    def accuracy_score(y_true, y_pred):
        return (y_true == y_pred).astype(int).mean()

    def roc_auc_score(y_true, y_score):
        # compute ROC AUC using trapezoidal rule from TPR/FPR points
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        # sort by score desc
        desc = np.argsort(-y_score)
        y_true_sorted = y_true[desc]
        # compute TPR and FPR at each threshold
        P = y_true.sum()
        N = y_true.shape[0] - P
        if P == 0 or N == 0:
            return 0.5
        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)
        tpr = tps / P
        fpr = fps / N
        # prepend (0,0)
        tpr = np.concatenate(([0.0], tpr))
        fpr = np.concatenate(([0.0], fpr))
        # trapezoidal integral
        auc = np.trapz(tpr, fpr)
        return float(auc)

# Model
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def decode_embeddings(z, edge_index):
    src, dst = edge_index
    scores = (z[src] * z[dst]).sum(dim=1)
    return torch.sigmoid(scores)

def get_link_labels(pos_edge_index, neg_edge_index):
    pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.float32, device=device)
    neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=device)
    return torch.cat([pos_labels, neg_labels], dim=0)

# prepare data on device
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
val_pos_edge_index = data.val_pos_edge_index.to(device)
test_pos_edge_index = data.test_pos_edge_index.to(device)
adj_train_edge_index = train_pos_edge_index  # use training positive edges for message passing

model = GCNEncoder(dataset.num_node_features, hidden_dim, emb_dim, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
bce_loss = nn.BCELoss()

# logging containers
train_losses = []
val_aucs = []
test_aucs = []
val_accs = []
test_accs = []

best_val_auc = 0.0
best_test_auc_at_val = 0.0
best_epoch = -1

for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    z = model(x, adj_train_edge_index)

    # sample negative edges equal to positives
    neg_edge_index = negative_sampling(
        edge_index=adj_train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=train_pos_edge_index.size(1),
        method='sparse'
    ).to(device)

    pos_scores = decode_embeddings(z, train_pos_edge_index)
    neg_scores = decode_embeddings(z, neg_edge_index)
    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = get_link_labels(train_pos_edge_index, neg_edge_index)

    loss = bce_loss(scores, labels)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # validation & test evaluation
    model.eval()
    with torch.no_grad():
        z = model(x, adj_train_edge_index)
        # val
        val_neg = negative_sampling(edge_index=adj_train_edge_index,
                                    num_nodes=data.num_nodes,
                                    num_neg_samples=val_pos_edge_index.size(1),
                                    method='sparse').to(device)
        val_pos_scores = decode_embeddings(z, val_pos_edge_index).cpu().numpy()
        val_neg_scores = decode_embeddings(z, val_neg).cpu().numpy()
        val_scores = np.concatenate([val_pos_scores, val_neg_scores])
        val_labels = np.concatenate([np.ones(val_pos_scores.shape[0]), np.zeros(val_neg_scores.shape[0])])
        val_auc = roc_auc_score(val_labels, val_scores)
        val_pred = (val_scores >= 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_pred)

        # test
        test_neg = negative_sampling(edge_index=adj_train_edge_index,
                                     num_nodes=data.num_nodes,
                                     num_neg_samples=test_pos_edge_index.size(1),
                                     method='sparse').to(device)
        test_pos_scores = decode_embeddings(z, test_pos_edge_index).cpu().numpy()
        test_neg_scores = decode_embeddings(z, test_neg).cpu().numpy()
        test_scores = np.concatenate([test_pos_scores, test_neg_scores])
        test_labels = np.concatenate([np.ones(test_pos_scores.shape[0]), np.zeros(test_neg_scores.shape[0])])
        test_auc = roc_auc_score(test_labels, test_scores)
        test_pred = (test_scores >= 0.5).astype(int)
        test_acc = accuracy_score(test_labels, test_pred)

    val_aucs.append(val_auc)
    test_aucs.append(test_auc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_test_auc_at_val = test_auc
        best_epoch = epoch

    if epoch % 10 == 0 or epoch <= 5:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val AUC {val_auc:.4f} Acc {val_acc:.4f} | Test AUC {test_auc:.4f} Acc {test_acc:.4f}")

print("Training finished.")
print(f"Best val AUC {best_val_auc:.4f} at epoch {best_epoch}, test AUC at that epoch {best_test_auc_at_val:.4f}")

# final evaluation (fresh negative samples)
model.eval()
with torch.no_grad():
    z = model(x, adj_train_edge_index)
    test_neg = negative_sampling(edge_index=adj_train_edge_index,
                                 num_nodes=data.num_nodes,
                                 num_neg_samples=test_pos_edge_index.size(1),
                                 method='sparse').to(device)
    test_pos_scores = decode_embeddings(z, test_pos_edge_index).cpu().numpy()
    test_neg_scores = decode_embeddings(z, test_neg).cpu().numpy()
    final_test_scores = np.concatenate([test_pos_scores, test_neg_scores])
    final_test_labels = np.concatenate([np.ones(test_pos_scores.shape[0]), np.zeros(test_neg_scores.shape[0])])
    final_test_auc = roc_auc_score(final_test_labels, final_test_scores)
    final_test_acc = accuracy_score(final_test_labels, (final_test_scores >= 0.5).astype(int))
    print(f"Final Test AUC: {final_test_auc:.4f}, Final Test Acc: {final_test_acc:.4f}")

# Plotting curves and save
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, label='train loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss vs Epoch'); plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), val_aucs, label='val AUC')
plt.plot(range(1, epochs+1), test_aucs, label='test AUC')
plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.title('AUC vs Epoch'); plt.legend()

plt.tight_layout()
out_png = "cora_linkpred_curves.png"
plt.savefig(out_png, dpi=200)
print(f"Saved training curves to {out_png}")
plt.show()
