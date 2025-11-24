# citeseer_gcn.py
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# -----------------------
# Reproducibility
# -----------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# -----------------------
# Hyperparameters
# -----------------------
dataset_name = 'Citeseer'
hidden_dim = 64
dropout_p = 0.5
lr = 0.01
weight_decay = 5e-4
epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# Load dataset (Planetoid split)
# -----------------------
root = os.path.join(os.getcwd(), 'data', dataset_name)
dataset = Planetoid(root=root, name=dataset_name)
data = dataset[0].to(device)

print("Dataset:", dataset)
print("Number of graphs:", len(dataset))
print("Num nodes:", data.num_nodes)
print("Num node features:", dataset.num_node_features)
print("Num classes:", dataset.num_classes)

# -----------------------
# GCN model
# -----------------------
class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim, cached=True, normalize=True)
        self.conv2 = GCNConv(hidden_dim, out_dim, cached=True, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits

model = GCNNet(dataset.num_node_features, hidden_dim, dataset.num_classes, dropout=dropout_p).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# -----------------------
# Training / Evaluation helpers
# -----------------------
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # Use only training nodes for loss
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate():
    model.eval()
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)

    accs = {}
    for split, mask in [('train', data.train_mask), ('val', data.val_mask), ('test', data.test_mask)]:
        correct = preds[mask].eq(data.y[mask]).sum().item()
        total = mask.sum().item()
        accs[split] = correct / total if total > 0 else 0.0
    return accs, logits

# -----------------------
# Run training
# -----------------------
train_losses = []
train_accs = []
val_accs = []
test_accs = []

best_val = 0.0
best_test_at_val = 0.0
best_epoch = -1

for epoch in range(1, epochs + 1):
    loss = train()
    accs, _ = evaluate()
    train_losses.append(loss)
    train_accs.append(accs['train'])
    val_accs.append(accs['val'])
    test_accs.append(accs['test'])

    if accs['val'] > best_val:
        best_val = accs['val']
        best_test_at_val = accs['test']
        best_epoch = epoch

    if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train: {accs['train']:.4f} | Val: {accs['val']:.4f} | Test: {accs['test']:.4f}")

print("=== Training finished ===")
print(f"Best val acc {best_val:.4f} at epoch {best_epoch}, corresponding test acc {best_test_at_val:.4f}")

# final evaluation
final_accs, _ = evaluate()
print("Final accuracies:", final_accs)

# -----------------------
# Plot training curves (loss + val/test acc)
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, label='train loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), train_accs, label='train acc')
plt.plot(range(1, epochs+1), val_accs, label='val acc')
plt.plot(range(1, epochs+1), test_accs, label='test acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.tight_layout()
plt.savefig('citeseer_gcn_training_curves.png')
plt.show()
