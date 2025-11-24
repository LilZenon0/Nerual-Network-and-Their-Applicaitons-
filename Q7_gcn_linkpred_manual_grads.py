import torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
import random, numpy as np

# reproducibility
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]
data = train_test_split_edges(data)
print("Nodes:", data.num_nodes, "Features:", dataset.num_node_features)

# build normalized adjacency
def build_hatA(edge_index, num_nodes):
    A = torch.zeros((num_nodes,num_nodes),dtype=torch.float32)
    src,dst = edge_index
    A[src,dst]=1; A[dst,src]=1
    A += torch.eye(num_nodes)
    deg=A.sum(1)
    Dinv=torch.diag(1.0/torch.sqrt(deg))
    return Dinv@A@Dinv

hatA = build_hatA(to_undirected(dataset[0].edge_index), data.num_nodes).to(device)
X = data.x.to(device); N,d_in = X.shape
train_pos = data.train_pos_edge_index.to(device)

# params
d_h, d_out = 32, 16
W0 = torch.randn(d_in,d_h,device=device,requires_grad=True)*0.1
W1 = torch.randn(d_h,d_out,device=device,requires_grad=True)*0.1

def analytic_grads(X,hatA,W0,W1,pos_edge_index,neg_edge_index):
    AX=hatA@X
    M1=AX@W0
    H1=torch.relu(M1)
    B=hatA@H1
    Z=B@W1
    pos_s=(Z[pos_edge_index[0]]*Z[pos_edge_index[1]]).sum(1)
    neg_s=(Z[neg_edge_index[0]]*Z[neg_edge_index[1]]).sum(1)
    pos_p=torch.sigmoid(pos_s); neg_p=torch.sigmoid(neg_s)
    y=torch.cat([torch.ones_like(pos_p),torch.zeros_like(neg_p)])
    p=torch.cat([pos_p,neg_p])
    loss=F.binary_cross_entropy(p,y)
    gp=torch.cat([pos_p-1,neg_p-0])          # dL/ds
    dZ=torch.zeros_like(Z)
    # accumulate
    for (edges,g) in [(pos_edge_index,gp[:pos_s.size(0)]),
                      (neg_edge_index,gp[pos_s.size(0):])]:
        for k in range(edges.size(1)):
            i,j=edges[0,k].item(),edges[1,k].item()
            dZ[i]+=g[k]*Z[j]; dZ[j]+=g[k]*Z[i]
    dW1=B.t()@dZ
    dB=dZ@W1.t()
    dH1=hatA.t()@dB
    dM1=dH1*(M1>0).float()
    dW0=(hatA@X).t()@dM1
    return loss,dW0,dW1

# negative edges
neg=negative_sampling(edge_index=train_pos,num_nodes=N,
                      num_neg_samples=train_pos.size(1)).to(device)

# autograd path
AX=hatA@X; M1=AX@W0; H1=torch.relu(M1); B=hatA@H1; Z=B@W1
pos_s=(Z[train_pos[0]]*Z[train_pos[1]]).sum(1)
neg_s=(Z[neg[0]]*Z[neg[1]]).sum(1)
p=torch.sigmoid(torch.cat([pos_s,neg_s]))
y=torch.cat([torch.ones_like(pos_s),torch.zeros_like(neg_s)])
loss=F.binary_cross_entropy(p,y)
loss.backward()

# ensure grads exist
if W0.grad is None: W0.grad=torch.zeros_like(W0)
if W1.grad is None: W1.grad=torch.zeros_like(W1)

# analytic grads (clone/detach safely)
loss_a,dW0_a,dW1_a=analytic_grads(X,hatA,
                                  W0.clone().detach(),
                                  W1.clone().detach(),
                                  train_pos,neg)

# compare
print("Autograd loss:",float(loss),"Analytic loss:",float(loss_a))
print("ΔW0 max:",float((W0.grad-dW0_a).abs().max()))
print("ΔW1 max:",float((W1.grad-dW1_a).abs().max()))
