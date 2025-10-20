# example3_identify.py
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange

# ----------------- SETTINGS -----------------
SEED = 0
np.random.seed(SEED); torch.manual_seed(SEED)

# training lengths (set smaller for quick tests)
RAND_STEPS = 100000    # paper: 100000 (set 20000 for quick test)
TEST_STEPS = 500
LR = 0.1               # paper uses eta = 0.1
DEVICE = 'cpu'
H1 = 20                # first hidden layer size
H2 = 10                # second hidden layer size

# ------------ plant definition (Example 3) ------------
def f_true(yp):
    return yp / (1.0 + yp**2)

def g_true(u):
    return u**3

def plant_step(yp_k, yp_km1, u_k):
    return f_true(yp_k) + g_true(u_k)

# --------- network: N^3_{1,20,10,1} (two-hidden-layers) ----------
class MLP_1_20_10_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, H1),
            nn.Sigmoid(),
            nn.Linear(H1, H2),
            nn.Sigmoid(),
            nn.Linear(H2, 1)  # linear output
        )
    def forward(self, x):
        return self.net(x)

# ---------------- training (series-parallel) ----------------
def train_identifier(net_f, net_g, rand_steps=RAND_STEPS, lr=LR):
    opt = torch.optim.SGD(list(net_f.parameters()) + list(net_g.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    # initial plant outputs
    yp = np.zeros(2)  # yp[0], yp[1]
    for k in trange(rand_steps, desc='Training'):
        u = np.random.uniform(-2,2)   # paper uses U[-2,2]
        # simulate plant
        yp_next = plant_step(yp[-1], yp[-2], u)
        # targets for nets:
        # target_f = f_true(yp_k) = yp_next - g(u_k)  (since yp_next = f(yp_k) + g(u_k))
        target_f = yp_next - g_true(u)
        target_g = g_true(u)
        # train nets using series-parallel idea: we use true yp for target computation
        uf = torch.tensor([[yp[-1]]], dtype=torch.float32)
        ug = torch.tensor([[u]], dtype=torch.float32)
        tf = torch.tensor([[target_f]], dtype=torch.float32)
        tg = torch.tensor([[target_g]], dtype=torch.float32)
        pf = net_f(uf)
        pg = net_g(ug)
        loss = loss_fn(pf, tf) + loss_fn(pg, tg)
        opt.zero_grad(); loss.backward(); opt.step()
        # update plant history
        yp = np.array([yp[-1], yp_next])
    return net_f, net_g

# ----------------- testing / comparison -----------------
def test_and_plot(net_f, net_g, test_steps=TEST_STEPS):
    # initial conditions for plant and model
    yp = np.zeros(test_steps+2)
    ym = np.zeros_like(yp)  # model output using nets: yhat(k+1) = Nf[yp(k)] + Ng[u(k)]
    us = np.zeros(test_steps)
    for k in range(test_steps):
        u = np.sin(2*np.pi*k/25) + np.sin(2*np.pi*k/10)
        us[k] = u
        yp[k+2] = plant_step(yp[k+1], yp[k], u)
        with torch.no_grad():
            nf = net_f(torch.tensor([[yp[k+1]]], dtype=torch.float32)).item()
            ng = net_g(torch.tensor([[u]], dtype=torch.float32)).item()
        ym[k+2] = nf + ng

    t = np.arange(0, test_steps)
    plt.figure(figsize=(9,4))
    plt.plot(t, yp[t+2], '-', label='plant (true)')
    plt.plot(t, ym[t+2], '--', label='identified model')
    plt.title('Example 3: plant vs identified model (test)')
    plt.xlabel('k'); plt.ylabel('y')
    plt.legend(); plt.grid(True)
    plt.show()

    # compare true f,g with identified functions on grids
    us_grid = np.linspace(-2,2,201)
    yp_grid = np.linspace(-10,10,201)  # broad range for f
    with torch.no_grad():
        g_hat = net_g(torch.tensor(us_grid.reshape(-1,1), dtype=torch.float32)).numpy().ravel()
        f_hat = net_f(torch.tensor(yp_grid.reshape(-1,1), dtype=torch.float32)).numpy().ravel()
    plt.figure(figsize=(7,4))
    plt.plot(us_grid, g_true(us_grid), label='g true')
    plt.plot(us_grid, g_hat, '--', label='g_hat')
    plt.title('g(u) true vs identified')
    plt.legend(); plt.grid(True); plt.show()
    plt.figure(figsize=(7,4))
    plt.plot(yp_grid, f_true(yp_grid), label='f true')
    plt.plot(yp_grid, f_hat, '--', label='f_hat')
    plt.title('f(yp) true vs identified')
    plt.legend(); plt.grid(True); plt.show()

# ---------------- main ----------------
def main():
    net_f = MLP_1_20_10_1().to(DEVICE)
    net_g = MLP_1_20_10_1().to(DEVICE)
    print("Training Nf and Ng ...")
    net_f, net_g = train_identifier(net_f, net_g, rnd_steps:=RAND_STEPS)
    print("Testing and plotting ...")
    test_and_plot(net_f, net_g, TEST_STEPS)

if __name__ == '__main__':
    main()
