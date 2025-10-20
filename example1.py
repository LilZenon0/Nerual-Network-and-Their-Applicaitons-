
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time

# ----------------- config -----------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

SIN_STEPS = 500         # initial sinusoid training steps (paper stops at k=500)
POST_STOP = 200         # simulate additional steps after stopping adaptation (for Fig12(a))
RAND_STEPS = 50000      # random training steps (paper uses 50000) — can set smaller for quick runs
TEST_STEPS = 500        # test length for Fig12(b)
LR = 0.25               # learning rate (paper used 0.25 for Example 1)
HIDDEN = 20             # network hidden units (N^{2,20,1})
DEVICE = torch.device('cpu')

# ------------------ plant and true f(u) ------------------
def f_true(u):
    return 0.6*np.sin(np.pi*u) + 0.3*np.sin(3*np.pi*u) + 0.1*np.sin(5*np.pi*u)

def plant_step(y_k, y_km1, u_k):
    return 0.3*y_k + 0.6*y_km1 + f_true(u_k)

# ------------------ network: 1 -> 20 -> 1 (sigmoid hidden, linear out) ------------------
class FNet(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.l1 = nn.Linear(1, hidden)
        self.act = nn.Sigmoid()
        self.l2 = nn.Linear(hidden, 1)
        # initialize small weights (optional)
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.1)
    def forward(self, u):
        x = self.l1(u)
        x = self.act(x)
        x = self.l2(x)
        return x

# ------------------ helper: manual optimizer step (SGD) ------------------
def sgd_manual_step(params, lr):
    # params: iterable of parameters (with .grad)
    for p in params:
        if p.grad is None:
            continue
        # in-place update
        p.data.add_( - lr * p.grad.data )

def zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

# ------------------ Phase A: sinusoid series-parallel training then parallel sim ------------------
def phase_a(net, lr=LR):
    net.train()
    loss_fn = nn.MSELoss()
    total = SIN_STEPS + POST_STOP
    y_p = np.zeros(total+2)    # plant outputs, store y_p(0), y_p(1), ...
    y_m = np.zeros_like(y_p)   # model outputs (series-parallel during training)
    u_seq = np.zeros(total)

    # initial conditions y_p[0]=y_p[1]=0, y_m same
    # 1) sinusoid training with adaptation
    for k in range(SIN_STEPS):
        u = np.sin(2*np.pi*k/250)
        u_seq[k] = u
        # plant step
        y_next = plant_step(y_p[k+1], y_p[k], u)
        y_p[k+2] = y_next
        # prepare target for network: f(u) = y_next - linear_part_using_true_y
        linear_part = 0.3*y_p[k+1] + 0.6*y_p[k]
        target = torch.tensor([[y_next - linear_part]], dtype=torch.float32, device=DEVICE)
        # forward and loss
        u_t = torch.tensor([[u]], dtype=torch.float32, device=DEVICE)
        pred = net(u_t)
        loss = loss_fn(pred, target)
        # backward
        loss.backward()
        # manual sgd step
        sgd_manual_step(net.parameters(), lr)
        # zero grads
        zero_grads(net.parameters())
        # update model output (series-parallel uses true y for linear part during training)
        y_m[k+2] = linear_part + pred.item()

    # 2) Stop adaptation at k = SIN_STEPS, switch to parallel simulation (no grad updates)
    net.eval()
    for idx in range(POST_STOP):
        k = SIN_STEPS + idx
        u = np.sin(2*np.pi*k/250)
        u_seq[k] = u
        # plant evolves
        y_p[k+2] = plant_step(y_p[k+1], y_p[k], u)
        # model in parallel mode uses its own history
        linear_model = 0.3*y_m[k+1] + 0.6*y_m[k]
        with torch.no_grad():
            fhat = net(torch.tensor([[u]], dtype=torch.float32, device=DEVICE)).item()
        y_m[k+2] = linear_model + fhat

    return u_seq, y_p, y_m

# ------------------ Phase B: random training (series-parallel) with manual updates ------------------
def random_training(net, lr=LR, steps=RAND_STEPS):
    net.train()
    loss_fn = nn.MSELoss()
    # initialize plant history
    y_p_last = 0.0
    y_p_prev = 0.0
    t0 = time()
    for k in range(steps):
        u = np.random.uniform(-1,1)
        # plant step
        y_next = plant_step(y_p_last, y_p_prev, u)
        # target for f(u) is y_next - linear_part(true y's)
        linear_part = 0.3*y_p_last + 0.6*y_p_prev
        target = torch.tensor([[y_next - linear_part]], dtype=torch.float32, device=DEVICE)
        u_t = torch.tensor([[u]], dtype=torch.float32, device=DEVICE)
        pred = net(u_t)
        loss = loss_fn(pred, target)
        # backward + manual update
        loss.backward()
        sgd_manual_step(net.parameters(), lr)
        zero_grads(net.parameters())
        # update plant history
        y_p_prev, y_p_last = y_p_last, y_next
        # optional progress print
        if (k+1) % 10000 == 0:
            print(f"Random train step {k+1}/{steps}, loss {loss.item():.6e}")
    print("Random training done. Time elapsed:", time() - t0)

# ------------------ Phase C: test (parallel sim) with sum-of-sinusoids ------------------
def test_parallel(net, test_steps=TEST_STEPS):
    net.eval()
    y_p = np.zeros(test_steps+2)
    y_m = np.zeros_like(y_p)
    us = np.zeros(test_steps)
    for k in range(test_steps):
        u = np.sin(2*np.pi*k/250) + np.sin(2*np.pi*k/25)
        us[k] = u
        y_p[k+2] = plant_step(y_p[k+1], y_p[k], u)
        linear_model = 0.3*y_m[k+1] + 0.6*y_m[k]
        with torch.no_grad():
            fhat = net(torch.tensor([[u]], dtype=torch.float32, device=DEVICE)).item()
        y_m[k+2] = linear_model + fhat
    return us, y_p, y_m

# ------------------ main ------------------
def main():
    net = FNet(HIDDEN).to(DEVICE)

    print("Phase A: sinusoid training (series-parallel) then parallel sim after stop.")
    u_seq, y_p_a, y_m_a = phase_a(net, lr=LR)

    # Plot show k roughly 300..700 (we simulated SIN_STEPS+POST_STOP)
    start = 300
    end = SIN_STEPS + POST_STOP  # ~700 if POST_STOP=200
    t = np.arange(start, end)
    plt.figure(figsize=(6,4))
    plt.plot(t, y_p_a[t+2], '-', label=r'$y_p$ (plant)', linewidth=1.0)
    plt.plot(t, y_m_a[t+2], '--', label=r'$\hat{y}_p$ (model)', linewidth=1.0)
    plt.xlabel(r'$k$'); plt.ylabel(r'$y_p$ and $\hat{y}_p$')
    plt.title('adaptation stopped at k=500 ')
    plt.legend(); plt.grid(True); plt.show()

    # Phase B: random training (may take time)
    print("Phase B: random training. This may take a while if RAND_STEPS is large.")
    random_training(net, lr=LR, steps=RAND_STEPS)

    # Phase C: test (parallel simulation)
    us, y_p_b, y_m_b = test_parallel(net, test_steps=TEST_STEPS)

    # Plot 
    t2 = np.arange(0, TEST_STEPS)
    plt.figure(figsize=(7,4))
    plt.plot(t2, y_p_b[t2+2], '-', linewidth=1.0, label=r'$y_p$ (plant)')
    plt.plot(t2, y_m_b[t2+2], '--', linewidth=1.0, label=r'$\hat{y}_p$ (model)')
    plt.xlim(0, TEST_STEPS)
    plt.xlabel(r'$k$'); plt.ylabel(r'$y_p$ and $\hat{y}_p$')
    plt.title('response after random-input identification ')
    plt.legend(); plt.grid(True); plt.show()


if __name__ == '__main__':
    main()
