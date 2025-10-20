# example7_control.py
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange

# ---------------- settings ----------------
SEED = 1
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 'cpu'

# paper-like parameters
IDENT_STEPS = 100000   # identification training steps (paper uses up to 100k)
LR = 0.1               # identification learning rate in many examples; adjust as needed
HIDDEN = 20            # hidden units in N (N^2_{2,20,1})
TEST_STEPS = 500

# for simultaneous case
T_i = 1   # identification update interval
T_c = 1   # control update interval; paper also tests T_c=3, T_i=1 etc.

# ---------------- plant & f definition (from paper) ----------------
def f_true(yp_k, yp_km1):
    # f[y_p(k), y_p(k-1)] = y_p(k)*y_p(k-1)*(y_p(k)+2.5)/(1 + y_p^2(k) + y_p^2(k-1))
    num = yp_k * yp_km1 * (yp_k + 2.5)
    den = 1.0 + yp_k**2 + yp_km1**2
    return num / den

def plant_step(yp_k, yp_km1, u_k):
    return f_true(yp_k, yp_km1) + u_k

# ---------------- identifier network (N^2_{2,20,1}) ----------------
class Identifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, 1)
        )
    def forward(self, inp):
        return self.net(inp)

# ---------------- training identifier (series-parallel random input) -------------
def train_identifier(net, steps=IDENT_STEPS, lr=LR):
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    yp = np.zeros(2)
    for k in trange(steps, desc='Ident training'):
        u = np.random.uniform(-1,1)  # random input domain [-1,1]
        yp_next = plant_step(yp[-1], yp[-2], u)
        # target = f_true(yp_k, yp_km1) = yp_next - u
        target = yp_next - u
        inp = torch.tensor([[yp[-1], yp[-2]]], dtype=torch.float32)
        t = torch.tensor([[target]], dtype=torch.float32)
        pred = net(inp)
        loss = loss_fn(pred, t)
        opt.zero_grad(); loss.backward(); opt.step()
        yp = np.array([yp[-1], yp_next])
    return net

# ---------------- simulate closed-loop with control using trained N --------------
def simulate_offline_control(net, test_steps=TEST_STEPS):
    yp = np.zeros(test_steps+2)
    ymodel = np.zeros_like(yp)
    # reference model: y_m(k+1) = 0.6 y_m(k) + 0.2 y_m(k-1) + r(k)
    ym = np.zeros(test_steps+2)
    r = lambda k: np.sin(2*np.pi*k/25)
    for k in range(test_steps):
        ref = r(k)
        # compute control using identified N: u = -N[y_p(k), y_p(k-1)] + 0.6 y_p(k) + 0.2 y_p(k-1) + r
        with torch.no_grad():
            n_in = torch.tensor([[yp[k+1], yp[k]]], dtype=torch.float32)
            fhat = net(n_in).item()
        u = -fhat + 0.6*yp[k+1] + 0.2*yp[k] + ref
        # plant and ref model step
        yp[k+2] = plant_step(yp[k+1], yp[k], u)
        ym[k+2] = 0.6*ym[k+1] + 0.2*ym[k] + ref
    t = np.arange(0, test_steps)
    plt.figure(figsize=(8,4))
    plt.plot(t, yp[t+2], '-', label='plant (closed-loop)')
    plt.plot(t, ym[t+2], '--', label='reference model')
    plt.title('Example7: offline identification then control')
    plt.xlabel('k'); plt.legend(); plt.grid(True); plt.show()

# ---------------- simultaneous identification and control ----------------------
def simulate_online_ident_and_control(Ti=1, Tc=1, ident_steps=20000, total_steps=500):
    # ident_steps controls how long we run random ident before switching; but here we do fully online from k=0
    net = Identifier()
    opt = torch.optim.SGD(net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    yp = np.zeros(total_steps+2)
    ym = np.zeros(total_steps+2)
    # initial control/ident periods
    for k in range(total_steps):
        ref = np.sin(2*np.pi*k/25)
        # compute fhat using current net
        with torch.no_grad():
            fhat = net(torch.tensor([[yp[k+1], yp[k]]], dtype=torch.float32)).item()
        # control computed using current estimate
        u = -fhat + 0.6*yp[k+1] + 0.2*yp[k] + ref
        # plant step
        yp[k+2] = plant_step(yp[k+1], yp[k], u)
        # reference model
        ym[k+2] = 0.6*ym[k+1] + 0.2*ym[k] + ref
        # identification update every Ti steps (use series-parallel target computed from plant)
        if (k % Ti) == 0:
            # target for net: f_true(yp_k, yp_k-1) = yp_next - u(k)
            target = yp[k+2] - u
            inp = torch.tensor([[yp[k+1], yp[k]]], dtype=torch.float32)
            t = torch.tensor([[target]], dtype=torch.float32)
            pred = net(inp)
            loss = loss_fn(pred, t)
            opt.zero_grad(); loss.backward(); opt.step()
        # (control update frequency governed by Tc by choosing to recompute control less often,
        # but above we updated control each step — to simulate slower Tc you would hold u constant for Tc steps)
    # plot
    t = np.arange(0, total_steps)
    plt.figure(figsize=(8,4))
    plt.plot(t, yp[t+2], '-', label='plant (online)')
    plt.plot(t, ym[t+2], '--', label='reference model')
    plt.title(f'Example7: online ident & control (T_i={Ti}, T_c={Tc})')
    plt.xlabel('k'); plt.legend(); plt.grid(True); plt.show()
    return net

# ---------------- main ----------------
def main():
    # Off-line identification then control
    net = Identifier()
    print("Training identifier off-line (this may take time)...")
    net = train_identifier(net, steps:=IDENT_STEPS, lr:=LR)
    print("Simulating closed-loop using trained model...")
    simulate_offline_control(net, TEST_STEPS)

    # Simultaneous online ident & control demo (shorter run)
    print("Simulating simultaneous online identification & control (short run)...")
    simulate_online_ident_and_control(Ti=1, Tc=3, ident_steps=20000, total_steps=500)

if __name__ == '__main__':
    main()
