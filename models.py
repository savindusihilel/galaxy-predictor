import torch
import torch.nn as nn
import math

# ----------------------
# Joint PINN model (Stage C)
# ----------------------
class PINNJoint(nn.Module):
    def __init__(self, dim, context_dim=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(dim,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU()
        )

        self.m_head = nn.Sequential(nn.Linear(128,1), nn.Tanh())
        self.s_head = nn.Sequential(nn.Linear(128,1), nn.Tanh())

        self.m_logvar = nn.Linear(128,1)
        self.s_logvar = nn.Linear(128,1)

        self.context_net = nn.Sequential(
            nn.Linear(128,context_dim), nn.ReLU(),
            nn.Linear(context_dim,context_dim), nn.ReLU()
        )

        # Physical bounds (MUST match training)
        self.M_CENTER = 9.0
        self.M_SCALE  = 4.0
        self.S_CENTER = -1.5
        self.S_SCALE  = 3.5

    def forward(self, x):
        h = self.shared(x)

        m_mu = self.M_CENTER + self.M_SCALE * self.m_head(h).squeeze(-1)
        s_mu = self.S_CENTER + self.S_SCALE * self.s_head(h).squeeze(-1)

        m_logvar = self.m_logvar(h).squeeze(-1)
        s_logvar = self.s_logvar(h).squeeze(-1)

        ctx = self.context_net(h)

        return {
            "context": ctx,
            "mu_mass": m_mu,
            "logvar_mass": m_logvar,
            "mu_sfr": s_mu,
            "logvar_sfr": s_logvar
        }
