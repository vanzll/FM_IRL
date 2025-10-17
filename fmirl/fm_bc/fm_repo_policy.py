import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the Flow Matching repo package is importable without installation
_FM_PKG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "FM_repository", "flow_matching")
)
if _FM_PKG_DIR not in sys.path:
    sys.path.insert(0, _FM_PKG_DIR)

try:
    from flow_matching.path.affine import AffineProbPath, CondOTProbPath
    from flow_matching.path.scheduler.scheduler import CondOTScheduler
    from flow_matching.utils.model_wrapper import ModelWrapper
except Exception as e:  # pragma: no cover - robust import fallback
    AffineProbPath = None
    CondOTProbPath = None
    CondOTScheduler = None
    ModelWrapper = None
    _FM_IMPORT_ERROR = e
else:
    _FM_IMPORT_ERROR = None


class _TimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim // 2), nn.SiLU(), nn.Linear(dim // 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t.unsqueeze(-1).float())


class _LabelCondVelocityNet(nn.Module):
    """MLP velocity field u_t(x | c) with time and label conditioning, where x = [s, a]."""

    def __init__(
        self,
        x_dim: int,
        cond_dim: int,
        hidden_dim: int,
        depth: int,
        time_embed_dim: int = 128,
    ):
        super().__init__()
        self.time_embed = _TimeEmbed(time_embed_dim)
        input_dim = x_dim + cond_dim + time_embed_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        x = torch.cat([x_t, cond, te], dim=-1)
        return self.net(x)


class _VelocityWrapper(ModelWrapper if ModelWrapper is not None else nn.Module):
    """Wraps the velocity model to the Flow Matching interface (x, t, **extras)."""

    def __init__(self, model: _ConditionalVelocityNet):
        super().__init__(model) if ModelWrapper is not None else nn.Module.__init__(self)
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        cond: torch.Tensor = extras.get("cond")
        return self.model(x_t=x, t=t, cond=cond)


class CreateAction:
    def __init__(self, action: torch.Tensor):
        self.action = action
        self.hxs = {}
        self.extra = {}
        self.take_action = action


from rlf.policies.basic_policy import BasicPolicy


class FMRepoFlowPolicy(BasicPolicy, nn.Module):
    """
    Behavior Cloning via Flow Matching using the official Flow Matching package.

    - Trains a conditional velocity field u_t(a | s)
    - Uses affine (CondOT) probability path to compute targets
    - Samples actions by ODE integration from noise to a
    """

    def __init__(
        self,
        n_steps: int = 100,
        action_dim: int = 2,
        state_dim: int = 6,
        num_units: int = 256,
        depth: int = 2,
        device: str = "cuda",
        is_stoch: bool = False,
    ):
        super().__init__()
        
        self.n_steps = n_steps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.is_stoch = is_stoch

        if _FM_IMPORT_ERROR is not None:
            raise ImportError(
                f"Failed to import Flow Matching package: {_FM_IMPORT_ERROR}. "
                "Ensure FM_repository/flow_matching is present or install torchdiffeq."
            )

        # Label-conditioned velocity over x = [s, a], condition c (in BC c=1)
        x_dim = state_dim + action_dim
        self.cond_dim = 1
        self.velocity = _LabelCondVelocityNet(
            x_dim=x_dim,
            cond_dim=self.cond_dim,
            hidden_dim=num_units,
            depth=depth,
        ).to(self.device)
        self.velocity_wrapper = _VelocityWrapper(self.velocity).to(self.device)

        # CondOT path: X_t = (1-t) X_0 + t X_1 with known velocity target
        self.path = AffineProbPath(CondOTScheduler())

    def init(self, obs_space, action_space, args):
        self.action_space = action_space
        self.obs_space = obs_space
        self.args = args

    def _sample_time(self, batch_size: int) -> torch.Tensor:
        return torch.rand(batch_size, device=self.device)

    def get_loss(self, expert_actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        expert_actions = expert_actions.to(self.device)
        states = states.to(self.device)
        batch_size = expert_actions.shape[0]

        # Build x1 = [s, a_expert], x0 = [s, noise_a] so s stays constant along the path
        noise_a = torch.randn_like(expert_actions)
        x1 = torch.cat([states, expert_actions], dim=-1)
        x0 = torch.cat([states, noise_a], dim=-1)
        t = self._sample_time(batch_size)

        # Sample along path and compute target velocity
        sample = self.path.sample(x_0=x0, x_1=x1, t=t)
        c = torch.ones(batch_size, self.cond_dim, device=self.device)
        pred_velocity = self.velocity_wrapper(sample.x_t, t, cond=c)

        return F.mse_loss(pred_velocity, sample.dx_t)

    @torch.no_grad()
    def get_action(self, state, add_state, rnn_hxs, mask, step_info):
        state_tensor = state.to(self.device)

        # Forward Euler integration on x = [s, a]; keep s clamped to input state
        batch = state_tensor.shape[0]
        a = torch.randn(batch, self.action_dim, device=self.device)
        x = torch.cat([state_tensor, a], dim=-1)
        dt = 1.0 / max(1, self.n_steps)
        t_val = 0.0
        c = torch.ones(batch, self.cond_dim, device=self.device)
        for _ in range(self.n_steps):
            t = torch.full((batch,), t_val, device=self.device)
            v = self.velocity_wrapper(x, t, cond=c)
            x = x + v * dt
            # Clamp state part to the given state to respect conditional generation
            x[:, : self.state_dim] = state_tensor
            t_val += dt

        a_out = x[:, self.state_dim :]
        return CreateAction(a_out)


