import math
from typing import Optional

import torch
import torch.nn as nn

from drail.flow_matching import FlowMatchingModel


class _FMActionDistribution:
    """
    Distribution-like object with sample(), log_probs(action), entropy(), mode().
    Implements a conditional continuous normalizing flow using an FM vector field.
    - Condition c is a feature embedding provided at construction.
    - Action flow: a_{k+1} = a_k + f_theta(a_k, t_k, c) * dt, t in [0,1].
    - log pi(a|c) = log p0(z) - \int tr(\partial f / \partial a)(a_t, t, c) dt.
    """

    def __init__(
        self,
        flow_model: FlowMatchingModel,
        cond: torch.Tensor,
        action_dim: int,
        n_steps: int = 50,
        hutchinson_samples: int = 1,
        device: Optional[torch.device] = None,
        sanitize_actions: bool = False,
        low_bound: Optional[torch.Tensor] = None,
        high_bound: Optional[torch.Tensor] = None,
    ):
        self.flow_model = flow_model
        self.cond = cond  # [B, cond_dim]
        self.action_dim = action_dim
        self.n_steps = max(1, int(n_steps))
        self.hutchinson_samples = max(1, int(hutchinson_samples))
        self.device = device or cond.device
        self.sanitize_actions = sanitize_actions
        self.low_bound = low_bound
        self.high_bound = high_bound

        # Cache for entropy estimate when sampling
        self._last_log_prob: Optional[torch.Tensor] = None
        self._last_action: Optional[torch.Tensor] = None

    def _divergence_hutchinson(self, a: torch.Tensor, t: torch.Tensor, c: torch.Tensor, create_graph: bool) -> torch.Tensor:
        """
        Hutchinson trace estimator: tr(J) ~ E_e [ e^T J e ], where J = df/da.
        Returns shape [B, 1].
        """
        # Ensure gradients are enabled even if outer scope used no_grad (acting phase)
        with torch.enable_grad():
            a = a.detach().requires_grad_(True)
            total = 0.0
            for _ in range(self.hutchinson_samples):
                e = torch.randn_like(a)
                f_val = self.flow_model(a, c, t)
                dot = (f_val * e).sum()
                grad = torch.autograd.grad(dot, a, create_graph=create_graph, retain_graph=True)[0]
                total = total + (grad * e).sum(dim=1, keepdim=True)
        return total / float(self.hutchinson_samples)

    def _forward_sample_and_logp(self) -> (torch.Tensor, torch.Tensor):
        """
        Forward integrate from base z ~ N(0,I) to action a_1.
        Accumulate divergence integral to compute log_prob(a|c).
        Returns: (action, log_prob) with shapes [B, A], [B, 1].
        """
        batch_size = self.cond.size(0)
        dt = 1.0 / self.n_steps
        # Base distribution z ~ N(0, I)
        a = torch.randn(batch_size, self.action_dim, device=self.device)

        # Forward integrate without building a large graph
        with torch.no_grad():
            t_val = 0.0
            for _ in range(self.n_steps):
                t = torch.full((batch_size,), t_val, device=self.device)
                v = self.flow_model(a, self.cond, t)
                a = a + v * dt
                t_val += dt

        # Compute log_prob via reverse pass without higher-order graph
        self._next_log_prob_create_graph = False
        # Make sure log_probs runs with enabled grad context internally when needed
        logp = self.log_probs(a)
        return a.detach(), logp.detach()

    def sample(self) -> torch.Tensor:
        # For stability and correct log_prob, do a proper forward sample and cache log_prob.
        # Use the reverse-based log_prob for accuracy.
        a, logp = self._forward_sample_and_logp()
        # Optional sanitize only for algorithms that don't rely on exact log_prob (e.g., FPO)
        if self.sanitize_actions:
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            if self.low_bound is not None and self.high_bound is not None:
                a = torch.max(torch.min(a, self.high_bound.to(a.device)), self.low_bound.to(a.device))
            else:
                a = a.tanh()
        self._last_action = a
        self._last_log_prob = logp
        return a

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log pi(actions | cond) via reverse-time Euler from t=1 to 0.
        Returns shape [B, 1].
        """
        if (self._last_action is not None) and (actions is self._last_action):
            return self._last_log_prob

        batch_size = actions.size(0)
        dt = 1.0 / self.n_steps
        a = actions
        # Accumulate divergence integral along reverse trajectory using the same f
        div_int = torch.zeros(batch_size, 1, device=self.device)
        t_val = 1.0
        # Decide whether to build graph for PPO update
        create_graph = getattr(self, "_next_log_prob_create_graph", True)
        for _ in range(self.n_steps):
            t = torch.full((batch_size,), t_val, device=self.device)
            div = self._divergence_hutchinson(a, t, self.cond, create_graph=create_graph)
            div_int = div_int + div * dt
            v = self.flow_model(a, self.cond, t)
            a = a - v * dt  # reverse step
            t_val -= dt

        # Base log-density at recovered z ≈ a_0
        log_p0 = -0.5 * (a.pow(2).sum(dim=1, keepdim=True) + self.action_dim * math.log(2 * math.pi))
        # CNF formula: log p(a1) = log p(z0) - ∫ tr(J_f) dt
        logp = log_p0 - div_int
        # Reset the hint for next call
        self._next_log_prob_create_graph = True
        return logp

    def entropy(self) -> torch.Tensor:
        # Monte Carlo: H ≈ -E[log pi(a)] with a ~ pi; use cached sample if available.
        if self._last_log_prob is None:
            _ = self.sample()
        return -self._last_log_prob.squeeze(-1)

    def mode(self) -> torch.Tensor:
        # Use deterministic integration starting from z=0 as a proxy for mode.
        batch_size = self.cond.size(0)
        dt = 1.0 / self.n_steps
        a = torch.zeros(batch_size, self.action_dim, device=self.device)
        t_val = 0.0
        for _ in range(self.n_steps):
            t = torch.full((batch_size,), t_val, device=self.device)
            v = self.flow_model(a, self.cond, t)
            a = a + v * dt
            t_val += dt
        return a


class FMActionDistHead(nn.Module):
    """
    Dist head that constructs a flow-matching distribution from actor features.
    Compatible with DistActorCritic via get_dist_fn.
    """

    def __init__(
        self,
        actor_feature_dim: int,
        action_dim: int,
        n_steps: int = 50,
        hidden_units: int = 256,
        depth: int = 2,
        hutchinson_samples: int = 1,
        device: Optional[str] = None,
        sanitize_actions: bool = False,
        low_bound: Optional[torch.Tensor] = None,
        high_bound: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.hutchinson_samples = hutchinson_samples
        self.device = torch.device(device) if isinstance(device, str) else device
        self.sanitize_actions = sanitize_actions
        self.low_bound = low_bound
        self.high_bound = high_bound

        # Project actor features to a conditioning vector
        self.cond_proj = nn.Identity() if actor_feature_dim > 0 else nn.Identity()
        self.flow_model = FlowMatchingModel(
            cond_dim=actor_feature_dim,
            data_dim=action_dim,
            num_units=hidden_units,
            depth=depth,
            device=str(self.device) if self.device is not None else 'cuda',
        )

    def forward(self, actor_features: torch.Tensor):
        cond = self.cond_proj(actor_features)
        # Ensure flow model device matches features device
        feat_device = actor_features.device
        if next(self.flow_model.parameters()).device != feat_device:
            self.flow_model = self.flow_model.to(feat_device)
            # Keep internal device string consistent for any new tensors the model creates
            try:
                self.flow_model.device = str(feat_device)
            except Exception:
                pass
        return _FMActionDistribution(
            flow_model=self.flow_model,
            cond=cond,
            action_dim=self.action_dim,
            n_steps=self.n_steps,
            hutchinson_samples=self.hutchinson_samples,
            device=feat_device,
            sanitize_actions=self.sanitize_actions,
            low_bound=self.low_bound.to(feat_device) if isinstance(self.low_bound, torch.Tensor) else None,
            high_bound=self.high_bound.to(feat_device) if isinstance(self.high_bound, torch.Tensor) else None,
        )


