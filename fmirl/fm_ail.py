import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time

from rlf.algos.il.base_irl import BaseIRLAlgo
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO, RegIRL_PPO
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.args import str2bool
import rlf.rl.utils as rutils

from fmirl.drail import DRAILDiscrim
from fmirl.flow_matching.flow_matching import FlowMatchingModel


class FMLabelCondDiscriminator(nn.Module):
    """
    Flow-Matching discriminator that mirrors DRAIL's interface:
    - Data x is the concatenated (state, action)
    - Condition c is the binary label vector (expert=1, agent=0)
    - Loss is per-sample FM loss computed at a single sampled time step

    This lets us keep DRAIL's training loop and reward computation identical,
    only changing the underlying L_diff implementation from DDPM to FM.
    """

    def __init__(self, state_dim: int, action_dim: int, args, base_net, num_units: int = 128):
        super().__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.base_net = False

        try:
            self.base_net = base_net.net.to(self.args.device)
        except Exception:
            self.base_net = False

        # FlowMatching model with label as condition, and (s,a) as data
        self.n_steps = getattr(self.args, 'fm_num_steps', 100)
        data_dim = state_dim + action_dim
        cond_dim = self.args.label_dim
        self.model = FlowMatchingModel(
            cond_dim=cond_dim,
            data_dim=data_dim,
            num_units=num_units,
            depth=self.args.discrim_depth,
            device=str(self.args.device),
        ).to(self.args.device)

    def _sample_t(self, batch_size: int) -> torch.Tensor:
        if getattr(self.args, 'sample_strategy', 'random') == "constant":
            step = getattr(self.args, 'sample_strategy_value', 0)
            step = min(max(int(step), 0), self.n_steps - 1)
            t_idx = torch.full((batch_size,), step, device=self.args.device)
        else:
            half = torch.randint(0, self.n_steps, size=(batch_size // 2,), device=self.args.device)
            t_idx = torch.cat([half, self.n_steps - 1 - half], dim=0)
        # scale to [0, 1]
        return t_idx.float() / float(self.n_steps)

    def _per_sample_fm_loss(self, x1: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample Flow Matching loss with a single time sample.
        x1: [B, data_dim], c: [B, cond_dim]
        returns: [B, 1]
        """
        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = self._sample_t(batch_size)  # [B] in [0,1]
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        true_velocity = x1 - x0
        pred_velocity = self.model(xt, c, t)
        loss_vec = (pred_velocity - true_velocity).pow(2).mean(dim=1, keepdim=True)
        return loss_vec

    def forward(self, state: torch.Tensor, action: torch.Tensor, label: float):
        if self.base_net:
            state = self.base_net(state)
        x1 = torch.cat([state, action], dim=1)
        batch_size = x1.shape[0]
        c = torch.full((batch_size, self.args.label_dim), float(label), device=self.args.device)
        return self._per_sample_fm_loss(x1, c)

    def compute_bc_loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """FM-BC loss on expert pairs with c=1 using x=[s,a]. Returns mean scalar."""
        if self.base_net:
            state = self.base_net(state)
        x1 = torch.cat([state, action], dim=1)
        batch_size = x1.shape[0]
        c = torch.ones(batch_size, self.args.label_dim, device=self.args.device)
        loss_vec = self._per_sample_fm_loss(x1, c)
        return loss_vec.mean()

    @torch.no_grad()
    def generate_sa(self, batch_size: int, num_steps: int = 10):
        """
        Generate expert-like (s_feature, a_expert) pairs by conditioning on expert label.
        Returns:
            s_feat: [B, state_dim] in the same feature space as the policy actor input
            a_expert: [B, action_dim]
        """
        if batch_size <= 0:
            return None
        c = torch.ones(batch_size, self.args.label_dim, device=self.args.device)
        x0 = torch.randn(batch_size, self.state_dim + self.action_dim, device=self.args.device)
        x_gen = self.model.generate(c=c, x0=x0, num_steps=num_steps)
        s_feat = x_gen[:, : self.state_dim]
        a_expert = x_gen[:, self.state_dim :]
        return s_feat, a_expert


def get_default_discrim(state_dim, action_dim, args, base_net, num_units=128):
    return FMLabelCondDiscriminator(state_dim, action_dim, args, base_net, num_units)


class FMAIL(NestedAlgo):
    """
    FM-AIL that is identical to DRAIL in all respects except L_diff implementation.
    We reuse DRAILDiscrim (training loop, reward computation, etc.) and only swap
    in the discriminator to use Flow Matching with label-conditioned inputs.
    """

    def __init__(self, agent_updater=None, get_discrim=None):
        if agent_updater is None:
            # Use RegIRL_PPO by default; if coef=0 this reduces to PPO behavior
            agent_updater = RegIRL_PPO()
        if get_discrim is None:
            get_discrim = get_default_discrim
        discrim_stage = DRAILDiscrim(get_discrim, policy=agent_updater)
        super().__init__([discrim_stage, agent_updater], 1)

    def get_add_args(self, parser):
        # Inherit DRAILDiscrim args via stage; additionally accept FM-specific flags
        super().get_add_args(parser)
        parser.add_argument('--fm-num-steps', type=int, default=100)
        parser.add_argument('--reward-update-freq', type=int, default=1)

    def init(self, policy, args):
        super().init(policy, args)
        # If the updater supports RegIRL, pass the discriminator for generation
        updater = self.modules[1]
        discrim_stage = self.modules[0]
        if hasattr(updater, 'set_regirl_discriminator') and hasattr(discrim_stage, 'discrim_net'):
            updater.set_regirl_discriminator(discrim_stage.discrim_net)



class DecoupledFMDiscriminator(nn.Module):
    """
    Flow-Matching discriminator without label conditioning, using two separate
    networks: one fit to expert (label=1) and one fit to agent (label=0).

    Keeps the same forward(state, action, label) interface so existing
    DRAIL/DRAILDiscrim code paths remain unchanged, but internally routes
    to the expert or agent FM model based on the label value.
    """

    def __init__(self, state_dim: int, action_dim: int, args, base_net, num_units: int = 128):
        super().__init__()
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.base_net = False

        try:
            self.base_net = base_net.net.to(self.args.device)
        except Exception:
            self.base_net = False

        # Two separate unconditioned FM models (cond_dim=0) over x=[s,a]
        self.n_steps = getattr(self.args, 'fm_num_steps', 100)
        data_dim = state_dim + action_dim
        self.expert_model = FlowMatchingModel(
            cond_dim=0,
            data_dim=data_dim,
            num_units=num_units,
            depth=self.args.discrim_depth,
            device=str(self.args.device),
        ).to(self.args.device)
        self.agent_model = FlowMatchingModel(
            cond_dim=0,
            data_dim=data_dim,
            num_units=num_units,
            depth=self.args.discrim_depth,
            device=str(self.args.device),
        ).to(self.args.device)

    def _sample_t(self, batch_size: int) -> torch.Tensor:
        if getattr(self.args, 'sample_strategy', 'random') == "constant":
            step = getattr(self.args, 'sample_strategy_value', 0)
            step = min(max(int(step), 0), self.n_steps - 1)
            t_idx = torch.full((batch_size,), step, device=self.args.device)
        else:
            half = torch.randint(0, self.n_steps, size=(batch_size // 2,), device=self.args.device)
            t_idx = torch.cat([half, self.n_steps - 1 - half], dim=0)
        return t_idx.float() / float(self.n_steps)

    def _per_sample_fm_loss(self, model: FlowMatchingModel, x1: torch.Tensor) -> torch.Tensor:
        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = self._sample_t(batch_size)
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        true_velocity = x1 - x0
        # empty condition tensor with 0 features
        c = xt.new_zeros(batch_size, 0)
        pred_velocity = model(xt, c, t)
        loss_vec = (pred_velocity - true_velocity).pow(2).mean(dim=1, keepdim=True)
        return loss_vec

    def forward(self, state: torch.Tensor, action: torch.Tensor, label: float):
        if self.base_net:
            state = self.base_net(state)
        x1 = torch.cat([state, action], dim=1)
        # choose expert vs agent branch by label
        label_val = float(label) if not torch.is_tensor(label) else float(label.mean().item())
        if label_val >= 0.5:
            return self._per_sample_fm_loss(self.expert_model, x1)
        else:
            return self._per_sample_fm_loss(self.agent_model, x1)

    def compute_bc_loss(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Optional FM-BC on expert branch only (used by training loop if present)."""
        if self.base_net:
            state = self.base_net(state)
        x1 = torch.cat([state, action], dim=1)
        loss_vec = self._per_sample_fm_loss(self.expert_model, x1)
        return loss_vec.mean()

    @torch.no_grad()
    def generate_sa(self, batch_size: int, num_steps: int = 10):
        if batch_size <= 0:
            return None
        # empty condition tensor for generation
        c = torch.zeros(batch_size, 0, device=self.args.device)
        x0 = torch.randn(batch_size, self.state_dim + self.action_dim, device=self.args.device)
        x_gen = self.expert_model.generate(c=c, x0=x0, num_steps=num_steps)
        s_feat = x_gen[:, : self.state_dim]
        a_expert = x_gen[:, self.state_dim :]
        return s_feat, a_expert


def get_decoupled_discrim(state_dim, action_dim, args, base_net, num_units=128):
    return DecoupledFMDiscriminator(state_dim, action_dim, args, base_net, num_units)


class Decoupled_FMAIL(NestedAlgo):
    """
    FM-AIL variant with decoupled discriminators (no label conditioning):
    - Two separate FM models for expert and agent branches
    - Reuses DRAILDiscrim loop by exposing the same interface
    """

    def __init__(self, agent_updater=None, get_discrim=None):
        if agent_updater is None:
            agent_updater = RegIRL_PPO()
        if get_discrim is None:
            get_discrim = get_decoupled_discrim
        discrim_stage = DRAILDiscrim(get_discrim, policy=agent_updater)
        super().__init__([discrim_stage, agent_updater], 1)

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--fm-num-steps', type=int, default=100)
        parser.add_argument('--reward-update-freq', type=int, default=1)

    def init(self, policy, args):
        super().init(policy, args)
        updater = self.modules[1]
        discrim_stage = self.modules[0]
        if hasattr(updater, 'set_regirl_discriminator') and hasattr(discrim_stage, 'discrim_net'):
            updater.set_regirl_discriminator(discrim_stage.discrim_net)
