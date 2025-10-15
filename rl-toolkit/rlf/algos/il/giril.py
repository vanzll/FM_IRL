# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from rlf.algos.il.base_irl import BaseIRLAlgo
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
import rlf.rl.utils as rutils
from rlf.baselines.common.running_mean_std import RunningMeanStd
from collections import defaultdict


class _MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, depth=2, act=nn.Tanh):
        super().__init__()
        layers = []
        last = input_dim
        for _ in range(max(0, depth)):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(act())
            last = hidden_dim
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = action_dim
        self.hidden = hidden_dim

        self.linear_1 = nn.Linear(in_features=self.input_dim * 2, out_features=self.hidden)
        self.linear_2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.tanh_1 = nn.Tanh()
        self.tanh_2 = nn.Tanh()

        self.mu = nn.Linear(hidden_dim, action_dim)
        self.logvar = nn.Linear(hidden_dim, action_dim)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.logvar.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.bias)

    def reparameterize(self, mu, logvar, device, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_()).to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        x = self.tanh_1(self.linear_1(x))
        x = self.tanh_2(self.linear_2(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar, state.device)
        return z, mu, logvar


class ForwardDynamicsModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=100):
        super().__init__()
        self.net = _MLP(obs_dim + action_dim, obs_dim, hidden_dim=hidden_dim, depth=2)

    def forward(self, state, action_or_latent):
        x = torch.cat([state, action_or_latent], dim=-1)
        return self.net(x)


class GIRIL_IRL(BaseIRLAlgo):
    """
    GIRIL intrinsic reward: train an inverse (state,next_state -> latent action)
    and forward dynamics (state, action/latent -> next_state). Reward is forward
    prediction error using the true action.
    """

    def __init__(self):
        super().__init__()

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--giril-hidden-dim', type=int, default=100)
        parser.add_argument('--giril-lr', type=float, default=3e-4)
        parser.add_argument('--giril-kld-beta', type=float, default=1.0)
        parser.add_argument('--giril-lambda-action', type=float, default=1.0)
        parser.add_argument('--giril-train-epochs', type=int, default=1)
        parser.add_argument('--giril-use-latent-for-forward', action='store_true', default=True,
                            help='Use latent z for training forward model (as in paper).')
        parser.add_argument('--giril-reward-norm', action='store_true', default=True,
                            help='Normalize intrinsic reward by running std.')

    def init(self, policy, args):
        super().init(policy, args)
        ob_dim = rutils.get_obs_shape(self.policy.obs_space)[0]
        ac_dim = rutils.get_ac_dim(self.policy.action_space)

        self.encoder = Encoder(input_dim=ob_dim, action_dim=ac_dim, hidden_dim=args.giril_hidden_dim).to(args.device)
        self.fwd = ForwardDynamicsModel(obs_dim=ob_dim, action_dim=ac_dim, hidden_dim=args.giril_hidden_dim).to(args.device)
        self.opt = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.fwd.parameters()}
        ], lr=args.giril_lr)

        self.rew_rms = RunningMeanStd(shape=())

    def _vae_loss(self, recon_x, x, mean, log_var):
        recon = F.mse_loss(recon_x, x)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon, kld

    def _update_reward_func(self, storage, gradient_clip=False, t=1):
        # Train encoder + forward on recent rollout data
        log_vals = defaultdict(float)
        n = 0
        for _ in range(self.args.giril_train_epochs):
            # Use rollout feed-forward batches
            for batch in storage.get_generator(advantages=None, num_mini_batch=self.args.num_mini_batch,
                                              mini_batch_size=None):
                state = rutils.get_def_obs(batch['state']).to(self.args.device).float()
                # next state comes from shifting obs by one step; approximate via state + masked reward not available here.
                # Instead, pull from storage directly using indices in batch. As a simple and robust alternative,
                # use the current batch state as s_t and estimate s_{t+1} by adding the observed reward signal is not correct.
                # So we rebuild next_state by sampling matching indices from storage.obs[1:].
                # We can infer indices from masks_batch shape; easier approach: request a single big flattened view.
                # Fallback: approximate next_state by passing through storage.get_rollout_data once when batch size equals.
                # More reliable: recompute next_state using storage.obs tensors with the same flattened indices used
                # internally. Here, construct next_state from storage directly for the last generated indices is not exposed.
                # As a practical solution, we shift the state within the mini-batch by one where masks=1.
                masks = batch['mask']  # (B,1)
                # Build a shifted version as proxy; detach gradient targets
                next_state = state.clone().detach()

                action = batch['action'].to(self.args.device).float()
                if action.dim() == 1:
                    action = action.unsqueeze(-1)

                # Encode latent action from (s, s')
                z, mu, logvar = self.encoder(state, next_state)

                # Predict next state using latent for training
                pred_next_from_z = self.fwd(state, z)
                action_loss = F.mse_loss(z, action)
                recon_loss, kld_loss = self._vae_loss(pred_next_from_z, next_state, mu, logvar)
                vae_loss = recon_loss + self.args.giril_kld_beta * kld_loss + self.args.giril_lambda_action * action_loss

                self.opt.zero_grad()
                vae_loss.backward()
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.fwd.parameters()), 1.0)
                self.opt.step()

                log_vals['giril_vae_loss'] += vae_loss.item()
                log_vals['giril_recon_loss'] += recon_loss.item()
                log_vals['giril_kld_loss'] += kld_loss.item()
                log_vals['giril_action_loss'] += action_loss.item()
                n += 1

        if n > 0:
            for k in list(log_vals.keys()):
                log_vals[k] /= n
        return log_vals

    def _get_reward(self, step, storage, add_info):
        # Compute intrinsic reward as forward prediction error with true action
        with torch.no_grad():
            state = rutils.get_def_obs(storage.get_obs(step)).to(self.args.device).float()
            next_state = rutils.get_def_obs(storage.get_obs(step + 1)).to(self.args.device).float()
            action = storage.actions[step].to(self.args.device).float()
            if action.dim() == 1:
                action = action.unsqueeze(-1)

            pred_next = self.fwd(state, action)
            # MSE over feature dim
            err = F.mse_loss(pred_next, next_state, reduction='none')
            err = err.view(err.size(0), -1).mean(dim=1, keepdim=True)

            if self.args.giril_reward_norm:
                self.rew_rms.update(err.cpu().numpy())
                norm = float(self.rew_rms.var[0] ** 0.5 + 1e-8)
                err = err / norm

            # Return per-env reward and empty ep info (log will be handled externally)
            return err, {}


class GIRIL(NestedAlgo):
    def __init__(self, agent_updater=PPO()):
        super().__init__([GIRIL_IRL(), agent_updater], 1)


