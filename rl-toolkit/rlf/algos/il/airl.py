# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# See repo license header.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import defaultdict

import rlf.rl.utils as rutils
import rlf.algos.utils as autils
from rlf.baselines.common.running_mean_std import RunningMeanStd
from rlf.algos.il.base_irl import BaseIRLAlgo
from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO


class AIRLDiscrim(BaseIRLAlgo):
    """
    AIRL-style discriminator with entropy regularization in the logits, and
    reward computed from discriminator output.
    Trains by zipping expert loader with agent rollout batches.
    """

    def __init__(self, get_discrim=None):
        super().__init__()
        self.get_discrim = get_discrim

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--airl-ent-coef', type=float, default=0.01)
        parser.add_argument('--airl-disc-lr', type=float, default=3e-4)
        parser.add_argument('--airl-grad-pen', type=float, default=10.0)
        parser.add_argument('--airl-epochs', type=int, default=1)
        parser.add_argument('--airl-reward-norm', action='store_true', default=False)
        parser.add_argument('--reward-type', type=str, default='airl', help='[airl, gail, raw]')

    def init(self, policy, args):
        super().init(policy, args)
        self.action_space = self.policy.action_space

        # Build discriminator similar to GAIL
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        ac_dim = rutils.get_ac_dim(self.action_space)
        base_net = self.policy.get_base_net_fn(ob_shape)
        if self.get_discrim is None:
            # Default small MLP head
            hidden_dim = getattr(self.args, 'discrim_num_unit', 64)
            depth = getattr(self.args, 'discrim_depth', 4)
            layers = [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            for _ in range(max(0, depth - 1)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            layers += [nn.Linear(hidden_dim, 1)]
            discrim = nn.Sequential(*layers)
            dhidden_dim = hidden_dim
        else:
            discrim, dhidden_dim = self.get_discrim()

        from rlf.rl.model import InjectNet
        self.disc_net = InjectNet(
            base_net.net,
            discrim,
            base_net.output_shape[0], dhidden_dim, ac_dim,
            getattr(self.args, 'action_input', True)
        ).to(self.args.device)

        self.opt = optim.Adam(self.disc_net.parameters(), lr=self.args.airl_disc_lr)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def _get_sampler(self, storage):
        agent_experience = storage.get_generator(None, mini_batch_size=self.expert_train_loader.batch_size)
        return self.expert_train_loader, agent_experience

    def _norm_expert_state(self, state, obsfilt):
        if not getattr(self.args, 'gail_state_norm', True):
            return torch.as_tensor(state).to(self.args.device)
        state = state.cpu().numpy() if isinstance(state, torch.Tensor) else state
        if obsfilt is not None:
            state = obsfilt(state, update=False)
        return torch.tensor(state).to(self.args.device)

    def _trans_agent_state(self, state, other_state=None):
        if not getattr(self.args, 'gail_state_norm', True):
            if other_state is None:
                return state['raw_obs']
            return other_state['raw_obs']
        return rutils.get_def_obs(state)

    def _compute_disc_val(self, state, action):
        return self.disc_net(state, action)

    def _compute_loss(self, expert_batch, agent_batch, obsfilt):
        # Prepare expert
        expert_actions = expert_batch['actions'].to(self.args.device)
        expert_actions = rutils.get_ac_repr(self.action_space, expert_actions)
        expert_states = self._norm_expert_state(expert_batch['state'], obsfilt)

        # Prepare agent
        agent_states = self._trans_agent_state(agent_batch['state'], agent_batch.get('other_state'))
        agent_actions = agent_batch['action']
        agent_actions = rutils.get_ac_repr(self.action_space, agent_actions)

        # Discriminator logits (second column)
        agent_d = self._compute_disc_val(agent_states, agent_actions)
        expert_d = self._compute_disc_val(expert_states, expert_actions)

        # Policy log-probs for AIRL entropy regularization term (first column)
        # Use evaluate_actions to get log_prob; supply add_state/hxs/masks from batches
        def get_logp(states, actions, batch, is_expert: bool):
            add_state = batch.get('other_state') if not is_expert else None
            hxs = batch.get('hxs') if not is_expert else None
            masks = batch.get('mask') if not is_expert else None
            eval_out = self.policy.evaluate_actions(states, add_state, hxs, masks, actions)
            logp = eval_out['log_prob']
            # Ensure 2D (B,1)
            if logp.dim() == 1:
                logp = logp.view(-1, 1)
            elif logp.dim() > 2:
                logp = logp.view(logp.size(0), -1)
                logp = logp[:, :1]
            return logp

        logp_agent = get_logp(agent_states, agent_actions, agent_batch, is_expert=False)
        # For expert, we do not have RNN states; pass None
        logp_expert = self.policy.evaluate_actions(expert_states, None, None, None, expert_actions)['log_prob']
        if logp_expert.dim() == 1:
            logp_expert = logp_expert.view(-1, 1)
        elif logp_expert.dim() > 2:
            logp_expert = logp_expert.view(logp_expert.size(0), -1)[:, :1]

        ent_c = self.args.airl_ent_coef
        # Build 2-class logits: [ent_coef * logpi, disc]
        logits_expert = torch.cat([ent_c * logp_expert.detach(), expert_d], dim=1)
        logits_agent = torch.cat([ent_c * logp_agent.detach(), agent_d], dim=1)

        labels_expert = torch.ones(logits_expert.size(0), dtype=torch.long, device=self.args.device)
        labels_agent = torch.zeros(logits_agent.size(0), dtype=torch.long, device=self.args.device)

        expert_loss = F.cross_entropy(logits_expert, labels_expert)
        agent_loss = F.cross_entropy(logits_agent, labels_agent)

        # Gradient penalty (WGAN-GP style) using util for stability
        grad_pen = self.args.airl_grad_pen * autils.wass_grad_pen(
            expert_states, expert_actions, agent_states, agent_actions,
            getattr(self.args, 'action_input', True), self._compute_disc_val)

        total = expert_loss + agent_loss + grad_pen
        return total, expert_loss, agent_loss, grad_pen

    def _update_reward_func(self, storage, *args, **kwargs):
        self.disc_net.train()
        log_vals = defaultdict(lambda: 0.0)
        obsfilt = self.get_env_ob_filt()

        n = 0
        expert_sampler, agent_sampler = self._get_sampler(storage)
        if agent_sampler is None:
            return {}

        for _ in range(self.args.airl_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                total, e_loss, a_loss, gp = self._compute_loss(expert_batch, agent_batch, obsfilt)
                self.opt.zero_grad()
                total.backward()
                self.opt.step()
                n += 1
                log_vals['airl_total'] += float(total.item())
                log_vals['airl_expert'] += float(e_loss.item())
                log_vals['airl_agent'] += float(a_loss.item())
                log_vals['airl_gp'] += float(gp.item())

        if n > 0:
            for k in list(log_vals.keys()):
                log_vals[k] /= n
        return log_vals

    def _compute_reward_from_d(self, d_val):
        s = torch.sigmoid(d_val)
        eps = 1e-20
        if self.args.reward_type == 'airl':
            return (s + eps).log() - (1 - s + eps).log()
        elif self.args.reward_type == 'gail':
            return (s + eps).log()
        elif self.args.reward_type == 'raw':
            return d_val
        else:
            return (s + eps).log() - (1 - s + eps).log()

    def _get_reward(self, step, storage, add_info):
        masks = storage.masks[step]
        with torch.no_grad():
            self.disc_net.eval()
            state = self._trans_agent_state(storage.get_obs(step))
            action = storage.actions[step]
            action = rutils.get_ac_repr(self.action_space, action)
            d_val = self._compute_disc_val(state, action)
            reward = self._compute_reward_from_d(d_val)

            if self.args.airl_reward_norm:
                if self.returns is None:
                    self.returns = reward.clone()
                self.returns = self.returns * masks * self.args.gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())
                reward = reward / np.sqrt(self.ret_rms.var[0] + 1e-8)

            return reward, {}


class AIRL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([AIRLDiscrim(get_discrim), agent_updater], 1)


