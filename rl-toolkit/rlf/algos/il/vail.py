# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rlf.algos.nested_algo import NestedAlgo
from rlf.algos.on_policy.ppo import PPO
from rlf.algos.il.gail import GailDiscrim
from rlf.rl.model import InjectNet
import rlf.rl.utils as rutils


class _VAILHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Map InjectNet hidden to VAIL latent stats
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh_2 = nn.Tanh()
        # Output uses half of hidden (means) -> 1 logit
        self.output = nn.Linear(int(hidden_dim // 2), 1)
        self.sigmoid_output = nn.Sigmoid()

    def forward(self, x, mean_mode: bool = True):
        x = self.linear_2(x)
        x = self.tanh_2(x)

        parameters = x
        halfpoint = parameters.shape[-1] // 2
        mus, sigmas = parameters[..., :halfpoint], parameters[..., halfpoint:]
        sigmas = self.sigmoid_output(sigmas)

        if mean_mode:
            z = mus
        else:
            z = mus + torch.randn_like(mus) * sigmas

        logits = self.output(z)
        return logits, mus, sigmas


class VAIL(NestedAlgo):
    def __init__(self, agent_updater=PPO(), get_discrim=None):
        super().__init__([VailDiscrim(get_discrim), agent_updater], 1)


class VailDiscrim(GailDiscrim):
    """
    Variational Adversarial Imitation Learning discriminator and loss.
    Mirrors GAIL pipeline with InjectNet while adding a KL regularizer.
    """
    def __init__(self, get_discrim=None):
        super().__init__(get_discrim)

    def get_default_discrim(self):
        """
        Returns a head that expects `hidden_dim` as input (from InjectNet),
        and outputs (logits, mus, sigmas).
        """
        hidden_dim = self.args.discrim_num_unit
        return _VAILHead(hidden_dim), hidden_dim

    def _create_discrim(self):
        ob_shape = rutils.get_obs_shape(self.policy.obs_space)
        ac_dim = rutils.get_ac_dim(self.action_space)
        base_net = self.policy.get_base_net_fn(ob_shape)
        discrim, dhidden_dim = self.get_discrim()
        discrim_head = InjectNet(
            base_net.net,
            discrim,
            base_net.output_shape[0], dhidden_dim, ac_dim,
            self.args.action_input)
        return discrim_head.to(self.args.device)

    def init(self, policy, args):
        super().init(policy, args)
        self.action_space = self.policy.action_space
        self.discrim_net = self._create_discrim()
        self.opt = optim.Adam(self.discrim_net.parameters(), lr=self.args.disc_lr)

    def _disc_forward(self, states, actions):
        # InjectNet returns whatever the head returns; in VAIL it's (logits, mu, sigma)
        out = self.discrim_net(states, actions)
        # Backward compatibility if someone swaps head
        if isinstance(out, tuple) and len(out) == 3:
            return out
        else:
            # Fallback: synthesize zero mus/sigmas for KL term
            logits = out
            hidden_dim = self.args.discrim_num_unit
            half = hidden_dim // 2
            device = logits.device
            mus = torch.zeros(logits.shape[0], half, device=device)
            sigmas = torch.zeros_like(mus)
            return logits, mus, sigmas

    def _compute_disc_val(self, state, action):
        out = self.discrim_net(state, action)
        if isinstance(out, tuple):
            return out[0]
        return out

    def _kl_to_standard_normal(self, mus, sigmas, eps: float = 1e-8):
        # KL(q(z|x) || N(0,1)) with q ~ N(mu, sigma^2)
        var = sigmas * sigmas + eps
        logvar = torch.log(var + eps)
        kl = 0.5 * (mus * mus + var - logvar - 1.0)
        # Mean over latent dims, then mean over batch
        return kl.mean()

    def _compute_discrim_loss(self, agent_batch, expert_batch, obsfilt):
        expert_actions = expert_batch['actions'].to(self.args.device)
        expert_actions = self._adjust_action(expert_actions)
        expert_states = self._norm_expert_state(expert_batch['state'], obsfilt)

        agent_states = self._trans_agent_state(
            agent_batch['state'],
            agent_batch['other_state'] if 'other_state' in agent_batch else None
        )
        agent_actions = agent_batch['action']
        agent_actions = rutils.get_ac_repr(self.action_space, agent_actions)
        expert_actions = rutils.get_ac_repr(self.action_space, expert_actions)

        expert_logits, expert_mus, expert_sigmas = self._disc_forward(expert_states, expert_actions)
        agent_logits, agent_mus, agent_sigmas = self._disc_forward(agent_states, agent_actions)

        grad_pen = self.compute_pen(expert_states, expert_actions, agent_states, agent_actions)

        # BCE parts (same as GAIL)
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones(expert_logits.shape, device=self.args.device)
        )
        agent_loss = F.binary_cross_entropy_with_logits(
            agent_logits, torch.zeros(agent_logits.shape, device=self.args.device)
        )

        # KL regularizer
        kl_expert = self._kl_to_standard_normal(expert_mus, expert_sigmas)
        kl_agent = self._kl_to_standard_normal(agent_mus, agent_sigmas)
        kl = 0.5 * (kl_expert + kl_agent)
        kl_weight = getattr(self.args, 'vail_kl_coef', 1e-3)

        discrim_loss = expert_loss + agent_loss + kl_weight * kl

        return expert_logits, agent_logits, grad_pen, {
            'kl': kl.detach(),
            'kl_expert': kl_expert.detach(),
            'kl_agent': kl_agent.detach(),
        }

    def _update_reward_func(self, storage):
        self.discrim_net.train()
        obsfilt = self.get_env_ob_filt()

        expert_sampler, agent_sampler = self._get_sampler(storage)
        if agent_sampler is None:
            return {}

        n = 0
        log_vals = {}
        for _ in range(self.args.n_gail_epochs):
            for expert_batch, agent_batch in zip(expert_sampler, agent_sampler):
                expert_batch, agent_batch = self._trans_batches(expert_batch, agent_batch)
                n += 1

                expert_d, agent_d, grad_pen, add_stats = self._compute_discrim_loss(
                    agent_batch, expert_batch, obsfilt
                )
                expert_loss = self._compute_expert_loss(expert_d, expert_batch)
                agent_loss = self._compute_agent_loss(agent_d, agent_batch)
                discrim_loss = expert_loss + agent_loss

                if self.args.disc_grad_pen != 0.0:
                    total_loss = discrim_loss + grad_pen
                else:
                    total_loss = discrim_loss

                # Add KL losses from the add_stats by recomputing with weight to be consistent
                total_loss = total_loss + getattr(self.args, 'vail_kl_coef', 1e-3) * add_stats['kl']

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

                # Logging
                for k, v in {
                    'discrim_loss': discrim_loss.item(),
                    'expert_loss': expert_loss.item(),
                    'agent_loss': agent_loss.item(),
                    'kl': add_stats['kl'].item(),
                    'kl_expert': add_stats['kl_expert'].item(),
                    'kl_agent': add_stats['kl_agent'].item(),
                }.items():
                    log_vals[k] = log_vals.get(k, 0.0) + v

        for k in list(log_vals.keys()):
            log_vals[k] /= max(n, 1)

        return log_vals

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--vail-kl-coef', type=float, default=1e-3)

