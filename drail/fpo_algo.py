import math
from collections import defaultdict

import torch
import torch.nn as nn

from rlf.algos.on_policy.on_policy_base import OnPolicy
import rlf.rl.utils as rutils


class FPOAlgo(OnPolicy):
    """
    FPO-style on-policy algorithm using a Flow-Matching policy.
    Mirrors the core idea: construct a clipped surrogate objective with ratio
    built from contrastive flow-matching losses between old and current policy.
    """

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--fpo-flow-steps', type=int, default=10)
        parser.add_argument('--fpo-n-samples-per-action', type=int, default=8)
        parser.add_argument('--fpo-average-before-exp', type=bool, default=True)
        parser.add_argument('--fpo-discretize-t', type=bool, default=False)
        parser.add_argument('--fpo-clipping-epsilon', type=float, default=0.05)
        parser.add_argument('--fpo-value-loss-coef', type=float, default=0.25)
        parser.add_argument('--fpo-loss-mode', type=str, default='u_but_supervise_as_eps')
        parser.add_argument('--fpo-noise-sigma', type=float, default=0.0)

    def pre_update(self, cur_update):
        super().pre_update(cur_update)
        # Snapshot behavior policy used during rollout
        self._old_policy = self._copy_policy()

    def _get_actor_features(self, policy, state, add_state, hxs, masks):
        # Replicate DistActorCritic.forward pipeline to extract actor features
        base_features, _ = policy._apply_base_net(state, add_state, hxs, masks)
        base_features = policy._fuse_base_out(base_features, add_state)
        actor_features, _ = policy.actor(base_features, hxs, masks)
        return actor_features, base_features

    def _get_flow_model(self, policy):
        # Expect FMActionDistHead with attribute flow_model
        return getattr(policy.dist, 'flow_model', None)

    def _sample_t(self, batch_size, device, flow_steps, discretize):
        if discretize:
            # Sample indices in [0, flow_steps)
            idx = torch.randint(low=0, high=max(1, flow_steps), size=(batch_size, 1), device=device)
            # Map to [0,1] grid points
            if flow_steps <= 1:
                t = torch.zeros(batch_size, 1, device=device)
            else:
                t = (flow_steps - 1 - idx).float() / float(flow_steps - 1)
        else:
            t = torch.rand(batch_size, 1, device=device)
        return t

    def _compute_cfm_loss(self, flow_model, cond, action, n_samples, flow_steps, discretize_t, loss_mode, noise_sigma):
        """
        cond: [B, C] actor features; action: [B, A]
        Returns: [B, S] losses (one per sample) or [B, 1] if reduced.
        """
        B, A = action.shape
        device = action.device
        eps = torch.randn(B, n_samples, A, device=device)
        t = self._sample_t(B * n_samples, device, flow_steps, discretize_t).view(B, n_samples, 1)

        # x_t = t * eps + (1 - t) * action
        x_t = t * eps + (1.0 - t) * action.unsqueeze(1)

        # Broadcast cond: [B, C] -> [B, n_samples, C]
        cond_rep = cond.unsqueeze(1).expand(B, n_samples, cond.shape[-1]).contiguous().view(B * n_samples, -1)
        x_t_flat = x_t.view(B * n_samples, A)
        t_flat = t.view(B * n_samples)

        # Predict velocity
        v_pred = flow_model(x_t_flat, cond_rep, t_flat)
        v_pred = v_pred.view(B, n_samples, A)

        if loss_mode == 'u':
            velocity_gt = eps - action.unsqueeze(1)
            loss = (v_pred - velocity_gt).pow(2).mean(dim=-1)  # [B, S]
        else:
            # 'u_but_supervise_as_eps'
            x0_pred = x_t - t * v_pred
            x1_pred = x0_pred + v_pred
            loss = (eps - x1_pred).pow(2).mean(dim=-1)  # [B, S]

        return loss  # [B, S]

    def update(self, rollouts, args=None, beginning=False, t=1):
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()

        log_vals = defaultdict(lambda: 0)

        for _ in range(self._arg('num_epochs')):
            data_generator = rollouts.get_generator(advantages, self._arg('num_mini_batch'))
            for sample in data_generator:
                state = sample['state']
                other_state = sample['other_state']
                hxs = sample['hxs']
                masks = sample['mask']
                action = sample['action']
                adv = sample['adv']
                ret = sample['return']

                # Values (current policy)
                ac_eval = self.policy.evaluate_actions(state, other_state, hxs, masks, action)
                value = ac_eval['value']

                # Extract actor features and flow models for current and old
                cur_actor_features, cur_base = self._get_actor_features(self.policy, state, other_state, hxs, masks)
                old_actor_features, _ = self._get_actor_features(self._old_policy, state, other_state, hxs, masks)
                cur_flow = self._get_flow_model(self.policy)
                old_flow = self._get_flow_model(self._old_policy)
                if (cur_flow is None) or (old_flow is None):
                    raise RuntimeError('FPOAlgo expects FMActionDistHead as policy distribution with flow_model attribute.')

                # Compute CFM losses
                S = int(self._arg('fpo_n_samples_per_action'))
                flow_steps = int(self._arg('fpo_flow_steps'))
                discretize_t = bool(self._arg('fpo_discretize_t'))
                loss_mode = self._arg('fpo_loss_mode')
                noise_sigma = float(self._arg('fpo_noise_sigma'))

                with torch.no_grad():
                    old_loss = self._compute_cfm_loss(old_flow, old_actor_features, action, S, flow_steps, discretize_t, loss_mode, noise_sigma)
                cur_loss = self._compute_cfm_loss(cur_flow, cur_actor_features, action, S, flow_steps, discretize_t, loss_mode, noise_sigma)

                # Ratio rho_s
                if bool(self._arg('fpo_average_before_exp')):
                    # [B,1]
                    init_mean = old_loss.mean(dim=-1, keepdim=True)
                    cur_mean = cur_loss.mean(dim=-1, keepdim=True)
                    rho = torch.exp(init_mean - cur_mean)
                else:
                    # Clip per-sample difference then exp and average
                    delta = (old_loss - cur_loss).clamp(-3.0, 3.0)
                    rho = torch.exp(delta).mean(dim=-1, keepdim=True)

                # Clipped surrogate
                eps = float(self._arg('fpo_clipping_epsilon'))
                surr1 = rho * adv
                surr2 = torch.clamp(rho, 1.0 - eps, 1.0 + eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean(0)

                # Value loss
                value_loss = 0.5 * (value - ret).pow(2).mean()

                loss = actor_loss + float(self._arg('fpo_value_loss_coef')) * value_loss
                self._standard_step(loss)

                log_vals['actor_loss'] += actor_loss.item()
                log_vals['value_loss'] += value_loss.item()

        return log_vals


