import torch
import torch.nn as nn
import torch.nn.functional as F

from rlf.algos.on_policy.on_policy_base import OnPolicy
from rlf.policies.actor_critic.base_actor_critic import ActorCritic
from rlf.policies.base_policy import ActionData
import rlf.policies.utils as putils
import rlf.rl.utils as rutils
from rlf.rl.model import TwoLayerMlpWithAction, MLPBasic

from drail.flow_matching import FlowMatchingModel


class FMA2CPolicy(ActorCritic):
    """
    Actor-Critic policy with a Flow-Matching actor (reparameterizable) and
    both V(s) critic (for returns) and Q(s,a) critic (for actor pathwise loss).
    No action log-probabilities are used.
    """

    def __init__(self,
            get_actor_fn=None,
            get_critic_fn=None,
            get_critic_head_fn=None,
            use_goal=False,
            fuse_states=[],
            get_base_net_fn=None,
            fm_steps: int = 40,
            fm_hidden: int = 256,
            fm_depth: int = 2):
        super().__init__(get_critic_fn, get_critic_head_fn, use_goal, fuse_states, get_base_net_fn)
        if get_actor_fn is None:
            get_actor_fn = putils.get_def_actor
        self.get_actor_fn = get_actor_fn
        self.fm_steps = fm_steps
        self.fm_hidden = fm_hidden
        self.fm_depth = fm_depth

    def init(self, obs_space, action_space, args):
        super().init(obs_space, action_space, args)

        self.actor = self.get_actor_fn(
            rutils.get_obs_shape(obs_space, args.policy_ob_key), self._get_base_out_shape())

        # Q-critic over (state features, action) with head to scalar
        self.q_critic = TwoLayerMlpWithAction(
            self._get_base_out_shape()[0], (self.fm_hidden, self.fm_hidden), action_space.shape[0])
        self.q_head = nn.Linear(self.q_critic.output_shape[0], 1)

        # FM actor flow over action space, conditioned on actor features
        self.flow_model = FlowMatchingModel(
            cond_dim=self.actor.output_shape[0],
            data_dim=action_space.shape[0],
            num_units=self.fm_hidden,
            depth=self.fm_depth,
            device=str(self.args.device) if hasattr(self.args, 'device') else 'cuda')

        self.ac_low_bound = torch.tensor(self.action_space.low).to(self.args.device)
        self.ac_high_bound = torch.tensor(self.action_space.high).to(self.args.device)

    def _actor_features(self, state, add_state, hxs, masks):
        base_features, hxs = self._apply_base_net(state, add_state, hxs, masks)
        base_features = self._fuse_base_out(base_features, add_state)
        actor_features, _ = self.actor(base_features, hxs, masks)
        return actor_features, base_features, hxs

    def _integrate_flow(self, cond: torch.Tensor, steps: int, noise: torch.Tensor = None, requires_grad: bool = True):
        B = cond.size(0)
        A = self.action_space.shape[0]
        if noise is None:
            a = torch.randn(B, A, device=cond.device)
        else:
            a = noise
        dt = 1.0 / max(1, steps)
        t_val = 0.0
        if requires_grad:
            for _ in range(steps):
                t = torch.full((B,), t_val, device=cond.device)
                v = self.flow_model(a, cond, t)
                a = a + v * dt
                t_val += dt
        else:
            with torch.no_grad():
                for _ in range(steps):
                    t = torch.full((B,), t_val, device=cond.device)
                    v = self.flow_model(a, cond, t)
                    a = a + v * dt
                    t_val += dt
        return a

    def get_action(self, state, add_state, hxs, masks, step_info):
        actor_features, base_features, hxs = self._actor_features(state, add_state, hxs, masks)
        action = self._integrate_flow(actor_features, self.fm_steps, requires_grad=False)
        action = rutils.multi_dim_clip(action, self.ac_low_bound, self.ac_high_bound)
        # State-value for storage/returns
        value = self._get_value_from_features(base_features, hxs, masks)
        return ActionData(value, action, torch.zeros(action.size(0), 1, device=action.device), hxs, {'dist_entropy': torch.tensor(0.0, device=action.device)})

    # Helpers for update
    def sample_action_with_grad(self, state, add_state, hxs, masks):
        actor_features, base_features, _ = self._actor_features(state, add_state, hxs, masks)
        action = self._integrate_flow(actor_features, self.fm_steps, requires_grad=True)
        action = rutils.multi_dim_clip(action, self.ac_low_bound, self.ac_high_bound)
        return action, base_features

    def Q(self, base_features, action):
        q_feat, _ = self.q_critic(base_features, action, None, None)
        return self.q_head(q_feat)


class FMA2C(OnPolicy):
    """
    On-policy actor-critic with FM reparameterized actor.
    - Actor loss: -E[ Q(s, a_θ(s, ξ)) ]
    - Critic-V loss: MSE(V(s), return)
    - Critic-Q loss: MSE(Q(s, a_t), return)
    """

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument('--a2c-value-loss-coef', type=float, default=0.5)
        parser.add_argument('--a2c-q-loss-coef', type=float, default=1.0)
        parser.add_argument('--fm-a2c-steps', type=int, default=40)
        parser.add_argument('--fm-a2c-hidden', type=int, default=256)
        parser.add_argument('--fm-a2c-depth', type=int, default=2)

    def update(self, rollouts, args=None, beginning=False, t=1):
        self._compute_returns(rollouts)

        total_actor, total_v, total_q = 0.0, 0.0, 0.0

        data_generator = rollouts.get_generator(num_mini_batch=self._arg('num_mini_batch'))
        for sample in data_generator:
            state = sample['state']
            other_state = sample['other_state']
            hxs = sample['hxs']
            masks = sample['mask']

            # Values for V-loss
            V_pred = self.policy.get_value(state, other_state, hxs, masks)
            value_loss = 0.5 * (V_pred - sample['return']).pow(2).mean()

            # Actor: pathwise via Q(s, aθ)
            a_theta, base_features = self.policy.sample_action_with_grad(state, other_state, hxs, masks)
            Q_pred_actor = self.policy.Q(base_features, a_theta)
            actor_loss = -Q_pred_actor.mean()

            # Q-critic: supervised to Monte-Carlo return using rollout action
            Q_onbatch = self.policy.Q(base_features.detach(), sample['action'])
            q_loss = 0.5 * (Q_onbatch - sample['return']).pow(2).mean()

            loss = actor_loss + self._arg('a2c_value_loss_coef') * value_loss + self._arg('a2c_q_loss_coef') * q_loss

            self._standard_step(loss)

            total_actor += actor_loss.item()
            total_v += value_loss.item()
            total_q += q_loss.item()

        n_batches = max(1, self._arg('num_mini_batch'))
        return {
            'actor_loss': total_actor / n_batches,
            'value_loss': total_v / n_batches,
            'q_loss': total_q / n_batches,
        }


