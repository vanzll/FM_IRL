import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from rlf.algos.on_policy.on_policy_base import OnPolicy
from tqdm import tqdm

class PPO(OnPolicy):
    def update(self, rollouts, args=None, beginning=False, t=1):
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()

        use_clipped_value_loss = True

        log_vals = defaultdict(lambda: 0)

        for e in range(self._arg('num_epochs')):
            data_generator = rollouts.get_generator(advantages,
                    self._arg('num_mini_batch'))

            for sample in data_generator:
                # Get all the data from our batch sample
                ac_eval = self.policy.evaluate_actions(sample['state'],
                        sample['other_state'],
                        sample['hxs'], sample['mask'],
                        sample['action'])

                ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
                surr1 = ratio * sample['adv']
                surr2 = torch.clamp(ratio,
                        1.0 - self._arg('clip_param'),
                        1.0 + self._arg('clip_param')) * sample['adv']
                actor_loss = -torch.min(surr1, surr2).mean(0)

                if use_clipped_value_loss:
                    value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
                                    -self._arg('clip_param'),
                                    self._arg('clip_param'))
                    value_losses = (ac_eval['value'] - sample['return']).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - sample['return']).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

                loss = (value_loss * self._arg('value_loss_coef') + actor_loss -
                     ac_eval['ent'].mean() * self._arg('entropy_coef'))
                
                # TODO: Add action loss

                self._standard_step(loss)

                log_vals['value_loss'] += value_loss.sum().item()
                log_vals['actor_loss'] += actor_loss.sum().item()
                log_vals['dist_entropy'] += ac_eval['ent'].mean().item()
                log_vals["policy_update_data"] += self._arg('num_mini_batch')

        num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
        for k in log_vals:
            log_vals[k] /= num_updates

        log_vals["policy_update_data"] *= num_updates
        return log_vals

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(f"--{self.arg_prefix}clip-param",
            type=float,
            default=0.2,
            help='ppo clip parameter')

        parser.add_argument(f"--{self.arg_prefix}entropy-coef",
            type=float,
            default=0.01,
            help='entropy term coefficient (old default: 0.01)')

        parser.add_argument(f"--{self.arg_prefix}value-loss-coef",
            type=float,
            default=0.5,
            help='value loss coefficient')


class RegIRL_PPO(PPO):
    def __init__(self):
        super().__init__()
        self._regirl_discrim = None

    def set_regirl_discriminator(self, discrim_net):
        # Expect FMLabelCondDiscriminator that exposes generate_sa
        self._regirl_discrim = discrim_net

    def _compute_regirl_loss(self):
        coef = getattr(self.args, 'regirl_coef', 0.0)
        if coef <= 0.0 or self._regirl_discrim is None:
            return None
        num_gen = getattr(self.args, 'regirl_num_gen', 256)
        num_steps = getattr(self.args, 'regirl_num_steps', max(1, getattr(self.args, 'fm_num_steps', 100)//10))

        with torch.no_grad():
            sa = self._regirl_discrim.generate_sa(num_gen, num_steps=num_steps)
        if sa is None:
            return None
        s_feat, a_expert = sa

        # Feed generated state features directly to actor head
        actor = getattr(self.policy, 'actor', None)
        dist_mod = getattr(self.policy, 'dist', None)
        if actor is None or dist_mod is None:
            return None

        actor_features, _ = actor(s_feat, None, None)
        dist = dist_mod(actor_features)
        # Use mode for stability (Gaussian -> mean)
        try:
            #print("Valid Reg Gradient.")
            #print('coef ', coef)
            a_pi = dist.mode()
        except Exception:
            #print("Invalid gradient!!!!!!!!!")
            a_pi = dist.sample()

        reg_loss = 0.5 * (a_pi - a_expert).pow(2).mean()
        return coef * reg_loss

    def update(self, rollouts, args=None, beginning=False, t=1):
        # Keep standard PPO losses
        self._compute_returns(rollouts)
        advantages = rollouts.compute_advantages()

        use_clipped_value_loss = True

        log_vals = defaultdict(lambda: 0)

        for e in range(self._arg('num_epochs')):
            data_generator = rollouts.get_generator(advantages,
                    self._arg('num_mini_batch'))

            for sample in data_generator:
                ac_eval = self.policy.evaluate_actions(sample['state'],
                        sample['other_state'],
                        sample['hxs'], sample['mask'],
                        sample['action'])

                ratio = torch.exp(ac_eval['log_prob'] - sample['prev_log_prob'])
                surr1 = ratio * sample['adv']
                surr2 = torch.clamp(ratio,
                        1.0 - self._arg('clip_param'),
                        1.0 + self._arg('clip_param')) * sample['adv']
                actor_loss = -torch.min(surr1, surr2).mean(0)

                if use_clipped_value_loss:
                    value_pred_clipped = sample['value'] + (ac_eval['value'] - sample['value']).clamp(
                                    -self._arg('clip_param'),
                                    self._arg('clip_param'))
                    value_losses = (ac_eval['value'] - sample['return']).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - sample['return']).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (sample['return'] - ac_eval['value']).pow(2).mean()

                loss = (value_loss * self._arg('value_loss_coef') + actor_loss -
                     ac_eval['ent'].mean() * self._arg('entropy_coef'))

                # Add RegIRL regularization if enabled
                
                reg_loss = self._compute_regirl_loss()
                if reg_loss is not None:
                    loss = loss + reg_loss
                    log_vals['regirl_loss'] += float(reg_loss)
                #print('actor_loss ', actor_loss)
                #print('reg_loss ', reg_loss)
                self._standard_step(loss)
                """ for _ in tqdm(range(10)):
                    reg_loss = self._compute_regirl_loss()
                    if reg_loss is not None:
                        self._standard_step(reg_loss) # only preserve one """
                
                
                log_vals['value_loss'] += value_loss.sum().item()
                log_vals['actor_loss'] += actor_loss.sum().item()
                log_vals['dist_entropy'] += ac_eval['ent'].mean().item()
                log_vals["policy_update_data"] += self._arg('num_mini_batch')

        num_updates = self._arg('num_epochs') * self._arg('num_mini_batch')
        for k in log_vals:
            log_vals[k] /= num_updates

        log_vals["policy_update_data"] *= num_updates
        return log_vals

    def get_add_args(self, parser):
        super().get_add_args(parser)
        parser.add_argument(f"--{self.arg_prefix}regirl-coef", type=float, default=0.1,
            help='Coefficient for RegIRL action MSE regularization')
        parser.add_argument(f"--{self.arg_prefix}regirl-num-gen", type=int, default=256,
            help='Number of generated expert (s,a) pairs per minibatch for RegIRL')
        parser.add_argument(f"--{self.arg_prefix}regirl-num-steps", type=int, default=10,
            help='Number of ODE steps for FM generation when computing RegIRL')
        parser.add_argument(f"--{self.arg_prefix}regirl-bc-coef", type=float, default=0.0,
            help='Coefficient for FM-BC distillation regularization (0.0 to disable)')
        parser.add_argument(f"--{self.arg_prefix}regirl-bc-steps", type=int, default=0,
            help='Number of optimizer steps per minibatch for FM-BC distillation')
