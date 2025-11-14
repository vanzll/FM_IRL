# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from functools import partial
import torch.nn as nn
import fmirl.ddpm.policy_model as policy_model
from rlf.policies import BasicPolicy, DistActorCritic
from rlf.policies.actor_critic.dist_actor_q import (DistActorQ, get_sac_actor,
                                                    get_sac_critic)
from rlf.policies.actor_critic.reg_actor_critic import RegActorCritic
from rlf.rl.model import MLPBase, MLPBasic, TwoLayerMlpWithAction
from goal_prox.models import GwImgEncoder
from fmirl.flow_matching import MLPFlowPolicy
from fmirl.fm_ppo import FMActionDistHead
from fmirl.fm_a2c import FMA2CPolicy


def get_ppo_policy(env_name, args):
    if env_name.startswith("MiniGrid") and args.gw_img:
        return DistActorCritic(get_base_net_fn=lambda i_shape: GwImgEncoder(i_shape))

    return DistActorCritic(
        get_actor_fn=lambda _, i_shape: MLPBasic(
            i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers
        ),
        get_critic_fn=lambda _, i_shape, asp: MLPBasic(
            i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers
        ),
    )

# def get_deep_sac_policy(env_name, args):
#     return DistActorQ(
#         get_critic_fn=partial(get_sac_critic, hidden_dim=256),
#         get_actor_fn=partial(get_sac_actor, hidden_dim=256),
#     )

def get_deep_sac_policy(env_name, args):
    return DistActorQ(
        get_critic_fn=get_sac_critic,
        get_actor_fn=get_sac_actor,
    )

def get_deep_iqlearn_policy(env_name, args):
    return DistActorQ(
        get_critic_fn=get_sac_critic,
        get_actor_fn=get_sac_actor,
    )


def get_deep_ddpg_policy(env_name, args):
    def get_actor_head(hidden_dim, action_dim):
        return nn.Sequential(nn.Linear(hidden_dim, action_dim), nn.Tanh())

    return RegActorCritic(
        get_actor_fn=lambda _, i_shape: MLPBase(i_shape[0], False, (256, 256)),
        get_actor_head_fn=get_actor_head,
        get_critic_fn=lambda _, i_shape, a_space: TwoLayerMlpWithAction(
            i_shape[0], (256, 256), a_space.shape[0]
        ),
    )

def get_basic_policy(env_name, args, is_stoch):
    if env_name.startswith("MiniGrid") and args.gw_img:
        return BasicPolicy(
            is_stoch=is_stoch, get_base_net_fn=lambda i_shape: GwImgEncoder(i_shape)
        )
    else:
        return BasicPolicy(
            is_stoch=is_stoch,
            get_base_net_fn=lambda i_shape: MLPBasic(
                i_shape[0],
                hidden_size=args.hidden_dim,
                num_layers=args.depth
            ),
        )

    return BasicPolicy()

def get_diffusion_policy(env_name, args, is_stoch):    
    if env_name[:9] == 'FetchPush':
        state_dim = 16
        action_dim = 3
    if env_name[:9] == 'FetchPick':
        state_dim = 16
        action_dim = 4
    if env_name[:10] == 'CustomHand':
        state_dim = 68
        action_dim = 20
    if env_name[:4] == 'maze':
        state_dim = 6
        action_dim = 2
    if env_name[:6] == 'Walker':
        state_dim = 17
        action_dim = 6
    if env_name[:11] == 'HalfCheetah':
        state_dim = 17
        action_dim = 6
    if env_name[:7] == 'AntGoal':
        state_dim = 132
        action_dim = 8
    if 'halfcheetah' in env_name:
        state_dim = 17
        action_dim = 6
    if 'hopper' in env_name:
        state_dim = 11
        action_dim = 3
    return policy_model.MLPDiffusion(
        n_steps = 1000,
        action_dim=action_dim, 
        state_dim=state_dim,
        num_units=args.hidden_dim,
        depth=args.depth,
        is_stoch=is_stoch,
        )

def get_flow_matching_policy(env_name, args, is_stoch):
    # Minimal dimension setup consistent with existing branches
    env_lower = env_name.lower()
    if env_name.startswith('Circle'):
        state_dim, action_dim = 1, 1
    elif env_name[:9] == 'FetchPush':
        state_dim, action_dim = 16, 3
    elif env_name[:9] == 'FetchPick':
        state_dim, action_dim = 16, 4
    elif env_name[:10] == 'CustomHand':
        state_dim, action_dim = 68, 20
    elif env_name[:4] == 'maze':
        state_dim, action_dim = 6, 2
    elif env_name[:6] == 'Walker':
        state_dim, action_dim = 17, 6
    elif env_name[:11] == 'HalfCheetah':
        state_dim, action_dim = 17, 6
    elif env_name[:7] == 'AntGoal':
        state_dim, action_dim = 132, 8
    # D4RL-style env names (lowercase prefixes), e.g., hopper-medium-v0, halfcheetah-medium-v0
    elif 'hopper' in env_lower:
        # D4RL Hopper-v2/v3 obs=11, act=3
        state_dim, action_dim = 11, 3
    elif 'halfcheetah' in env_lower:
        # D4RL HalfCheetah-v2/v3 obs=17, act=6
        state_dim, action_dim = 17, 6
    else:
        # Fallback to BasicPolicy shapes if available via obs_space later
        state_dim = getattr(args, 'state_dim', 6)
        action_dim = getattr(args, 'action_dim', 2)

    # Prefer repository-integrated FM policy that uses the official FM package
    # Use built-in minimal FM model
    return MLPFlowPolicy(
        n_steps=getattr(args, 'flow_matching_steps', 100),
        action_dim=action_dim,
        state_dim=state_dim,
        num_units=args.hidden_dim,
        depth=args.depth,
        is_stoch=is_stoch
        ) # This works!
        

def get_deep_basic_policy(env_name, args):
    return BasicPolicy(
        get_base_net_fn=lambda i_shape: MLPBase(i_shape[0], False, (512, 512, 256, 128))
    )


def get_fmppo_policy(env_name, args):
    """Build DistActorCritic with a Flow-Matching action distribution."""
    # Actor uses standard MLP; critic as in PPO
    def actor_fn(_, i_shape):
        return MLPBasic(i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers)

    def critic_fn(_, i_shape, asp):
        return MLPBasic(i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers)

    def fm_dist_fn(input_shape, action_space):
        # input_shape is a tuple like (feature_dim,)
        feat_dim = input_shape[0]
        if hasattr(action_space, 'shape') and len(action_space.shape) > 0:
            action_dim = action_space.shape[0]
        else:
            raise ValueError('FM-PPO requires continuous (Box) action space')

        n_steps = getattr(args, 'fmppo_steps', 40)
        hidden = getattr(args, 'fmppo_hidden', args.ppo_hidden_dim)
        depth = getattr(args, 'fmppo_depth', max(2, args.ppo_layers))
        hutch = getattr(args, 'fmppo_hutchinson', 1)
        device = getattr(args, 'device', 'cuda')
        sanitize = getattr(args, 'alg', '') == 'fpo'
        low = None
        high = None
        try:
            import torch
            low = torch.tensor(action_space.low)
            high = torch.tensor(action_space.high)
        except Exception:
            pass

        return FMActionDistHead(
            actor_feature_dim=feat_dim,
            action_dim=action_dim,
            n_steps=n_steps,
            hidden_units=hidden,
            depth=depth,
            hutchinson_samples=hutch,
            device=device,
            sanitize_actions=sanitize,
            low_bound=low,
            high_bound=high,
        )

    return DistActorCritic(
        get_actor_fn=actor_fn,
        get_critic_fn=critic_fn,
        get_dist_fn=fm_dist_fn,
    )


def get_fm_a2c_policy(env_name, args):
    def actor_fn(_, i_shape):
        return MLPBasic(i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers)
    def critic_fn(_, i_shape, asp):
        return MLPBasic(i_shape[0], hidden_size=args.ppo_hidden_dim, num_layers=args.ppo_layers)

    # Pack FM params
    fm_steps = getattr(args, 'fm_a2c_steps', 40)
    fm_hidden = getattr(args, 'fm_a2c_hidden', args.ppo_hidden_dim)
    fm_depth = getattr(args, 'fm_a2c_depth', max(2, args.ppo_layers))

    pol = FMA2CPolicy(get_actor_fn=actor_fn, get_critic_fn=critic_fn,
                      fm_steps=fm_steps, fm_hidden=fm_hidden, fm_depth=fm_depth)
    return pol

