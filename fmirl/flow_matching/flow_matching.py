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
import numpy as np
from typing import Optional, Tuple
import math
from torch.utils.checkpoint import checkpoint

class FlowMatchingModel(nn.Module):
    """
    Enhanced Flow Matching Model for FM-IL
    
    This model serves dual purposes:
    1. Reward modeling via distribution matching (Eq. 2)
    2. Policy regularization via behavior cloning (Eq. 4)
    
    Based on Flow Matching theory with conditional flows
    """
    
    def __init__(self, 
                 cond_dim: int = 6,
                 data_dim: int = 1, 
                 num_units: int = 128, 
                 depth: int = 4, 
                 device: str = 'cuda',
                 time_embed_dim: int = 2,
                 use_layer_norm: bool = True,
                 use_residual: bool = True):
        super(FlowMatchingModel, self).__init__()
        
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.device = device
        self.time_embed_dim = time_embed_dim
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Enhanced time embedding with sinusoidal encoding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, time_embed_dim),
            nn.LayerNorm(time_embed_dim) if use_layer_norm else nn.Identity()
        )
        
        # Main network with residual connections and normalization
        input_dim = cond_dim + data_dim + time_embed_dim
        
        self.input_layer = nn.Linear(input_dim, num_units)
        self.input_norm = nn.LayerNorm(num_units) if use_layer_norm else nn.Identity()
        
        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        
        for i in range(depth - 1):
            self.hidden_layers.append(nn.Linear(num_units, num_units))
            if use_layer_norm:
                self.hidden_norms.append(nn.LayerNorm(num_units))
            else:
                self.hidden_norms.append(nn.Identity())
        
        # Output layer
        self.output_layer = nn.Linear(num_units, data_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def sinusoidal_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding for better temporal representation
        """
        half_dim = self.time_embed_dim // 4
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
        
    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
      
        t = t.unsqueeze(-1).float()  # [batch_size, 1]
        return self.time_embed(t)  # [batch_size, time_embed_dim]
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  
        # Enhanced time embedding combining learned and sinusoidal features
        t_embed_learned = self.time_embedding(t)  # [batch_size, time_embed_dim]
        t_embed_sin = self.sinusoidal_time_embedding(t)  # [batch_size, time_embed_dim//2]
        
        # Combine different time embeddings
        if t_embed_sin.shape[1] < self.time_embed_dim:
            # Pad sinusoidal embedding if needed
            pad_size = self.time_embed_dim - t_embed_sin.shape[1]
            t_embed_sin = torch.cat([t_embed_sin, torch.zeros(t_embed_sin.shape[0], pad_size, device=self.device)], dim=1)
        
        t_embed = t_embed_learned + t_embed_sin[:, :self.time_embed_dim]
        
        # Concatenate inputs
        x_input = torch.cat([x, c, t_embed], dim=-1)
        
        # Input layer
        h = F.silu(self.input_norm(self.input_layer(x_input)))
        
        # Hidden layers with residual connections
        for i, (layer, norm) in enumerate(zip(self.hidden_layers, self.hidden_norms)):
            if self.use_residual and h.shape[-1] == layer.weight.shape[0]:
                residual = h
                h = F.silu(norm(layer(h)))
                h = h + residual  # Residual connection
            else:
                h = F.silu(norm(layer(h)))
        
        # Output layer
        velocity = self.output_layer(h)
        
        return velocity
    
    def sample_trajectory(self, 
                         x0: torch.Tensor, 
                         x1: torch.Tensor, 
                         t: torch.Tensor) -> torch.Tensor:
 

        t_expanded = t.unsqueeze(-1)  # [batch_size, 1]
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        return xt
    
    def compute_loss(self, 
                    x0: torch.Tensor, 
                    x1: torch.Tensor, 
                    c: torch.Tensor,
                    t: Optional[torch.Tensor] = None) -> torch.Tensor:
    
        batch_size = x0.shape[0]
        
        if t is None:
           
            t = torch.rand(batch_size, device=self.device)
        
     
        xt = self.sample_trajectory(x0, x1, t)
        
  
        true_velocity = x1 - x0  
        
       
        pred_velocity = self.forward(xt, c, t)  
        
      
        loss = F.mse_loss(pred_velocity, true_velocity, reduction='mean')
        
        return loss
    
    def compute_distance_metric(self, 
                               state: torch.Tensor, 
                               action: torch.Tensor,
                               expert_actions: torch.Tensor,
                               num_samples: int = 100) -> torch.Tensor:
       
        batch_size = state.shape[0]

   
        if num_samples <= 0:
            num_samples = 1
        base_t = torch.arange(num_samples, device=self.device, dtype=state.dtype) / num_samples
        noise_t = torch.rand(num_samples, batch_size, device=self.device, dtype=state.dtype) / num_samples # why need that?
        
        t = (base_t.view(num_samples, 1) + noise_t).clamp_(0.0, 1.0)  


        noise = torch.randn(num_samples, batch_size, action.shape[1], device=self.device, dtype=action.dtype) * 0.5
        a_t = (1.0 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * action.unsqueeze(0)

    
        a_t_flat = a_t.reshape(num_samples * batch_size, -1)
        state_flat = state.unsqueeze(0).expand(num_samples, batch_size, state.shape[1]).reshape(num_samples * batch_size, -1)
        t_flat = t.reshape(num_samples * batch_size)

        if self.training:
            pred_velocity_flat = checkpoint(self.forward, a_t_flat, state_flat, t_flat, use_reentrant=False)
        else:
            pred_velocity_flat = self.forward(a_t_flat, state_flat, t_flat)

        pred_velocity = pred_velocity_flat.view(num_samples, batch_size, -1)


        target_velocity = action.unsqueeze(0) - noise

        diff = pred_velocity - target_velocity  
        l2_norm = torch.norm(diff, dim=2, keepdim=True)  
        
       
        avg_distance = l2_norm.mean(dim=0).squeeze(-1)  # [B]
        temperature = 0.1
        return avg_distance * temperature
    
    def compute_reward(self, 
                      state: torch.Tensor, 
                      action: torch.Tensor,
                      expert_actions: torch.Tensor = None) -> torch.Tensor:
  
        if expert_actions is None:
            expert_actions = action  
            
      
        dist = self.compute_distance_metric(state, action, expert_actions)
        
        reward = torch.exp(-dist)

        print(f"reward: {reward}")
        
        return reward
    
    def generate(self, 
                c: torch.Tensor, 
                x0: Optional[torch.Tensor] = None,
                num_steps: int = 100) -> torch.Tensor:
  
        batch_size = c.shape[0]
        
        if x0 is None:
       
            x0 = torch.randn(batch_size, self.data_dim, device=self.device)
        
    
        dt = 1.0 / num_steps
        xt = x0
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            
          
            velocity = self.forward(xt, c, t) 
            
          
            xt = xt + velocity * dt
        
        return xt
    
    def conditional_generate(self, 
                           c: torch.Tensor, 
                           target: torch.Tensor,
                           num_steps: int = 100) -> torch.Tensor:

        batch_size = c.shape[0]
        
    
        x0 = torch.randn(batch_size, self.data_dim, device=self.device)
        
       
        dt = 1.0 / num_steps
        xt = x0
        
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=self.device)
            
        
            velocity = self.forward(xt, c, t)
            
        
            xt = xt + velocity * dt
        
        return xt





def cosine_schedule(timesteps, s=0.008):

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) 


# =========================
# Flow Matching Behavior Cloning Policy
# =========================

class CreateAction:
    def __init__(self, action):
        self.action = action
        self.hxs = {}
        self.extra = {}
        self.take_action = action

from rlf.policies.basic_policy import BasicPolicy


class MLPFlowPolicy(BasicPolicy, nn.Module):
    """
    Minimal Flow Matching Behavior Cloning policy.

    Treats x=(s,a) with s as condition and learns the flow of a.
    Provides get_loss(actions, states) and get_action(state, ...)
    so it can be trained by the existing DiffPolicy loop.
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
        use_goal: bool = False,
        get_base_net_fn=None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.is_stoch = is_stoch
        self.use_goal = use_goal
        self.get_base_net_fn = get_base_net_fn
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Use label as condition (c in {1}) and x=[s,a] as data
        self.label_dim = 1
        self.flow_model = FlowMatchingModel(
            cond_dim=self.label_dim,
            data_dim=state_dim + action_dim,
            num_units=num_units,
            depth=depth,
            device=str(self.device),
        ).to(self.device)

    def init(self, obs_space, action_space, args):
        self.action_space = action_space
        self.obs_space = obs_space
        self.args = args

    def get_loss(self, expert_actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        expert_actions = expert_actions.to(self.device)
        states = states.to(self.device)
        batch = expert_actions.shape[0]
        # x1 = [s, a_expert], x0 = [s, noise_a]
        noise_a = torch.randn_like(expert_actions)
        x1 = torch.cat([states, expert_actions], dim=-1)
        x0 = torch.cat([states, noise_a], dim=-1)
        c = torch.ones(batch, self.label_dim, device=self.device)
        loss = self.flow_model.compute_loss(x0=x0, x1=x1, c=c)
        return loss

    def get_action(self, state, add_state, rnn_hxs, mask, step_info):
        state_tensor = state.to(self.device)
        batch = state_tensor.shape[0]
        with torch.no_grad():
            # Integrate x = [s, a] with c=1 and clamp s each step
            a = torch.randn(batch, self.action_dim, device=self.device)
            x = torch.cat([state_tensor, a], dim=-1)
            dt = 1.0 / max(1, self.n_steps)
            c = torch.ones(batch, self.label_dim, device=self.device)
            t_val = 0.0
            for _ in range(self.n_steps):
                t = torch.full((batch,), t_val, device=self.device)
                v = self.flow_model.forward(x, c, t)
                x = x + v * dt
                # keep state part fixed as the condition
                x[:, : self.state_dim] = state_tensor
                t_val += dt
            pred_action = x[:, self.state_dim :]
        return CreateAction(pred_action)



class MLPConditionFlowMatching(nn.Module):# not used!
    """
    Enhanced Flow Matching Wrapper for FM-IL Algorithm
    
    Compatible with original DRAIL interface while implementing
    the FM-IL algorithm components including:
    1. Reward modeling (Eq. 2)
    2. Policy regularization (Eq. 4) 
    3. Discriminator loss (Eq. 5)
    """
    
    def __init__(self, n_steps, cond_dim=6, data_dim=1, num_units=128, depth=4, device='cuda'):
        super(MLPConditionFlowMatching, self).__init__()
        
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.device = device
        self.n_steps = n_steps
        
 
        self.flow_model = FlowMatchingModel(
            cond_dim=cond_dim,
            data_dim=data_dim,
            num_units=num_units,
            depth=depth,
            device=device,
            use_layer_norm=True,
            use_residual=True
        )
        
        # Expert action buffer for reward computation
        self.expert_buffer = None
        self.expert_buffer_size = 1000
        
        # Training statistics
        self.training_stats = {
            'fm_loss': 0.0,
            'reward_loss': 0.0,
            'discriminator_loss': 0.0,
            'velocity_norm': 0.0
        }
        
    def forward(self, x, c, t):

        return self.flow_model.forward(x, c, t)
    
    def update_expert_buffer(self, expert_sa_pairs):
      
        if self.expert_buffer is None:
            self.expert_buffer = expert_sa_pairs.detach().clone()
        else:
            # 保持缓冲区大小固定
            self.expert_buffer = torch.cat([self.expert_buffer, expert_sa_pairs.detach()], dim=0)
            if self.expert_buffer.shape[0] > self.expert_buffer_size:
                self.expert_buffer = self.expert_buffer[-self.expert_buffer_size:]
    
    def flow_matching_loss(self, label, sa_pair, n_steps):
       
        batch_size = sa_pair.shape[0]
        
       
        state_dim = self.cond_dim
        state = sa_pair[:, :state_dim]
        action = sa_pair[:, state_dim:]
        
      
        if isinstance(label, (int, float)):
            label_tensor = torch.full((batch_size, 1), label, device=self.device)
        else:
            label_tensor = label.float()
        
      
        is_expert = (label_tensor.mean() > 0.5)
        
        if is_expert:
         
            x0 = torch.randn_like(action)  
            x1 = action  
            c = state 
            
            
            self.update_expert_buffer(sa_pair)
            
        else:

            x0 = torch.randn_like(action)
            x1 = action
            c = state
            loss_value = self.flow_model.compute_loss(x0, x1, c)
            
           
            self.training_stats['discriminator_loss'] = loss_value.item()
            
            return loss_value.unsqueeze(0).expand(batch_size, 1)
        
      
        loss = self.flow_model.compute_loss(x0, x1, c)
        
     
        self.training_stats['fm_loss'] = loss.item()
        with torch.no_grad():
            velocity = self.flow_model.forward(x1, c, torch.rand(batch_size, device=self.device))
            self.training_stats['velocity_norm'] = torch.norm(velocity, dim=1).mean().item()
        
       
        return loss.unsqueeze(0).expand(batch_size, 1)
    
    def compute_reward(self, state, action):
    
        if self.expert_buffer is not None:
           
            batch_size = state.shape[0]
            indices = torch.randint(0, self.expert_buffer.shape[0], (batch_size,))
            expert_sa = self.expert_buffer[indices]
            expert_actions = expert_sa[:, self.cond_dim:]
            
            reward = self.flow_model.compute_reward(state, action, expert_actions)
        else:
         
            reward = torch.zeros(state.shape[0], device=self.device)
            
        return reward
    
    def generate_action(self, state, num_steps=None):
  
        if num_steps is None:
            num_steps = self.n_steps // 10  
        
        return self.flow_model.generate(state, num_steps=num_steps)
    
    def p_sample_loop(self, cond, n_steps):
  
        batch_size = cond.shape[0]
        
        
        generated = self.flow_model.generate(cond, num_steps=n_steps)
        
        
        return [generated]
    
    def p_sample(self, x, c, t, betas, one_minus_alphas_bar_sqrt):

        dt = 1.0 / self.n_steps
        velocity = self.forward(x, c, t)
        next_x = x + velocity * dt
        
        return next_x
    
    def get_training_stats(self):
      
        return self.training_stats.copy()
    
    def reset_stats(self):
        
        for key in self.training_stats:
            self.training_stats[key] = 0.0