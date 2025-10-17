# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys

import copy
import gym
import numpy as np
import rlf.algos.utils as autils
import rlf.rl.utils as rutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from rlf.algos.il.base_il import BaseILAlgo
from rlf.args import str2bool
from rlf.storage.base_storage import BaseStorage
from tqdm import tqdm
import wandb
import math
import time
import matplotlib.pyplot as plt

class DiffPolicy(BaseILAlgo):

    def __init__(self, set_arg_defs=True):
        super().__init__()
        self.set_arg_defs = set_arg_defs

    def init(self, policy, args):
        super().init(policy, args)
        self.num_epochs = 0
        self.action_dim = rutils.get_ac_dim(self.policy.action_space)
        if self.args.bc_state_norm:
            self.norm_mean = self.expert_stats["state"][0]
            self.norm_var = torch.pow(self.expert_stats["state"][1], 2)
        else:
            self.norm_mean = None
            self.norm_var = None
        self.num_bc_updates = 0
        self.L1 = nn.L1Loss().cuda()
        self.start_time = time.time()
        # self.lambda_bc = args.lambda_bc
        # self.lambda_dm = args.lambda_dm
        num_steps = 1000
        dim = 2

    def get_env_settings(self, args):
        settings = super().get_env_settings(args)
        if args.bc_state_norm:
            print("Setting environment state normalization")
            settings.state_fn = self._norm_state
        return settings

    def _norm_state(self, x):
        obs_x = torch.clamp(
            (rutils.get_def_obs(x) - self.norm_mean)
            / (torch.pow(self.norm_var, 0.5) + 1e-8),
            -10.0,
            10.0,
        )
        if isinstance(x, dict):
            x["observation"] = obs_x
            return x
        else:
            return obs_x

    def get_num_updates(self):
        if self.exp_generator is None:
            return len(self.expert_train_loader) * self.args.bc_num_epochs
        else:
            return self.args.exp_gen_num_trans * self.args.bc_num_epochs

    def get_completed_update_steps(self, num_updates):
        return num_updates * self.args.traj_batch_size

    def _reset_data_fetcher(self):
        super()._reset_data_fetcher()
        self.num_epochs += 1

    def first_train(self, log, eval_policy, env_interface):
        """在训练开始前创建初始日志记录"""
        print("🚀 Diffusion Policy训练开始，创建初始日志记录...")
        
        # 记录训练开始的初始指标
        initial_metrics = {
            'step': 0,
            'episode': 0,
            'epoch': 0,
            'training_progress': 0.0,
            'predict_loss': 0.0,
            'diffusion_loss': 0.0,
            'timestamp': time.time(),
            'wall_time': time.time(),
            # 初始占位符，表示尚未评估
            'avg_r': 0.0,  
            'avg_ep_found_goal': 0.0,
            'dist_entropy': 0.0,
        }
        
        # 记录到日志
        log.log_vals(initial_metrics, 0)
        print(f"✅ DP初始日志记录已创建")
        
        # 调用父类的first_train（如果存在）
        if hasattr(super(), 'first_train'):
            super().first_train(log, eval_policy, env_interface)
        
        # 保存评估函数和其他必需的引用，供训练结束后使用
        self._log = log
        self._eval_policy = eval_policy
        self._env_interface = env_interface

    def full_train(self, update_iter=0):
        action_loss = []
        prev_num = 0

        # Diffusion Policy训练循环
        print(f"🎯 开始Diffusion Policy训练 ({self.args.bc_num_epochs} epochs)")
        with tqdm(total=self.args.bc_num_epochs) as pbar:
            while self.num_epochs < self.args.bc_num_epochs:
                super().pre_update(self.num_bc_updates)
                log_vals = self._bc_step(False)
                action_loss.append(log_vals.get("_pr_predict_loss", 0))
                
                pbar.update(self.num_epochs - prev_num)
                prev_num = self.num_epochs

        # 保存训练损失图
        rutils.plot_line(
            action_loss,
            f"diffusion_loss_{update_iter}.png",
            self.args.vid_dir,
            not self.args.no_wb,
            self.get_completed_update_steps(self.update_i),
        )
        
        # 训练结束后进行性能评估
        print("📊 Diffusion Policy训练完成，开始性能评估...")
        self._post_training_evaluation()
        self._plot_circle_predictions()
        
        self.num_epochs = 0

    def _post_training_evaluation(self):
        """在训练结束后进行性能评估并记录结果"""
        if not hasattr(self, '_eval_policy') or not hasattr(self, '_log'):
            print("⚠️  评估函数不可用，跳过性能评估")
            return
            
        try:
            # 创建评估参数
            import copy
            eval_args = copy.copy(self.args)
            eval_args.eval_num_processes = min(20, getattr(self.args, 'eval_num_processes', 10))
            eval_args.num_eval = getattr(self.args, 'num_eval', 100)
            eval_args.num_render = 0
            
            print(f"🔍 使用{eval_args.num_eval}个episodes评估策略性能...")
            
            # 运行评估
            tmp_env = self._eval_policy(self.policy, self.num_bc_updates, True, eval_args)
            
            # 关闭临时环境
            if tmp_env is not None:
                tmp_env.close()
            
            print("✅ Diffusion Policy性能评估完成")
            
        except Exception as e:
            print(f"⚠️  DP性能评估失败: {e}")
            # 即使评估失败，也记录一个最终的训练完成指标
            final_metrics = {
                'step': self.num_bc_updates,
                'episode': 0,
                'epoch': self.num_epochs,
                'training_progress': 1.0,
                'timestamp': time.time(),
                'wall_time': time.time(),
                'dp_training_complete': 1
            }
            
            try:
                self._log.log_vals(final_metrics, self.num_bc_updates)
                print("📝 已记录DP训练完成状态")
            except:
                print("⚠️  无法记录最终状态")

    def pre_update(self, cur_update):
        # Override the learning rate decay
        pass
    
    def _bc_step(self, decay_lr):
        if decay_lr:
            super().pre_update(self.num_bc_updates)
        expert_batch = self._get_next_data()
        if expert_batch is None:
            self._reset_data_fetcher()
            expert_batch = self._get_next_data()

        states, true_actions = self._get_data(expert_batch)
        
        log_dict = {}
        
        pred_loss = self.policy.get_loss(true_actions, states)

        self._standard_step(pred_loss) #backward
        self.num_bc_updates += 1

        # 添加标准日志指标
        log_dict["_pr_predict_loss"] = pred_loss.item()
        log_dict["predict_loss"] = pred_loss.item()  # 标准预测损失指标
        log_dict["diffusion_loss"] = pred_loss.item()  # 扩散损失指标
        
        # 添加训练进度相关的标准指标
        log_dict["step"] = self.num_bc_updates
        log_dict["episode"] = 0  # DP没有episode概念，设为0
        log_dict["epoch"] = self.num_epochs
        log_dict["training_progress"] = self.num_epochs / max(1, self.args.bc_num_epochs)
        log_dict["timestamp"] = time.time()
        log_dict["wall_time"] = time.time() - self.start_time
        
        return log_dict

    def _plot_circle_predictions(self):
        env_name = str(getattr(self.args, 'env_name', ''))
        if not env_name.startswith('Circle'):
            return
        try:
            device = getattr(self.args, 'device', 'cpu')
            s = np.linspace(-1.0, 1.0, 400, endpoint=True).astype(np.float32)
            states = torch.from_numpy(s).view(-1, 1).to(device)
            self.policy.eval()
            with torch.no_grad():
                actions, _, _ = self.policy(states, None, None)
                a = actions.squeeze(-1).detach().cpu().numpy()

            os.makedirs('./data/imgs', exist_ok=True)
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            ax.scatter(s, a, c='#16a085', s=10, alpha=0.8, label='DP/FM-BC predictions')
            theta = np.linspace(0.0, 2.0 * np.pi, 400)
            ax.plot(np.cos(theta), np.sin(theta), c='black', lw=1.2, label='s^2 + a^2 = 1')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim([-1.05, 1.05])
            ax.set_ylim([-1.05, 1.05])
            ax.set_xlabel('state s')
            ax.set_ylabel('action a')
            ax.legend()
            ax.grid(True, ls='--', alpha=0.3)
            plt.tight_layout()
            out_fp = os.path.join('./data/imgs', f"{self.args.prefix}_dp_circle.png")
            plt.savefig(out_fp)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] DP/FM-BC circle plot failed: {e}")

    def _get_data(self, batch):
        states = batch["state"].to(self.args.device)
        if self.args.bc_state_norm:
            states = self._norm_state(states)

        if self.args.bc_noise is not None:
            add_noise = torch.randn(states.shape) * self.args.bc_noise
            states += add_noise.to(self.args.device)
            states = states.detach()

        true_actions = batch["actions"].to(self.args.device)
        true_actions = self._adjust_action(true_actions)
        return states, true_actions
    '''
    def _compute_val_loss(self):
        if self.update_i % self.args.eval_interval != 0:
            return None
        if self.val_train_loader is None:
            return None
        with torch.no_grad():
            action_losses = []
            diff_losses = []
            for batch in self.val_train_loader:
                states, true_actions = self._get_data(batch)
                pred_actions = self.policy.predict_action(true_actions.size(), states, self.args.device)
                action_loss = autils.compute_ac_loss(
                    pred_actions,
                    true_actions.view(-1, self.action_dim),
                    self.policy.action_space,
                )
                action_losses.append(action_loss.item())
                
                pred_loss = self.get_loss(states, pred_actions)
                pred_density = math.exp(-pred_loss)
                diff_losses.append(pred_loss.item())
            return np.mean(diff_losses)
    '''
    def update(self, storage, args, beginning, t):
        top_log_vals = super().update(storage) #actor_opt_lr
        log_vals = self._bc_step(True) #_pr_action_loss
        log_vals.update(top_log_vals) #_pr_action_loss
        return log_vals

    def get_storage_buffer(self, policy, envs, args):
        return BaseStorage()

    def get_add_args(self, parser):
        if not self.set_arg_defs:
            # This is set when BC is used at the same time as another optimizer
            # that also has a learning rate.
            self.set_arg_prefix("bc")

        super().get_add_args(parser)
        #########################################
        # Overrides
        if self.set_arg_defs:
            parser.add_argument("--num-processes", type=int, default=1)
            parser.add_argument("--num-steps", type=int, default=0)
            ADJUSTED_INTERVAL = 200
            parser.add_argument("--log-interval", type=int, default=ADJUSTED_INTERVAL)
            parser.add_argument(
                "--save-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
            parser.add_argument(
                "--eval-interval", type=int, default=100 * ADJUSTED_INTERVAL
            )
        parser.add_argument("--no-wb", default=False, action="store_true")

        #########################################
        # New args
        parser.add_argument("--bc-num-epochs", type=int, default=1)
        parser.add_argument("--bc-state-norm", type=str2bool, default=False)
        parser.add_argument("--bc-noise", type=float, default=None)
        parser.add_argument("--lambda-bc", type=float, default=1)
        parser.add_argument("--lambda-dm", type=float, default=1)
