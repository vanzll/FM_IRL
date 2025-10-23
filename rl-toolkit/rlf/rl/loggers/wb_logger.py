import copy
import datetime
import os
import os.path as osp
import pipes
import random
import string
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
from rlf.exp_mgr import config_mgr
from rlf.rl import utils
from rlf.rl.loggers.base_logger import BaseLogger
from six.moves import shlex_quote
import matplotlib.pyplot as plt


try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from utils.simple_local_logger import init_simple_local_logger, log_simple_metrics, close_simple_logger
    SIMPLE_LOGGER_AVAILABLE = True
except ImportError:
    print("Simple local logger not available")
    SIMPLE_LOGGER_AVAILABLE = False

# Does not necessarily have WB installed
try:
    import wandb
except:
    pass

# Does not necessarily have Ray installed
try:
    from ray.tune.integration.wandb import WandbLogger
    from ray.tune.logger import DEFAULT_LOGGERS
except:
    pass


def get_wb_ray_kwargs():
    return {"loggers": DEFAULT_LOGGERS + (WandbLogger,)}


def get_wb_ray_config(config):
    config["wandb"] = {
        "project": config_mgr.get_prop("proj_name"),
        "api_key": config_mgr.get_prop("wb_api_key"),
        "log_config": True,
    }
    return config


def get_wb_media(v):
    if isinstance(v, torch.Tensor) and len(v.shape) == 1 and v.shape[0] > 1:
        v = wandb.Histogram(v.numpy())
    if isinstance(v, np.ndarray) and len(v.shape) == 1 and v.shape[0] > 1:
        v = wandb.Histogram(v)

    return v


class WbLogger(BaseLogger):
    """
    Logger for logging to the weights and W&B online service.
    """

    def __init__(
        self,
        wb_proj_name=None,
        should_log_vids=False,
        wb_entity=None,
        skip_create_wb=False,
        enable_local_logging=True,
    ):
        """
        - wb_proj_name: (string) if None, will use the proj_name provided in
          the `config.yaml` file.
        - enable_local_logging: (bool) whether to enable local CSV logging
        """
        super().__init__()
        if wb_proj_name is None:
            wb_proj_name = config_mgr.get_prop("proj_name")
        if wb_entity is None:
            wb_entity = config_mgr.get_prop("wb_entity")
        self.wb_proj_name = wb_proj_name
        self.wb_entity = wb_entity
        self.should_log_vids = should_log_vids
        self.skip_create_wb = skip_create_wb
        self.enable_local_logging = enable_local_logging and SIMPLE_LOGGER_AVAILABLE
        self.local_logger_initialized = False
        self.is_closed = False  

    def init(self, args, mod_prefix=lambda x: x):
        super().init(args, mod_prefix)
        
  
        if self.enable_local_logging and not self.local_logger_initialized and SIMPLE_LOGGER_AVAILABLE:
            try:
                log_dir = getattr(args, 'log_dir', 'data/log')
                init_simple_local_logger(args, log_dir)
                self.local_logger_initialized = True
          
            except Exception as e:
          
                self.enable_local_logging = False
        
        if self.skip_create_wb:
            return
        self.wandb = self._create_wandb(args)

    def _internal_log_vals(self, key_vals, step_count, histo=False):
        if self.is_closed:
            return
        
   
        if self.enable_local_logging:
            try:
              
                local_metrics = {}
                for k, v in key_vals.items():
                    if k[0] != '_':  
                        if isinstance(v, (int, float, np.integer, np.floating)):
                            local_metrics[k] = v
                        elif isinstance(v, (torch.Tensor)) and v.numel() == 1:
                            local_metrics[k] = v.item()
                        elif isinstance(v, (list, tuple)) and len(v) > 0:
                            if all(isinstance(x, (int, float)) for x in v):
                                local_metrics[f"{k}_mean"] = np.mean(v)
                
                if local_metrics:
                    log_simple_metrics(local_metrics, int(step_count))
            except Exception as e:
                pass
        
    
        if not histo:
            every_key_vals = {k: get_wb_media(v) for k, v in key_vals.items() if k[0] != '_'}
            if not self.skip_create_wb:
                wandb.log(every_key_vals, step=int(step_count))
            if "_reward_map" in key_vals:
                self.log_image("reward_map", key_vals["_reward_map"], int(step_count))
            if "_disc_val_map" in key_vals:
                self.log_image("disc_val_map", key_vals["_disc_val_map"], int(step_count))
        # else:
            # key_vals = {k: wandb.plot.histogram(
            #                     wandb.Table(data=v, columns=["rewards"]), 
            #                     "rewards", 
            #                     title=k
            #                 ) for k, v in key_vals.items() if k[0] == '_'}
            # for k, v in key_vals.items():
                # wandb.log({k: v}, step=step_count)
            # for k, v in key_vals.items():
            #     if k[0] == '_':
            #         plt.figure()
            #         plt.hist(np.asarray(v).squeeze(1), bins=20)
            #         plt.savefig(f"./data/hist/{k}.png")
            #         self.log_image(k, f"./data/hist/{k}.png", step_count)
                
        # except Exception as e:
        #    print(e)
        #    self.is_closed = True
        #    print("Wb logger was closed when trying to log")

    def watch_model(self, model, log_type="gradients", log_freq=1000, **kwargs):
        wandb.watch(model, log=log_type, log_freq=log_freq)

    def log_image(self, k, img_file, step_count):
        wandb.log({k: wandb.Image(img_file)}, step=step_count)

    def _create_wandb(self, args):
        args.prefix = self.prefix
        if self.prefix.count("-") >= 4:
            # Remove the seed and random ID info.
            parts = self.prefix.split("-")
            # group_id = "-".join([*parts[:2], *parts[4:]])
            group_id = self.prefix
        else:
            group_id = None

        self.run = wandb.init(
            project=self.wb_proj_name,
            name=self.prefix,
            entity=self.wb_entity,
            # group=group_id,
            reinit=True,
            config=args,
        )
        self.is_closed = False
        return wandb

    def get_config(self):
        return wandb.config

    def log_video(self, video_file, step_count, fps):
        if not self.should_log_vids:
            return
        wandb.log({"video": wandb.Video(video_file + ".mp4", fps=fps, format="mp4")}, step=step_count)

    def close(self):
        self.is_closed = True
        
     
        if self.enable_local_logging:
            try:
                close_simple_logger()
             
            except Exception as e:
                pass
        
      
        if not self.skip_create_wb:
            # Prefer finishing the run instead of calling save (API changed)
            try:
                if hasattr(self, 'run') and self.run is not None:
                    self.run.finish()
                else:
                    # Fallback to global finish if run handle not present
                    wandb.finish()
            except Exception as e:
                pass
