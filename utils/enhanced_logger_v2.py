

import os
import csv
import json
import time
import datetime
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Warning: pandas, matplotlib, seaborn not available. Some features disabled.")
    pd = None
    plt = None
    sns = None

class EnhancedLocalLoggerV2:
    """
    
    new directory structure, support multi-seed experiments
    """
    
    def __init__(self, log_dir: str, env_name: str, subfolder: str, model_name: str, 
                 seed: int, config: Dict[str, Any], allowed_metrics: List[str] = None):
        """
        initialize logger
        
        Args:
            log_dir: 日志根目录 (eg: data/log)
            env_name: environment name (eg: ant, pick, push, hand, maze, walker)
            subfolder: subfolder name/parameter value (eg: 000, 125, 75, 5traj)
            model_name: model name (eg: drail, gail, etc.)
            seed: random seed
            config: experiment configuration
            allowed_metrics: allowed metrics to record
        """
        self.log_dir = Path(log_dir)
        self.env_name = env_name
        self.subfolder = subfolder
        self.model_name = model_name
        self.seed = seed
        self.config = config
        
        # set allowed metrics
        if allowed_metrics is None:
            self.allowed_metrics = {
                'avg_r', 'avg_ep_found_goal', 'dist_entropy', 
                'timestamp', 'step', 'episode'
            }
        else:
            self.allowed_metrics = set(allowed_metrics)
        
        # create new directory structure: envname_subfoldername_model/seed/
        experiment_folder = f"{env_name}_{subfolder}_{model_name}"
        self.experiment_dir = self.log_dir / experiment_folder / f"seed{seed}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # file path
        self.metrics_file = self.experiment_dir / "metrics.csv"
        self.summary_file = self.experiment_dir / "summary.json"
        
        # performance metrics recording
        self.metrics_data = []
        self.current_metrics = {}
        self.csv_initialized = False
        
        # record experiment start time
        self.start_time = time.time()
        self._save_config()
        
        print(f"Local logger V2 initialized: {self.experiment_dir}")
        
    def _init_csv_file(self, first_metrics: Dict[str, Any]):
        """initialize CSV file based on the first recorded metrics"""
        if self.csv_initialized:
            return
        
        # base columns
        base_columns = [
            'timestamp', 'step', 'episode', 'total_env_steps', 'wall_time'
        ]
        
        # extract column names from actual metrics
        metric_columns = []
        for key in first_metrics.keys():
            if isinstance(first_metrics[key], (int, float, np.integer, np.floating)):
                metric_columns.append(key)
        
        all_columns = base_columns + sorted(metric_columns)
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
        
        self.csv_columns = all_columns
        self.csv_initialized = True
        print(f"CSV file initialized, contains {len(all_columns)} columns: {self.metrics_file}")
        
    def _save_config(self):
        """save experiment configuration"""
        config_data = {
            'env_name': self.env_name,
            'subfolder': self.subfolder,
            'model_name': self.model_name,
            'seed': self.seed,
            'start_time': datetime.datetime.now().isoformat(),
            'config': self.config
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int, episode: int = None):
        """
            record performance metrics
        
        Args:
            metrics: metrics dictionary
            step: current step
            episode: current episode
        """
        # normalize metrics
        normalized_metrics = self._normalize_metrics(metrics)
        
        # initialize CSV file when the first record is made
        if not self.csv_initialized:
            self._init_csv_file(normalized_metrics)
        
        # prepare record data
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'step': step,
            'episode': episode or 0,
            'total_env_steps': step,
            'wall_time': time.time() - self.start_time
        }
        
        # add metrics
        record.update(normalized_metrics)
        
        # save to CSV
        self._save_to_csv(record)
        
        # update current metrics
        self.current_metrics.update(record)
        self.metrics_data.append(record)
        
        # save summary periodically
        if len(self.metrics_data) % 50 == 0:
            self._save_summary()
    
    def _normalize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """normalize metric names and filter allowed metrics"""
        normalized = {}
        
        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                key = f"{prefix}_{k}" if prefix else k
                
                # only process allowed metrics
                if key not in self.allowed_metrics:
                    continue
                    
                if isinstance(v, dict):
                    normalized.update(flatten_dict(v, key))
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    normalized[key] = float(v)
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], (int, float)):
                    # if it is a list of numbers, take the average
                    normalized[f"{key}_mean"] = float(np.mean(v))
        
        flatten_dict(metrics)
        
        # directly check top-level keys
        for key, value in metrics.items():
            if key in self.allowed_metrics and isinstance(value, (int, float, np.integer, np.floating)):
                normalized[key] = float(value)
        
        return normalized
    
    def _save_to_csv(self, record: Dict[str, Any]):
        """save record to CSV file"""
        try:
            # only save columns defined in CSV
            filtered_record = {k: record.get(k, '') for k in self.csv_columns}
            
            with open(self.metrics_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_columns)
                writer.writerow(filtered_record)
        except Exception as e:
            print(f"Error saving CSV record: {e}")
    
    def _save_summary(self):
        """save experiment summary"""
        if not self.metrics_data:
            return
        
        try:
            summary = {
                'env_name': self.env_name,
                'subfolder': self.subfolder,
                'model_name': self.model_name,
                'seed': self.seed,
                'total_records': len(self.metrics_data),
                'last_step': self.metrics_data[-1]['step'] if self.metrics_data else 0,
                'total_time': time.time() - self.start_time,
                'last_update': datetime.datetime.now().isoformat()
            }
            
            if pd is not None and self.metrics_data:
                # use pandas to calculate detailed statistics
                df = pd.DataFrame(self.metrics_data)
                
                # calculate statistics for key metrics
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                key_metrics = [col for col in numeric_columns if col in self.allowed_metrics]
                
                summary['metrics_summary'] = {}
                for metric in key_metrics:
                    values = df[metric].dropna()
                    if len(values) > 0:
                        summary['metrics_summary'][metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'latest': float(values.iloc[-1])
                        }
            
            # save summary
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving summary: {e}")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """get latest metrics"""
        return self.current_metrics.copy()
    
    def close(self):
        """close logger"""
        self._save_summary()
        print(f"Experiment logs saved to: {self.experiment_dir}")


def parse_experiment_info(config_path: str) -> Tuple[str, str, str]:
    """
    parse experiment information from config file path
    
    Args:
        config_path: config file path, e.g. "configs/ant/0.00/drail.yaml"
    
    Returns:
        (env_name, subfolder, model_name)
    """
    path_parts = Path(config_path).parts
    
    if len(path_parts) >= 4 and path_parts[0] == 'configs':
        env_name = path_parts[1]
        subfolder = path_parts[2]
        model_name = Path(path_parts[3]).stem  # remove .yaml suffix
        
        # normalize environment name
        env_name_mapping = {
            'ant': 'ant',
            'pick': 'pick', 
            'push': 'push',
            'hand': 'hand',
            'maze': 'maze',
            'walker': 'walker'
        }
        env_name = env_name_mapping.get(env_name, env_name)
        
        # normalize subfolder name (remove dots, slashes, etc.)
        subfolder_clean = subfolder.replace('.', '').replace('/', '-')
        
        return env_name, subfolder_clean, model_name
    
    # fallback: infer from filename
    filename = Path(config_path).stem
    return 'unknown', 'unknown', filename


# global logger instance
_global_logger: Optional[EnhancedLocalLoggerV2] = None

def init_local_logger_v2(log_dir: str, config_path: str, seed: int, 
                        config: Dict[str, Any], allowed_metrics: List[str] = None) -> EnhancedLocalLoggerV2:
    """
    initialize global local logger V2
    
    Args:
        log_dir: log directory
        config_path: config file path, for parsing experiment information
        seed: random seed
        config: experiment configuration
        allowed_metrics: allowed metrics to record
    
    Returns:
        logger instance
    """
    global _global_logger
    
    # parse experiment information
    env_name, subfolder, model_name = parse_experiment_info(config_path)
    
    _global_logger = EnhancedLocalLoggerV2(
        log_dir, env_name, subfolder, model_name, seed, config, allowed_metrics
    )
    return _global_logger

def log_metrics(metrics: Dict[str, Any], step: int, episode: int = None):
    """
    global logger function
    
    Args:
        metrics: metrics dictionary
        step: current step
        episode: current episode
    """
    if _global_logger is not None:
        _global_logger.log_metrics(metrics, step, episode)

def close_logger():
    """close global logger"""
    global _global_logger
    if _global_logger is not None:
        _global_logger.close()
        _global_logger = None

def get_logger() -> Optional[EnhancedLocalLoggerV2]:
    """get global logger"""
    return _global_logger