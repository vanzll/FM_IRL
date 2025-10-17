# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os
import csv
import time
from pathlib import Path
from typing import Dict, Any

# global variable to store current logger state
_logger_state = {
    'initialized': False,
    'metrics_file': None,
    'experiment_dir': None,
    'csv_writer': None,
    'csv_file_handle': None,
    'fieldnames': None
}

def init_simple_local_logger(args, log_dir='data/log'):
    """initialize simple local logger"""
    global _logger_state
    
    try:
        # determine experiment directory structure: data/log/envname_subfoldername_model/seed/
        env_name = getattr(args, 'env_name', 'Unknown')
        prefix = getattr(args, 'prefix', 'unknown')
        seed = getattr(args, 'seed', 1)
        
        # parse environment name, remove version number etc.
        env_base = env_name.replace('-v0', '').replace('-v1', '').replace('-v2', '').replace('-v3', '')
        
        # build experiment folder name
        experiment_name = f"{env_base.lower()}_{prefix}"
        seed_name = f"seed{seed}"
        
        # create directory path
        experiment_dir = Path(log_dir) / experiment_name / seed_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # create metrics.csv file
        metrics_file = experiment_dir / 'metrics.csv'
        
        # initialize CSV writer
        csv_file_handle = open(metrics_file, 'w', newline='', encoding='utf-8')
        
        # update global state
        _logger_state.update({
            'initialized': True,
            'metrics_file': metrics_file,
            'experiment_dir': experiment_dir,
            'csv_file_handle': csv_file_handle,
            'csv_writer': None,  # will be initialized when first write
            'fieldnames': None
        })
        
        print(f"✅ Simple local logger initialized")
        print(f"   Experiment directory: {experiment_dir}")
        print(f"   Metrics file: {metrics_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialize simple local logger failed: {e}")
        return False

def _compute_experiment_dir(args, log_dir='data/log'):
    """Compute experiment directory path deterministically from args.
    Does not require prior initialization."""
    env_name = getattr(args, 'env_name', 'Unknown')
    prefix = getattr(args, 'prefix', 'unknown')
    seed = getattr(args, 'seed', 1)
    env_base = env_name.replace('-v0', '').replace('-v1', '').replace('-v2', '').replace('-v3', '')
    experiment_name = f"{env_base.lower()}_{prefix}"
    seed_name = f"seed{seed}"
    experiment_dir = Path(log_dir) / experiment_name / seed_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir

def log_time_metrics(args, metrics: Dict[str, Any], filename: str = 'time.csv'):
    """Append timing metrics to a dedicated CSV in the seed folder.
    This function is safe to call without prior logger initialization."""
    try:
        experiment_dir = _compute_experiment_dir(args, getattr(args, 'log_dir', 'data/log'))
        time_file = experiment_dir / filename
        write_header = not time_file.exists()
        fieldnames = ['timestamp', 'update_idx', 'time_update_s', 'num_steps', 'num_envs', 'alg', 'env']
        # Merge defaults
        row = {
            'timestamp': time.time(),
            'update_idx': metrics.get('update_idx', ''),
            'time_update_s': metrics.get('time_update_s', ''),
            'num_steps': metrics.get('num_steps', ''),
            'num_envs': metrics.get('num_envs', ''),
            'alg': getattr(args, 'alg', ''),
            'env': getattr(args, 'env_name', ''),
        }
        # Add any extra keys
        for k, v in metrics.items():
            if k not in row:
                row[k] = v
                if k not in fieldnames:
                    fieldnames.append(k)

        # If file exists with different header, we append columns dynamically
        if write_header:
            with open(time_file, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerow(row)
        else:
            # Read existing header
            with open(time_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_fields = next(reader, [])
            # If new fields, rewrite file with expanded header
            if any(k not in existing_fields for k in fieldnames):
                # Load existing rows
                with open(time_file, 'r', encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                # Expand header
                merged_fields = list(dict.fromkeys([*existing_fields, *fieldnames]))
                with open(time_file, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=merged_fields)
                    w.writeheader()
                    for r in rows:
                        w.writerow(r)
                    w.writerow(row)
            else:
                with open(time_file, 'a', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=existing_fields)
                    w.writerow(row)
    except Exception as e:
        print(f"⚠️  Write time log failed: {e}")

def log_simple_metrics(metrics: Dict[str, Any], step: int):
    """record metrics to CSV file"""
    global _logger_state
    
    if not _logger_state['initialized']:
        print("⚠️  Logger not initialized")
        return
    
    try:
        # add step to metrics
        metrics_with_step = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        
        # if this is the first write, initialize CSV writer
        if _logger_state['csv_writer'] is None:
            fieldnames = list(metrics_with_step.keys())
            _logger_state['fieldnames'] = fieldnames
            _logger_state['csv_writer'] = csv.DictWriter(
                _logger_state['csv_file_handle'], 
                fieldnames=fieldnames
            )
            _logger_state['csv_writer'].writeheader()
        
        # write data row, only include known fields
        row_data = {}
        for field in _logger_state['fieldnames']:
            if field in metrics_with_step:
                row_data[field] = metrics_with_step[field]
            else:
                row_data[field] = ''  # empty value handling
        
        # add new fields (if any)
        new_fields = [k for k in metrics_with_step.keys() if k not in _logger_state['fieldnames']]
        if new_fields:
            # need to re-create CSV writer to include new fields
            _logger_state['fieldnames'].extend(new_fields)
            _logger_state['csv_file_handle'].close()
            
            # re-open file in append mode
            _logger_state['csv_file_handle'] = open(_logger_state['metrics_file'], 'a', newline='', encoding='utf-8')
            _logger_state['csv_writer'] = csv.DictWriter(
                _logger_state['csv_file_handle'], 
                fieldnames=_logger_state['fieldnames']
            )
        
        _logger_state['csv_writer'].writerow(metrics_with_step)
        _logger_state['csv_file_handle'].flush()
        
    except Exception as e:
        print(f"⚠️  Record metrics failed: {e}")

def close_simple_logger():
    """close logger"""
    global _logger_state
    
    try:
        if _logger_state['csv_file_handle']:
            _logger_state['csv_file_handle'].close()
        
        _logger_state.update({
            'initialized': False,
            'metrics_file': None,
            'experiment_dir': None,
            'csv_writer': None,
            'csv_file_handle': None,
            'fieldnames': None
        })
        
        print("✅ Simple local logger closed")
        
    except Exception as e:
        print(f"⚠️  Close logger failed: {e}")

def get_logger_status():
    """get logger status"""
    return _logger_state.copy()