#!/usr/bin/env python3


import re
from pathlib import Path
from typing import Tuple

def normalize_env_name(env_name: str) -> str:
 
    env_mapping = {
        'AntGoal-v0': 'ant',
        'FetchPickAndPlaceDiffHoldout-v0': 'pick',
        'FetchPickAndPlaceCustom-v0': 'pick', 
        'FetchPushEnvCustom-v0': 'push',
        'CustomHandReach-v0': 'hand',
        'maze2d-medium-v2': 'maze',
        'Walker2d-v3': 'walker'
    }
    
   
    if env_name in env_mapping:
        return env_mapping[env_name]
    
  
    env_lower = env_name.lower()
    if 'ant' in env_lower:
        return 'ant'
    elif 'pick' in env_lower or 'place' in env_lower:
        return 'pick'
    elif 'push' in env_lower:
        return 'push'
    elif 'hand' in env_lower or 'reach' in env_lower:
        return 'hand'
    elif 'maze' in env_lower:
        return 'maze'
    elif 'walker' in env_lower:
        return 'walker'
    

    clean_name = re.sub(r'[^a-zA-Z0-9]', '', env_name).lower()
    return clean_name[:10] 

def extract_subfolder_from_config(args) -> str:
   
    
    config_path = None
    for attr in ['config_path', 'config', 'config_file']:
        if hasattr(args, attr):
            config_path = getattr(args, attr)
            break
    
   
    if not config_path:
        env_name = normalize_env_name(getattr(args, 'env_name', ''))
        
     
        env_noise_params = {
            'ant': ['ant_noise'],
            'hand': ['noise_ratio', 'noise-ratio'], 
            'pick': ['noise_ratio', 'noise-ratio'],
            'push': ['noise_ratio', 'noise-ratio'],
            'maze': ['expert_coverage', 'coverage'],
            'walker': ['traj_count', 'num_traj'],
        }
        
  
        if env_name in env_noise_params:
            for param_name in env_noise_params[env_name]:
             
                for attr_variant in [param_name, param_name.replace('-', '_'), param_name.replace('_', '-')]:
                    if hasattr(args, attr_variant):
                        param_val = getattr(args, attr_variant)
                        if isinstance(param_val, (int, float)):
                         
                            if env_name in ['ant', 'hand', 'pick', 'push']:
                                
                                noise_str = f"{param_val:.2f}".replace('.', '')
                                result = noise_str.zfill(3)
                                print(f"🔧 {env_name} environment: {attr_variant}={param_val} -> {result}")
                                return result
                            elif env_name == 'maze':
                                
                                print(f"🔧 {env_name} environment: {attr_variant}={param_val} -> {int(param_val)}")
                                return str(int(param_val))
                            elif env_name == 'walker':
                             
                                print(f"🔧 {env_name} environment: {attr_variant}={param_val} -> {param_val}")
                                return str(param_val)
        
  
        if env_name == 'maze' and hasattr(args, 'traj_load_path'):
            traj_path = getattr(args, 'traj_load_path')
            print(f"🔧 {env_name} environment: from traj_load_path extract coverage: {traj_path}")
            import re
            m_cov = re.search(r"maze2d_(\d+)\.pt", str(traj_path))
            if m_cov:
                coverage = m_cov.group(1)
                print(f"🔧 {env_name} environment: traj_load_path -> {coverage}")
                return coverage
        
    
        if hasattr(args, 'cwd'):
            cwd = getattr(args, 'cwd')
            import os
            if os.path.exists(os.path.join(cwd, 'configs')):
                configs_dir = os.path.join(cwd, 'configs', env_name)
                if os.path.exists(configs_dir):
                
                    alg_name = getattr(args, 'alg', getattr(args, 'prefix', ''))
                    for root, dirs, files in os.walk(configs_dir):
                        for file in files:
                            if file.endswith(f'{alg_name}.yaml'):
                                relative_path = os.path.relpath(os.path.join(root, file), cwd)
                                config_path = relative_path
                                break
                        if config_path:
                            break
        
        if not config_path:
            print(f"⚠️  {env_name} environment: cannot infer subfolder, using default")
            return 'default'
    
    try:
       
        parts = Path(config_path).parts
        if len(parts) >= 3 and parts[0] == 'configs':
            subfolder = parts[2]
            env_name = normalize_env_name(getattr(args, 'env_name', ''))
            

            if env_name in ['ant', 'pick', 'push', 'hand']:
                
                subfolder_clean = subfolder.replace('.', '')
                if len(subfolder_clean) <= 3:
                    subfolder_clean = subfolder_clean.zfill(3)
                return subfolder_clean
            elif env_name == 'maze':
                
                return subfolder
            elif env_name == 'walker':
                
                return subfolder
            else:
                return subfolder.replace('.', '').replace('/', '-')
    except Exception as e:
        print(f"⚠️  parse config path failed: {e}")
    
    return 'default' 
def extract_model_name(args) -> str:
   
   
    config_path = getattr(args, 'config_path', getattr(args, 'config', ''))
    if config_path:
        model_name = Path(config_path).stem 
        
       
        model_mapping = {
            'drail-un': 'drail-un',
            'drail': 'drail', 
            'gailGP': 'gailGP',
            'gail': 'gail',
            'wail': 'wail',
            'bc': 'bc',
            'diffusion-policy': 'dp',
            'dp': 'dp'  
        }
        return model_mapping.get(model_name, model_name)
    
   
    if hasattr(args, 'alg'):
        alg_name = str(args.alg)
       
        model_mapping = {
            'drail-un': 'drail-un',
            'drail': 'drail', 
            'gailGP': 'gailGP',
            'gail': 'gail',
            'wail': 'wail',
            'bc': 'bc',
            'dp': 'dp'
        }
        return model_mapping.get(alg_name, alg_name)
    
    return 'unknown'

def create_clean_experiment_name(args) -> str:
   
    env_name = normalize_env_name(getattr(args, 'env_name', 'unknown'))
    subfolder = extract_subfolder_from_config(args)
    model_name = extract_model_name(args)
    
    clean_name = f"{env_name}_{subfolder}_{model_name}"
    return clean_name

def parse_clean_experiment_name(experiment_name: str) -> Tuple[str, str, str]:
   
    parts = experiment_name.split('_')
    if len(parts) >= 3:
        env_name = parts[0]
        subfolder = parts[1]
        model_name = '_'.join(parts[2:]) 
        return env_name, subfolder, model_name
    return 'unknown', 'unknown', 'unknown'

def format_subfolder_description(env_name: str, subfolder: str) -> str:

    if env_name == 'ant' and subfolder.isdigit() and len(subfolder) == 3:
       
        noise_level = f"0.{subfolder[1:]}"
        return f'Noise Level {noise_level}'
    elif env_name in ['pick', 'push', 'hand'] and subfolder.isdigit():
        
        if len(subfolder) == 3:
            noise_level = f"{subfolder[0]}.{subfolder[1:]}"
        else:
            noise_level = subfolder
        return f'Noise Level {noise_level}'
    elif env_name == 'maze' and subfolder.isdigit():
        return f'Expert Coverage {subfolder}%'
    elif env_name == 'walker':
        return f'Trajectory Count {subfolder}'
    else:
        return subfolder

if __name__ == "__main__":
    
    print("🧪 test clean naming tool...")
    
    class MockArgs:
        def __init__(self, env_name, config_path):
            self.env_name = env_name
            self.config_path = config_path
    
    test_cases = [
        ('AntGoal-v0', 'configs/ant/0.00/bc.yaml'),
        ('FetchPickAndPlaceDiffHoldout-v0', 'configs/pick/1.25/drail.yaml'),
        ('maze2d-medium-v2', 'configs/maze/75/gail.yaml'),
    ]
    
    for env_name, config_path in test_cases:
        args = MockArgs(env_name, config_path)
        clean_name = create_clean_experiment_name(args)
        env, sub, model = parse_clean_experiment_name(clean_name)
        desc = format_subfolder_description(env, sub)
        print(f"📝 {env_name} + {config_path}")
        print(f"   → {clean_name}")
        print(f"   → parse: {env}, {desc}, {model}")
        print()
    
    print("✅ test done")