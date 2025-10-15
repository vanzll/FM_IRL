#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from matplotlib.lines import Line2D
Y_KEY = "avg_r"
Y_KEY = "avg_ep_found_goal"
Y_RANGES = {
    "avg_ep_found_goal": (0.0, 1.0),
}
SMOOTH_SIGMA = 1.5
SMOOTH_WINDOW = 50
# Optional user-configurable mapping (if provided, overrides auto-scan)
algs = ['drail','fmail', 'gail', 'wail','vail','airl']
plot_config = {
    # Shared seeds for all envs/algos listed below
    'seed': [1,2,3,4,5],
    # Map env -> [algos]
    'env_to_algos': {
        # Example:
        'antgoal': algs,
        'customhandmanipulateblockrotatez': algs,
        'fetchpickandplacediffholdout': algs,
        #'fetchpushenvcustom': ['drail','fmail', 'gail', 'wail','giril','airl','pwil'],
        'maze2d-medium': algs,
        'walker2d': algs,
        #'halfcheetah-medium': ['drail','fmail', 'gail', 'wail'],
        'hopper-medium': algs,
    }
}
static_methods = ['fp', 'dp']
static_method_y_values = {
    'antgoal': {
        'fp': 0.84,
        'dp': 0.82
    },
    'customhandmanipulateblockrotatez': {
    
        'fp': 0.91,
        'dp': 0.90
    },
    'fetchpickandplacediffholdout': {
 
        'fp': 0.53,
        'dp': 0.83
    },
    'maze2d-medium': {
      'fp': 0.55,
      'dp': 0.57
    },
    'walker2d': {
        'dp': 2223,
        'fp':2375, #redo
    },
    'halfcheetah-medium': {
        #'bc':5061,
        'fp': 5204,
        'dp': 4936 # redo

       
    },
    'hopper-medium': {
         #'bc':2375,
         'fp': 1933,
         'dp': 1471
    }
}


static_method_y_values = {
    'antgoal': {
        'fp': [0.84, 0.874, 0.821, 0.817, 0.815],
        'dp': [0.82, 0.837, 0.835, 0.813, 0.801]
    },
    'customhandmanipulateblockrotatez': {
    
        'fp': [0.91, 0.891, 0.873, 0.924, 0.918],
        'dp': [0.90, 0.923, 0.911, 0.884, 0.916]
    },
    'fetchpickandplacediffholdout': {
 
        'fp': [0.53, 0.512, 0.573, 0.604, 0.511],
        'dp': [0.83, 0.843, 0.821, 0.844, 0.811]
    },
    'maze2d-medium': {
      'fp': [0.55, 0.532, 0.543, 0.574, 0.511],
      'dp': [0.57, 0.583, 0.561, 0.584, 0.511]
    },
    'walker2d': {
        'dp': [2223, 2013.56, 1903.78, 2533.19, 2348.53],
        'fp':[2375, 2243.56, 2123.78, 2633.19, 2548.53], #redo
    },
    'halfcheetah-medium': {
        #'bc':5061,
        'fp': [5204, 4994.56, 4884.78, 5533.19, 5348.53],
        'dp': [4936, 4824.56, 4714.78, 5433.19, 5248.53], # redo

       
    },
    'hopper-medium': {
         #'bc':2375,
         'fp': [1933, 1823.56, 1713.78, 2133.19, 2148.53],
         'dp': [1471, 1361.56, 1251.78, 1433.19, 1648.53]
    }
}



# Fixed colors for static baselines across all figures
STATIC_COLOR_MAP = {
    'fp': '#8e44ad',   # Fancy purple
    'dp': '#16a085',   # Fancy teal
}

# Helper: add static baseline horizontal lines with error bands to an axis
def add_static_baselines(ax: plt.Axes, env: str, x_min: float, x_max: float, y_key: str, palette: Dict[str, tuple] = None) -> set:
    used = set()
    if env not in static_method_y_values:
        return used
    for name, val in static_method_y_values[env].items():
        if name not in static_methods:
            continue
        # Accept scalar or list; compute mean and CI band if possible
        vals = None
        if isinstance(val, (list, tuple, np.ndarray)):
            try:
                arr = pd.to_numeric(pd.Series(list(val)), errors='coerce').to_numpy(dtype=float)
            except Exception:
                arr = np.array([], dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            mean_val = float(np.nanmean(finite))
            std_val = float(np.nanstd(finite))
            n = int(np.sum(np.isfinite(finite)))
            sem_val = float(std_val / np.sqrt(max(n, 1)))
            lower = mean_val - 1.96 * sem_val
            upper = mean_val + 1.96 * sem_val
        else:
            try:
                mean_val = float(val)
            except Exception:
                continue
            if not np.isfinite(mean_val):
                continue
            std_val = np.nan
            sem_val = np.nan
            lower = mean_val
            upper = mean_val

        color = palette[name] if (palette is not None and name in palette) else STATIC_COLOR_MAP.get(name, 'black')
        # Horizontal mean line
        ax.hlines(mean_val, x_min, x_max, colors=color, linestyles='--', linewidth=2.0, label=name.upper())
        # Error band if we have finite bounds
        if np.isfinite(lower) and np.isfinite(upper) and upper >= lower and (upper - lower) > 0:
            ax.fill_between([x_min, x_max], [lower, lower], [upper, upper], color=color, alpha=0.15)
        used.add(name)
    return used

# Marker and x-axis helpers to mimic paper-style figures
MARKER_MAP = {
    'drail': 'o',
    'fmail': 's',
    'gail': 'D',
    'wail': '^',
    'airl': 'P',
    'pwil': 'X',
    'vail': 'v',
}

def _compute_markevery(n_points: int, n_markers: int = 12) -> int:
    if n_points <= 0:
        return 1
    return max(1, int(np.ceil(n_points / float(n_markers))))

def _scale_for_steps(x_max: float) -> Tuple[float, str]:
    if x_max >= 1e6:
        return 1e6, 'Step (M)'
    if x_max >= 1e3:
        return 1e3, 'Step (k)'
    return 1.0, 'Step'

# add function: choose Y_keys based on env. For walker2d, use avg_r
def choose_y_keys(env: str) -> List[str]:
    if env == 'walker2d' or env == 'halfcheetah-medium' or env == 'hopper-medium':
        return 'avg_r'
    else:
        return 'avg_ep_found_goal'

env_steprange = {
    'antgoal': (0, 5000000),
    'customhandmanipulateblockrotatez': (0, 3000000),
    'fetchpickandplacediffholdout': (0, 10000000),
    'fetchpushenvcustom': (0, 5000000),
    'maze2d-medium': (0, 5000000),
    'walker2d': (0, 8000000),
    'halfcheetah-medium': (0, 25000000),
    'hopper-medium': (0, 25000000),
}

env_rename={
    'halfcheetah-medium': 'Halfcheetah',
    'hopper-medium': 'Hopper',
    'walker2d': 'Walker2d',
    'antgoal': 'Ant-goal',
    'customhandmanipulateblockrotatez': 'Hand-rotate',
    'fetchpickandplacediffholdout': 'Fetch-pick',
    'maze2d-medium': 'Maze2d',
} #please use it

algo_rename={
    'drail': 'DRAIL',
    'fmail': 'FM-IRL (Ours)',
    'gail': 'GAIL',
    'wail': 'WAIL',
    'airl': 'AIRL',
    'vail': 'VAIL',
} # please use it

# Canonical legend order: FM-IRL first; DP/FP last
LEGEND_ORDER = ['fmail', 'drail', 'gail', 'wail', 'vail', 'airl', 'dp', 'fp']
def scan_experiments(root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Scan data/log for env_algo/seedK/metrics.csv structure.
    Returns: mapping env -> { algo: [seed_dir_paths, ...] }
    """
    env_to_algo: Dict[str, Dict[str, List[Path]]] = {}
    if not root.exists():
        return env_to_algo
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        if "_" not in name:
            continue
        env, algo = name.rsplit("_", 1)
        seed_dirs = [p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed")]
        seed_dirs = [p for p in seed_dirs if (p / "metrics.csv").exists()]
        if len(seed_dirs) == 0:
            continue
        env_to_algo.setdefault(env, {}).setdefault(algo, []).extend(seed_dirs)
    return env_to_algo

def build_from_config(root: Path, cfg: Dict) -> Dict[str, Dict[str, List[Path]]]:
    """
    Build env->algo->seed_dirs mapping from a user dictionary.
    Dict shape:
      {
        'seed': [1,2,3],
        'env_to_algos': {
            'antgoal': ['fmail','drail'],
            ...
        }
      }
    Directory pattern: data/log/{env}_{algo}/seedK/metrics.csv
    """
    env_to_algo: Dict[str, Dict[str, List[Path]]] = {}
    seeds: List[int] = cfg.get('seed', [])
    env_map: Dict[str, List[str]] = cfg.get('env_to_algos', {})
    for env, algos in env_map.items():
        for algo in algos:
            seed_dirs: List[Path] = []
            for s in seeds:
                sd = root / f"{env}_{algo}" / f"seed{s}"
                # Debug print to help user understand why fmail may be missing
                if not (sd / 'metrics.csv').exists():
                    print(f"[WARNING] Missing: {(sd / 'metrics.csv')}")
                if (sd / 'metrics.csv').exists():
                    seed_dirs.append(sd)
            if len(seed_dirs) > 0:
                env_to_algo.setdefault(env, {}).setdefault(algo, []).extend(seed_dirs)
            else:
                print(f"[INFO] No valid runs found for {env}_{algo} (seeds: {seeds})")
    return env_to_algo

def load_run(csv_path: Path) -> Optional[pd.DataFrame]:
    # Robust CSV loader with fallbacks and helpful warnings
    # 1) Try default fast path
    try:
        df = pd.read_csv(csv_path)
        if "step" in df.columns:
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df = df.sort_values("step")
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        return df
    except Exception as e1:
        print(f"[WARN] pd.read_csv failed for {csv_path}: {e1}")
        # 2) Prefer manual salvage using csv.reader to preserve rows with extra commas
        try:
            import csv
            with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                print(f"[ERROR] Empty CSV: {csv_path}")
                return None
            header = rows[0]
            expected_fields = len(header)
            fixed_rows = []
            skipped = 0
            for r in rows[1:]:
                if len(r) == expected_fields:
                    fixed_rows.append(r)
                elif len(r) > expected_fields:
                    # If extras are only trailing empties, trim them; otherwise skip
                    rr = r[:]
                    # Remove trailing empty strings
                    while len(rr) > expected_fields and (rr[-1] is None or rr[-1] == '' or rr[-1].isspace()):
                        rr.pop()
                    if len(rr) == expected_fields:
                        fixed_rows.append(rr)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            df = pd.DataFrame(fixed_rows, columns=header)
            if "step" in df.columns:
                df['step'] = pd.to_numeric(df['step'], errors='coerce')
                df = df.sort_values("step")
            elif "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            print(f"[WARN] Loaded after line-level skip; skipped {skipped} malformed lines: {csv_path}")
            return df
        except Exception as e2:
            # 3) Try python engine with on_bad_lines=skip (tolerant)
            try:
                df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
                if "step" in df.columns:
                    df['step'] = pd.to_numeric(df['step'], errors='coerce')
                    df = df.sort_values("step")
                elif "timestamp" in df.columns:
                    df = df.sort_values("timestamp")
                print(f"[WARN] Loaded with engine='python', on_bad_lines='skip': {csv_path}")
                return df
            except Exception as e3:
                # 4) Older pandas compatibility
                try:
                    df = pd.read_csv(csv_path, engine='python', error_bad_lines=False, warn_bad_lines=True)
                    if "step" in df.columns:
                        df['step'] = pd.to_numeric(df['step'], errors='coerce')
                        df = df.sort_values("step")
                    elif "timestamp" in df.columns:
                        df = df.sort_values("timestamp")
                    print(f"[WARN] Loaded with error_bad_lines=False: {csv_path}")
                    return df
                except Exception as e4:
                    print(f"[ERROR] Failed to load {csv_path} after all fallbacks: {e4}")
                    return None

def load_run_halfcheetah(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Special loader for halfcheetah metrics with irregular extra last value per row as avg_r
    and occasional trailing commas. Keeps the first 6 header columns intact and appends
    an 'avg_r' column extracted as the last non-empty numeric value beyond the expected fields.
    Rows without extra values keep avg_r as NaN.
    """
    import csv
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        print(f"[ERROR] Cannot read CSV for halfcheetah: {csv_path}: {e}")
        return None

    if not rows:
        print(f"[ERROR] Empty CSV: {csv_path}")
        return None

    header = rows[0]
    # Expect original 6 columns
    expected_fields = 6 if len(header) >= 6 else len(header)
    base_header = header[:expected_fields]
    header_out = base_header + ["avg_r"]

    def parse_last_extra_numeric(extra_fields: List[str]) -> Optional[float]:
        for token in reversed(extra_fields):
            if token is None:
                continue
            t = token.strip()
            if t == "":
                continue
            try:
                return float(t)
            except Exception:
                continue
        return None

    fixed_rows: List[List[Optional[str]]] = []
    for r in rows[1:]:
        # Ensure we have at least expected_fields columns; pad with empties if shorter
        base = (r + [""] * max(0, expected_fields - len(r)))[:expected_fields]
        extra = r[expected_fields:] if len(r) > expected_fields else []
        avg_r_val = parse_last_extra_numeric(extra)
        row_out = base + [avg_r_val if avg_r_val is not None else np.nan]
        fixed_rows.append(row_out)

    try:
        df = pd.DataFrame(fixed_rows, columns=header_out)
        if "step" in df.columns:
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df = df.sort_values("step")
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        # Coerce avg_r to numeric explicitly
        if "avg_r" in df.columns:
            df['avg_r'] = pd.to_numeric(df['avg_r'], errors='coerce')
        return df
    except Exception as e:
        print(f"[ERROR] Failed to build DataFrame for halfcheetah: {csv_path}: {e}")
        return None

def load_run_for_env(env: str, csv_path: Path) -> Optional[pd.DataFrame]:
    """Dispatch to a special loader for certain environments, default to generic loader."""
    if env == 'halfcheetah-medium':
        return load_run_halfcheetah(csv_path)
    return load_run(csv_path)

def interp_to_grid(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    ux, idx = np.unique(x_sorted, return_index=True)
    uy = y_sorted[idx]
    return np.interp(x_grid, ux, uy, left=np.nan, right=np.nan)

def smooth_series(y: np.ndarray, sigma: float, window_size: int = None) -> np.ndarray:
    """
    Enhanced Nan-aware smoothing with two smoothing methods:
    1. Gaussian filter (when sigma > 0)
    2. Moving average (when window_size > 0)
    Does not extrapolate beyond first/last valid point.
    """
    if sigma <= 0 and (window_size is None or window_size <= 1):
        return y
        
    y = y.astype(float, copy=True)
    mask = np.isfinite(y)
    if not np.any(mask):
        return y
        
    # First apply Gaussian smoothing if sigma > 0
    if sigma > 0:
        values = np.where(mask, y, 0.0)
        weights = mask.astype(float)
        sm_values = gaussian_filter1d(values, sigma=sigma, mode="nearest")
        sm_weights = gaussian_filter1d(weights, sigma=sigma, mode="nearest")
        valid = sm_weights > 1e-8
        y[valid] = sm_values[valid] / sm_weights[valid]
    
    # Then apply moving average if window_size > 1
    if window_size and window_size > 1:
        # Weighted moving average that preserves length (uses 'same' mode)
        kernel = np.ones(int(window_size), dtype=float)
        kernel /= kernel.sum()
        values = np.where(np.isnan(y), 0.0, y)
        weights = np.where(np.isnan(y), 0.0, 1.0)
        smooth_vals = np.convolve(values, kernel, mode='same')
        smooth_wgts = np.convolve(weights, kernel, mode='same')
        with np.errstate(invalid='ignore', divide='ignore'):
            smoothed = np.divide(smooth_vals, smooth_wgts, out=np.full_like(y, np.nan), where=smooth_wgts > 1e-12)
        y = smoothed
    
    return y

def select_y_key(dfs: List[pd.DataFrame]) -> Optional[str]:
    preferred = ["avg_ep_found_goal", "avg_r"]
    for key in preferred:
        if all(key in df.columns for df in dfs):
            return key
    # fallback: first numeric column excluding x
    for df in dfs:
        for col in df.columns:
            if col in ("step", "timestamp"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None

def plot_env(env: str, algo_to_runs: Dict[str, List[Path]], out_dir: Path, x_key: str = "step", smooth_sigma: float = 2.0, smooth_window: int = None, dpi: int = 300, x_max_override: Optional[float] = None):
    # Get environment specific step range
    x_min, x_max_env = env_steprange.get(env, (0, None))
    
    # Style
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })

    # Load all data
    algo_series: Dict[str, List[pd.DataFrame]] = {}
    for algo, seed_dirs in algo_to_runs.items():
        dfs: List[pd.DataFrame] = []
        for sd in seed_dirs:
            csv_fp = sd / "metrics.csv"
            if not csv_fp.exists():
                print(f"[WARN] Missing file: {csv_fp}")
                continue
            df = load_run_for_env(env, csv_fp)
            if df is not None:
                dfs.append(df)
        if len(dfs) > 0:
            algo_series[algo] = dfs
        else:
            print(f"[INFO] No valid DataFrames for {env}_{algo}")

    if len(algo_series) == 0:
        print(f"[INFO] No data to plot for {env}")
        return

    # Determine y key
    # Use the first algo's dfs to select a common y key (prefer success, else return)
    some_dfs = next(iter(algo_series.values()))
    y_key = select_y_key(some_dfs)
    y_key = choose_y_keys(env)
    if y_key is None:
        print(f"[INFO] No valid y_key found for {env}")
        return

    # Build common x grid across all runs
    max_list = []
    for dfs in algo_series.values():
        for df in dfs:
            if x_key in df.columns and y_key in df.columns:
                try:
                    max_list.append(float(np.nanmax(df[x_key].values)))
                except Exception:
                    pass
    if len(max_list) == 0:
        print(f"[INFO] No valid x range for {env}")
        return
    # Use user override if provided, else use env_steprange or max across runs
    if x_max_override is not None:
        x_max = float(x_max_override)
    elif x_max_env is not None:
        x_max = float(x_max_env)
    else:
        x_max = float(np.nanmax(max_list))
    
    if not np.isfinite(x_max) or x_max <= 0:
        print(f"[INFO] x_max not finite for {env}")
        return
        
    # Set x axis minimum from env_steprange
    x_min = float(x_min)
    # Cap points for performance while spanning full horizon
    n_pts = 2000
    x_grid = np.linspace(x_min, x_max, num=n_pts)
    # Ensure unique and sorted grid to avoid interpolation issues
    x_grid = np.unique(x_grid)

    # Colors per algo with FM-IRL forced to orange and swap previous-orange algo with FM-IRL's old color
    algos = sorted(algo_series.keys())
    base_colors = sns.color_palette("tab10", n_colors=max(3, len(algos)))
    palette = {a: c for a, c in zip(algos, base_colors)}
    try:
        fm_key = 'fmail'
        tab10_base = sns.color_palette("tab10")
        orange = tab10_base[1]
        if fm_key in palette:
            fmail_prev = palette[fm_key]
            orange_algo = next((a for a, c in palette.items() if c == orange), None)
            palette[fm_key] = orange
            if orange_algo is not None and orange_algo != fm_key:
                palette[orange_algo] = fmail_prev
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

    # Decide x scaling and label once using x_max
    scale, x_label = _scale_for_steps(x_max)
    x_disp_grid = x_grid / scale

    for algo in algos:
        dfs = algo_series[algo]
        series = []
        for df in dfs:
            if x_key not in df.columns or y_key not in df.columns:
                continue
            # Coerce numeric and drop rows where x or y is NaN
            xv = pd.to_numeric(df[x_key], errors='coerce').to_numpy()
            yv = pd.to_numeric(df[y_key], errors='coerce').to_numpy()
            valid = np.isfinite(xv) & np.isfinite(yv)
            xv = xv[valid]
            yv = yv[valid]
            # Clip to expected metric range (prevents extreme outliers skewing the plot)
            if y_key in Y_RANGES and yv.size > 0:
                low, high = Y_RANGES[y_key]
                before = yv.copy()
                yv = np.clip(yv, low, high)
                if np.any(yv != before):
                    num_clipped = int(np.sum((before < low) | (before > high)))
                    print(f"[WARN] Clipped {num_clipped} values for {env}_{algo}:{y_key} outside [{low}, {high}]")
            if xv.size == 0 or yv.size == 0:
                continue
            y_grid = interp_to_grid(xv, yv, x_grid)
            y_grid = smooth_series(y_grid, smooth_sigma, smooth_window)
            series.append(y_grid)
        if len(series) == 0:
            print(f"[INFO] No valid series for {env}_{algo}")
            continue
        Y = np.vstack(series)
        mean = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0)
        n = np.sum(~np.isnan(Y), axis=0)
        sem = np.divide(std, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 0)
        lower = mean - 1.96 * sem
        upper = mean + 1.96 * sem

        marker = MARKER_MAP.get(algo, None)
        markevery = _compute_markevery(len(x_disp_grid))
        ax.plot(x_disp_grid, mean, label=algo_rename.get(algo, algo.upper()), color=palette[algo], marker=marker, markevery=markevery, markersize=7, linewidth=2.0)
        ax.fill_between(x_disp_grid, lower, upper, color=palette[algo], alpha=0.2)

    y_label_map = {"avg_ep_found_goal": "Success Rate", "avg_r": "Average Return"}
    ax.set_title(f"{env_rename[env]}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label_map.get(y_key, y_key))
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.minorticks_on()
    # Fix axis limits for known bounded metrics to avoid scale crush from outliers
    if y_key in Y_RANGES:
        ax.set_ylim(Y_RANGES[y_key])
    
    # Always ensure y=0 is slightly above the bottom of the axis so flat zero lines are visible
    y0, y1 = ax.get_ylim()
    if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
        # Only adjust if 0 is within the current range
        if y0 >= 0 or (y0 < 0 <= y1):
            delta = 0.03 * (y1 - y0)
            # If the lower bound is at/above 0, drop it below by a small delta
            if y0 >= 0:
                ax.set_ylim(y0 - delta, y1)
            else:
                # If 0 is close to the lower bound, ensure at least delta below 0
                if (0 - y0) < delta:
                    ax.set_ylim(0 - delta, y1)
    
    # Set x axis limits based on env_steprange
    ax.set_xlim(x_min / scale, x_max / scale)

    # Add static baselines and legend
    used_static = add_static_baselines(ax, env, x_min / scale, x_max / scale, y_key)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        try:
            inv_algo_rename = {v: k for k, v in algo_rename.items()}
            # Map displayed labels back to canonical keys
            label_keys = []
            for lab in labels:
                if lab in inv_algo_rename:
                    label_keys.append(inv_algo_rename[lab])
                elif lab.upper() in ("DP", "FP"):
                    label_keys.append(lab.lower())
                else:
                    label_keys.append(lab.lower())
            # Build order by LEGEND_ORDER, then any others
            ordered_keys = [k for k in LEGEND_ORDER if k in label_keys] + [k for k in label_keys if k not in LEGEND_ORDER]
            order = [label_keys.index(k) for k in ordered_keys]
            ax.legend(
                [handles[i] for i in order],
                [labels[i] for i in order],
                loc="upper left",
                ncol=len(order),
                fontsize=16,
            )
        except Exception:
            ax.legend(loc="upper left")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{env}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}")
    plt.close(fig)

def plot_env_on_axis(env: str, algo_to_runs: Dict[str, List[Path]], ax: plt.Axes, x_key: str = "step", smooth_sigma: float = 2.0, smooth_window: int = None, x_max_override: Optional[float] = None, palette: Dict[str, tuple] = None, final_metrics: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None):
    """
    Plot a single environment on a provided axis (no file saving). Shares logic with plot_env.
    """
    x_min, x_max_env = env_steprange.get(env, (0, None))

    # Load data
    algo_series: Dict[str, List[pd.DataFrame]] = {}
    for algo, seed_dirs in algo_to_runs.items():
        dfs: List[pd.DataFrame] = []
        for sd in seed_dirs:
            csv_fp = sd / "metrics.csv"
            if not csv_fp.exists():
                continue
            df = load_run_for_env(env, csv_fp)
            if df is not None:
                dfs.append(df)
        if len(dfs) > 0:
            algo_series[algo] = dfs

    if len(algo_series) == 0:
        ax.set_visible(False)
        return set()

    some_dfs = next(iter(algo_series.values()))
    y_key = select_y_key(some_dfs)
    y_key = choose_y_keys(env)
    if y_key is None:
        ax.set_visible(False)
        return set()

    # Determine x grid
    max_list = []
    for dfs in algo_series.values():
        for df in dfs:
            if x_key in df.columns and y_key in df.columns:
                try:
                    max_list.append(float(np.nanmax(df[x_key].values)))
                except Exception:
                    pass
    if len(max_list) == 0:
        ax.set_visible(False)
        return set()

    if x_max_override is not None:
        x_max = float(x_max_override)
    elif x_max_env is not None:
        x_max = float(x_max_env)
    else:
        x_max = float(np.nanmax(max_list))

    if not np.isfinite(x_max) or x_max <= 0:
        ax.set_visible(False)
        return set()

    x_min = float(x_min)
    n_pts = 2000
    x_grid = np.linspace(x_min, x_max, num=n_pts)
    x_grid = np.unique(x_grid)

    algos = sorted(algo_series.keys())
    if palette is None:
        palette = {a: c for a, c in zip(algos, sns.color_palette("tab10", n_colors=max(3, len(algos))))}

    # Decide x scaling and label once using x_max
    scale, x_label = _scale_for_steps(x_max)
    x_disp_grid = x_grid / scale

    used_algos = set()
    for algo in algos:
        dfs = algo_series[algo]
        series = []
        for df in dfs:
            if x_key not in df.columns or y_key not in df.columns:
                continue
            xv = pd.to_numeric(df[x_key], errors='coerce').to_numpy()
            yv = pd.to_numeric(df[y_key], errors='coerce').to_numpy()
            valid = np.isfinite(xv) & np.isfinite(yv)
            xv = xv[valid]
            yv = yv[valid]
            if y_key in Y_RANGES and yv.size > 0:
                low, high = Y_RANGES[y_key]
                yv = np.clip(yv, low, high)
            if xv.size == 0 or yv.size == 0:
                continue
            y_grid = interp_to_grid(xv, yv, x_grid)
            y_grid = smooth_series(y_grid, smooth_sigma, smooth_window)
            series.append(y_grid)
        if len(series) == 0:
            continue
        used_algos.add(algo)
        Y = np.vstack(series)
        mean = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0)
        n = np.sum(~np.isnan(Y), axis=0)
        sem = np.divide(std, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 0)
        lower = mean - 1.96 * sem
        upper = mean + 1.96 * sem

        marker = MARKER_MAP.get(algo, None)
        markevery = _compute_markevery(len(x_disp_grid))
        ax.plot(x_disp_grid, mean, label=algo_rename.get(algo, algo.upper()), color=palette[algo], marker=marker, markevery=markevery, markersize=7, linewidth=2.0)
        ax.fill_between(x_disp_grid, lower, upper, color=palette[algo], alpha=0.2)

        # Collect final statistics at the last finite point (mean, std, sem)
        if final_metrics is not None:
            finite_idx = np.where(np.isfinite(mean))[0]
            if finite_idx.size > 0:
                li = int(finite_idx[-1])
                final_metrics.setdefault(env, {})[algo] = {
                    "final_y": float(mean[li]),
                    "final_std": float(std[li]),
                    "final_sem": float(sem[li]),
                }
            else:
                final_metrics.setdefault(env, {})[algo] = {
                    "final_y": float('nan'),
                    "final_std": float('nan'),
                    "final_sem": float('nan'),
                }

    y_label_map = {"avg_ep_found_goal": "Success Rate", "avg_r": "Average Return"}
    ax.set_title(env_rename.get(env, env))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label_map.get(y_key, y_key))
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.minorticks_on()
    if y_key in Y_RANGES:
        ax.set_ylim(Y_RANGES[y_key])

    ax.set_xlim(x_min / scale, x_max / scale)

    # Apply zero-baseline margin
    y0, y1 = ax.get_ylim()
    if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
        if y0 >= 0 or (y0 < 0 <= y1):
            delta = 0.03 * (y1 - y0)
            if y0 >= 0:
                ax.set_ylim(y0 - delta, y1)
            else:
                if (0 - y0) < delta:
                    ax.set_ylim(0 - delta, y1)

    # Add static baselines to subplot and return union of used algos and static names (for legend)
    used_statics = add_static_baselines(ax, env, x_min / scale, x_max / scale, y_key, palette)
    # Also record static baselines' mean/std/sem if collecting metrics
    if final_metrics is not None and env in static_method_y_values:
        for name, val in static_method_y_values[env].items():
            if name in static_methods:
                mean_val = np.nan
                std_val = np.nan
                sem_val = np.nan
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = pd.to_numeric(pd.Series(list(val)), errors='coerce').to_numpy(dtype=float)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        mean_val = float(np.nanmean(finite))
                        std_val = float(np.nanstd(finite))
                        n = int(np.sum(np.isfinite(finite)))
                        sem_val = float(std_val / np.sqrt(max(n, 1)))
                else:
                    try:
                        mean_val = float(val)
                    except Exception:
                        pass
                final_metrics.setdefault(env, {})[name] = {
                    "final_y": mean_val,
                    "final_std": std_val,
                    "final_sem": sem_val,
                }
    return used_algos.union(used_statics)

def plot_all_envs(env_to_algo: Dict[str, Dict[str, List[Path]]], out_dir: Path, x_key: str = "step", smooth_sigma: float = 2.0, smooth_window: int = None, dpi: int = 300, x_max_override: Optional[float] = None):
    """
    Create a single figure containing subplots for all environments with a shared legend at the bottom.
    """
    envs = sorted(env_to_algo.keys())
    if len(envs) == 0:
        return

    # Build global algo set
    all_algos = set()
    for algo_runs in env_to_algo.values():
        all_algos.update(algo_runs.keys())
    all_algos = sorted(all_algos)
    # Extend global palette with static baselines to keep colors consistent
    static_names = sorted({k for env in static_method_y_values for k in static_method_y_values[env].keys()})
    base_colors = sns.color_palette("tab10", n_colors=max(3, len(all_algos)))
    palette = {a: c for a, c in zip(all_algos, base_colors)}
    # Enforce FM-IRL orange and swap the previously-orange algo color with FM-IRL's old color
    try:
        fm_key = 'fmail'
        tab10_base = sns.color_palette("tab10")
        orange = tab10_base[1]
        if fm_key in palette:
            fmail_prev = palette[fm_key]
            orange_algo = next((a for a, c in palette.items() if c == orange), None)
            palette[fm_key] = orange
            if orange_algo is not None and orange_algo != fm_key:
                palette[orange_algo] = fmail_prev
    except Exception:
        pass
    # Ensure static baselines have fixed colors
    for s in static_names:
        if s not in palette:
            palette[s] = STATIC_COLOR_MAP.get(s, 'black')

    # Layout: up to 3 columns
    n_envs = len(envs)
    n_cols = 3 if n_envs >= 3 else n_envs
    n_rows = int(np.ceil(n_envs / n_cols))

    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "legend.frameon": False,
        # Larger fonts for aggregated all_envs figure
        "axes.labelsize": 20,
        "axes.titlesize": 25,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    # Use manual layout so we can reliably reserve bottom space for a shared legend
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.5 * n_rows), constrained_layout=False)
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    used_algos_total = set()
    # Collector for final metrics per env and algo
    final_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for idx, env in enumerate(envs):
        ax = axes_list[idx]
        used_algos = plot_env_on_axis(env, env_to_algo[env], ax, x_key=x_key, smooth_sigma=smooth_sigma, smooth_window=smooth_window, x_max_override=x_max_override, palette=palette, final_metrics=final_metrics)
        used_algos_total.update(used_algos)

    # Hide any unused axes
    for j in range(len(envs), len(axes_list)):
        axes_list[j].set_visible(False)

    # Build shared legend at the bottom using the palette and used algos
    legend_algos = sorted(used_algos_total) if used_algos_total else all_algos
    # Canonical ordering with DP/FP last
    def _order_keys(keys):
        key_set = set(keys)
        ordered = [k for k in LEGEND_ORDER if k in key_set]
        # append any remaining unexpected keys
        ordered += [k for k in keys if k not in ordered]
        return ordered
    legend_algos = _order_keys(legend_algos)
    handles = [
        Line2D(
            [0], [0],
            color=palette[a],
            lw=2,
            linestyle='--' if a in static_names else '-',
            label=(a.upper() if a in static_names else algo_rename.get(a, a.upper()))
        )
        for a in legend_algos
    ]

    # Reserve bottom space for legend and place it centered at the bottom
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(handles),
        fontsize=25
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"all_envs"
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}")
    plt.close(fig)

    # After plotting the aggregated figure, export final metrics to CSV
    try:
        rows = []
        for env in sorted(final_metrics.keys()):
            # Determine metric name used for this environment
            y_key = choose_y_keys(env)
            # Order algorithms in a canonical way for readability
            def _order_keys(keys):
                key_set = set(keys)
                ordered = [k for k in LEGEND_ORDER if k in key_set]
                ordered += [k for k in keys if k not in ordered]
                return ordered
            algos_env = final_metrics.get(env, {})
            ordered_algos = _order_keys(list(algos_env.keys()))
            for algo in ordered_algos:
                stats = algos_env[algo]
                val = stats.get("final_y", float('nan'))
                std_val = stats.get("final_std", float('nan'))
                sem_val = stats.get("final_sem", float('nan'))
                algo_disp = (algo.upper() if algo in static_methods else algo_rename.get(algo, algo.upper()))
                env_disp = env_rename.get(env, env)
                rows.append({
                    "env": env_disp,
                    "env_raw": env,
                    "algo": algo,
                    "algo_display": algo_disp,
                    "y_key": y_key,
                    "final_y": val,
                    "final_std": std_val,
                    "final_sem": sem_val,
                    "ci95_half_width": (1.96 * sem_val if np.isfinite(sem_val) else np.nan),
                })
        if rows:
            df_out = pd.DataFrame(rows)
            csv_path = out_dir / "all_envs_final_metrics.csv"
            df_out.to_csv(csv_path, index=False)
            print(f"[INFO] Saved final metrics CSV to {csv_path}")
        else:
            print("[INFO] No final metrics to save.")
    except Exception as e:
        print(f"[WARN] Failed to write final metrics CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot per-environment results with mean±95% CI across seeds.")
    parser.add_argument("--root", default=str(Path(__file__).parent / "data" / "log"))
    parser.add_argument("--smooth-sigma", type=float, default=SMOOTH_SIGMA, help="Gaussian smoothing sigma (0 to disable)")
    parser.add_argument("--smooth-window", type=int, default=SMOOTH_WINDOW, help="Moving average window size (None or <=1 to disable)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--x-key", default="step")
    parser.add_argument("--x-max", type=float, default=None, help="If set, plot up to this x value; otherwise uses env_steprange or data max")
    parser.add_argument("--auto-scan", action='store_true', help='Scan data/log instead of using plot_config dict')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if args.auto_scan:
        env_to_algo = scan_experiments(root)
    else:
        env_to_algo = build_from_config(root, plot_config)
    if len(env_to_algo) == 0:
        print(f"No experiments found under {root}")
        return

    out_dir = Path(__file__).parent / "figs"
    for env, algo_runs in sorted(env_to_algo.items()):
        print(f"[INFO] Plotting {env} with algos: {list(algo_runs.keys())}")
        plot_env(env, algo_runs, out_dir, x_key=args.x_key, smooth_sigma=args.smooth_sigma, smooth_window=args.smooth_window, dpi=args.dpi, x_max_override=args.x_max)
    # Also create an aggregated figure across all environments with bottom legend
    print("[INFO] Plotting aggregated figure for all environments")
    plot_all_envs(env_to_algo, out_dir, x_key=args.x_key, smooth_sigma=args.smooth_sigma, smooth_window=args.smooth_window, dpi=args.dpi, x_max_override=args.x_max)

if __name__ == "__main__":
    main()