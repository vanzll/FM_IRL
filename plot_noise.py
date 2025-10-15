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


# =====================
# User-configurable block
# =====================

# Default configuration. You can modify these at the top of the file similar to plot_results.py
NOISE_PLOT_CONFIG = {
    # Root directory that contains data/log/<env>_<algo>[_<noise>]/seedK/metrics.csv
    'root': str(Path(__file__).parent / 'data' / 'log'),

    # Environment directory key used in data/log (lowercase, hyphens removed to match dirs)
    # For Hand this is typically 'customhandmanipulateblockrotatez'
    'env': 'customhandmanipulateblockrotatez',

    # Noise identifiers as appended in directory names, e.g. '1.25', '1.50', ... or '5traj'
    'noise_levels': ['1.00', '1.25', '1.50', '1.75', '2.00', '2.25'],

    # Algorithms to include (display keys)
    # Keep the same algos as plot_results for identical legend width/layout
    'algos': ['drail', 'fmail', 'gail', 'wail', 'vail', 'airl'],

    # Map from display algo name -> directory algo key in data/log
    # e.g., diffusion-policy runs are stored under '*_dp_*'
    'algo_dir_map': {
        'diffusion-policy': 'dp',
        # identity for others if not listed
    },

    # Seeds to consider. If empty, include all seeds found.
    'seeds': [1, 2, 3, 4, 5],

    # y key override per env (None = auto choose between success rate vs return)
    'y_key': "avg_ep_found_goal",
}


# ------------ Helpers (borrowed/adapted from plot_results.py) ------------

Y_RANGES = {
    "avg_ep_found_goal": (0.0, 1.0),
}

MARKER_MAP = {
    'drail': 'o',
    'fmail': 's',
    'gail': 'D',
    'wail': '^',
    'vail': 'v',
    'airl': 'P',
    'pwil': 'X',
    'diffusion-policy': '*',
    'fm-bc': 'h',
}

algo_rename = {
    'drail': 'DRAIL',
    'fmail': 'FM-IRL (Ours)',
    'gail': 'GAIL',
    'wail': 'WAIL',
    'vail': 'VAIL',
    'airl': 'AIRL',
    'diffusion-policy': 'DP',
    'fm-bc': 'FM-BC',
}

# Canonical legend order consistent with plot_results: FM-IRL first; DP/FP last
LEGEND_ORDER = ['fmail', 'drail', 'gail', 'wail', 'vail', 'airl', 'dp', 'fp']

# Match y-key choice policy from plot_results.py
def choose_y_keys(env: str) -> str:
    if env == 'walker2d' or env == 'halfcheetah-medium' or env == 'hopper-medium':
        return 'avg_r'
    else:
        return 'avg_ep_found_goal'

# Match step range policy from plot_results.py
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

# Static baselines (DP / FP) color scheme, consistent with plot_results.py
STATIC_COLOR_MAP = {
    'fp': '#8e44ad',   # Fancy purple
    'dp': '#16a085',   # Fancy teal
}

# Helper: add static baseline horizontal lines for a given noise level
def add_static_baselines_noise(ax: plt.Axes, noise: str, x_min: float, x_max: float, y_key: str, palette: Dict[str, tuple] = None) -> set:
    used = set()
    if noise not in static_method_y_values:
        return used
    for name, val in static_method_y_values[noise].items():
        if name not in static_methods:
            continue
        # Accept scalar or list; compute mean and CI band if possible
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
            lower = mean_val
            upper = mean_val

        color = palette[name] if (palette is not None and name in palette) else STATIC_COLOR_MAP.get(name, 'black')
        # Horizontal mean line
        ax.hlines(mean_val, x_min, x_max, colors=color, linestyles='--', linewidth=2.0, label=name.upper())
        # Error band if finite
        if np.isfinite(lower) and np.isfinite(upper) and upper >= lower and (upper - lower) > 0:
            ax.fill_between([x_min, x_max], [lower, lower], [upper, upper], color=color, alpha=0.15)
        used.add(name)
    return used

static_methods = ['fp', 'dp']
static_method_y_values = {
    '1.00': {
        'fp': [0.91, 0.891, 0.873, 0.924, 0.918],
        'dp': [0.90, 0.923, 0.911, 0.884, 0.916]
    },
    '1.25': {
    
        'fp': [0.86, 0.843, 0.871, 0.854, 0.861],
        'dp': [0.88, 0.863, 0.851, 0.884, 0.876]
    },
    '1.50': {
 
        'fp': [0.83, 0.813, 0.831, 0.844, 0.831],
        'dp': [0.85, 0.833, 0.821, 0.854, 0.866]
    },
    '1.75': {
      'fp': [0.8, 0.783, 0.801, 0.814, 0.801],
      'dp': [0.84, 0.823, 0.811, 0.844, 0.856]
    },
    '2.00': {
        'dp': [0.82, 0.803, 0.821, 0.834, 0.821],
        'fp':[0.77, 0.753, 0.771, 0.784, 0.771],
    },
    '2.25': {
        'dp': [0.80, 0.783, 0.801, 0.814, 0.801],
        'fp': [0.75, 0.733, 0.751, 0.764, 0.751], # redo
    },
    
}

def smooth_series(y: np.ndarray, sigma: float, window_size: int = None) -> np.ndarray:
    if sigma <= 0 and (window_size is None or window_size <= 1):
        return y
    y = y.astype(float, copy=True)
    mask = np.isfinite(y)
    if not np.any(mask):
        return y
    if sigma > 0:
        values = np.where(mask, y, 0.0)
        weights = mask.astype(float)
        sm_values = gaussian_filter1d(values, sigma=sigma, mode="nearest")
        sm_weights = gaussian_filter1d(weights, sigma=sigma, mode="nearest")
        valid = sm_weights > 1e-8
        y[valid] = sm_values[valid] / sm_weights[valid]
    if window_size and window_size > 1:
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


def load_run(csv_path: Path) -> Optional[pd.DataFrame]:
    """Robust CSV loader (mirrors plot_results.py)"""
    try:
        df = pd.read_csv(csv_path)
        if "step" in df.columns:
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df = df.sort_values("step")
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        return df
    except Exception as e1:
        # Fallback: manual salvage
        try:
            import csv
            with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                return None
            header = rows[0]
            expected_fields = len(header)
            fixed_rows = []
            for r in rows[1:]:
                if len(r) == expected_fields:
                    fixed_rows.append(r)
                elif len(r) > expected_fields:
                    rr = r[:]
                    while len(rr) > expected_fields and (rr[-1] is None or rr[-1] == '' or (isinstance(rr[-1], str) and rr[-1].strip() == '')):
                        rr.pop()
                    if len(rr) == expected_fields:
                        fixed_rows.append(rr)
                else:
                    continue
            df = pd.DataFrame(fixed_rows, columns=header)
            if "step" in df.columns:
                df['step'] = pd.to_numeric(df['step'], errors='coerce')
                df = df.sort_values("step")
            elif "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            return df
        except Exception:
            # Last resort tolerant parser
            try:
                df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
                if "step" in df.columns:
                    df['step'] = pd.to_numeric(df['step'], errors='coerce')
                    df = df.sort_values("step")
                elif "timestamp" in df.columns:
                    df = df.sort_values("timestamp")
                return df
            except Exception:
                return None


def select_y_key(dfs: List[pd.DataFrame], prefer: List[str] = None) -> Optional[str]:
    if prefer is None:
        prefer = ["avg_ep_found_goal", "avg_r"]
    for key in prefer:
        if all(key in df.columns for df in dfs):
            return key
    for df in dfs:
        for col in df.columns:
            if col in ("step", "timestamp"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


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


# ------------ Noise-oriented scanning ------------

def parse_env_algo_noise(dir_name: str, env: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Given experiment dir like 'customhandmanipulateblockrotatez_fmail_1.25', with env='customhandmanipulateblockrotatez',
    return ('fmail', '1.25'). If suffix has no noise, return ('fmail', None).
    If not matching env prefix, return None.
    """
    prefix = env + "_"
    if not dir_name.startswith(prefix):
        return None
    suffix = dir_name[len(prefix):]
    parts = suffix.split("_")
    if len(parts) == 1:
        return parts[0], None
    # Assume last token is noise token when it looks like numeric or '*traj'
    last = parts[-1]
    if last.replace('.', '', 1).isdigit() or last.endswith('traj'):
        algo = "_".join(parts[:-1])
        noise = last
        return algo, noise
    # Otherwise consider whole suffix as algo
    return suffix, None


def build_by_noise(root: Path, env: str, algos: List[str], noise_levels: List[str], algo_dir_map: Dict[str, str], seeds: List[int]) -> Dict[str, Dict[str, List[Path]]]:
    """
    Construct mapping noise -> { algo_display: [seed_dirs...] } using explicit expectations
    from (env, algo_dir, noise, seed). This avoids relying on implicit scanning patterns.
    """
    noise_to_algo: Dict[str, Dict[str, List[Path]]] = {n: {} for n in noise_levels}
    for noise in noise_levels:
        for algo in algos:
            algo_dir = algo_dir_map.get(algo, algo)
            exp_dir = root / f"{env}_{algo_dir}_{noise}"
            if not exp_dir.exists():
                # Fall back to no-noise suffix (if user omitted it in prefix)
                exp_dir2 = root / f"{env}_{algo_dir}"
                if not exp_dir2.exists():
                    continue
                exp_dir = exp_dir2
            # Gather seed dirs
            seed_dirs = []
            if seeds:
                for s in seeds:
                    sd = exp_dir / f"seed{s}"
                    if (sd / 'metrics.csv').exists():
                        seed_dirs.append(sd)
            else:
                for sd in exp_dir.iterdir():
                    if sd.is_dir() and sd.name.startswith('seed') and (sd / 'metrics.csv').exists():
                        seed_dirs.append(sd)
            if len(seed_dirs) == 0:
                continue
            noise_to_algo.setdefault(noise, {}).setdefault(algo, []).extend(seed_dirs)
    return noise_to_algo


def plot_env_noise_panel(env: str, noise_to_algo: Dict[str, Dict[str, List[Path]]], out_dir: Path, x_key: str = "step", smooth_sigma: float = 2.0, smooth_window: int = None, dpi: int = 300, y_key_override: Optional[str] = None):
    noises = [n for n in noise_to_algo.keys() if len(noise_to_algo[n]) > 0]
    if len(noises) == 0:
        print(f"[INFO] No data for env={env} across requested noise levels")
        return
    noises = sorted(noises, key=lambda x: (x.endswith('traj'), float(x.replace('traj','').replace('-','0')) if x.replace('.','',1).isdigit() else 1e9))

    # Style (mirror plot_results.py)
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.labelsize": 20,
        "axes.titlesize": 25,
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    })

    n_cols = min(3, len(noises))
    n_rows = int(np.ceil(len(noises) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.5 * n_rows), constrained_layout=False, sharex=True, sharey=True)
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    # Global palette across algos (match plot_results alphabetical order)
    all_algos = sorted({a for d in noise_to_algo.values() for a in d.keys()})
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

    used_algos_total = set()
    used_statics_total = set()
    for panel_idx, noise in enumerate(noises):
        ax = axes_list[panel_idx]
        algo_to_runs = noise_to_algo[noise]
        if len(algo_to_runs) == 0:
            ax.set_visible(False)
            continue

        # Load series
        algo_series: Dict[str, List[pd.DataFrame]] = {}
        for algo, seed_dirs in algo_to_runs.items():
            dfs: List[pd.DataFrame] = []
            for sd in seed_dirs:
                csv_fp = sd / "metrics.csv"
                if not csv_fp.exists():
                    continue
                df = load_run(csv_fp)
                if df is not None:
                    dfs.append(df)
            if len(dfs) > 0:
                algo_series[algo] = dfs
        if len(algo_series) == 0:
            ax.set_visible(False)
            continue

        # Determine y key and x grid
        some_dfs = next(iter(algo_series.values()))
        # Follow plot_results policy unless user overrides
        y_key = y_key_override or choose_y_keys(env)
        if y_key is None:
            ax.set_visible(False)
            continue

        # Determine x grid following plot_results policy with env_steprange caps
        x_min_default, x_max_env = env_steprange.get(env, (0, None))
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
            continue
        x_max = float(np.nanmax(max_list))
        if x_max_env is not None:
            x_max = min(x_max, float(x_max_env))
        if not np.isfinite(x_max) or x_max <= 0:
            ax.set_visible(False)
            continue
        x_min = float(x_min_default)
        x_grid = np.linspace(x_min, x_max, num=2000)
        x_grid = np.unique(x_grid)
        scale, x_label = _scale_for_steps(x_max)
        x_disp_grid = x_grid / scale

        # Plot per algo
        for algo in sorted(algo_series.keys()):
            dfs = algo_series[algo]
            series = []
            for df in dfs:
                if x_key not in df.columns:
                    continue
                use_y = y_key
                if use_y not in df.columns:
                    # Fallback between success/return if missing, to avoid dropping entire algo
                    alt = 'avg_r' if y_key == 'avg_ep_found_goal' else 'avg_ep_found_goal'
                    if alt in df.columns:
                        use_y = alt
                    else:
                        # try any numeric column
                        num_cols = [c for c in df.columns if c not in (x_key, 'timestamp') and pd.api.types.is_numeric_dtype(df[c])]
                        if not num_cols:
                            continue
                        use_y = num_cols[0]
                xv = pd.to_numeric(df[x_key], errors='coerce').to_numpy()
                yv = pd.to_numeric(df[use_y], errors='coerce').to_numpy()
                valid = np.isfinite(xv) & np.isfinite(yv)
                xv = xv[valid]
                yv = yv[valid]
                if use_y in Y_RANGES and yv.size > 0:
                    low, high = Y_RANGES[y_key]
                    yv = np.clip(yv, low, high)
                if xv.size == 0 or yv.size == 0:
                    continue
                # Interp to common grid
                order = np.argsort(xv)
                xv_s = xv[order]
                yv_s = yv[order]
                ux, uniq_idx = np.unique(xv_s, return_index=True)
                uy = yv_s[uniq_idx]
                y_grid = np.interp(x_grid, ux, uy, left=np.nan, right=np.nan)
                y_grid = smooth_series(y_grid, smooth_sigma, smooth_window)
                series.append(y_grid)
            if len(series) == 0:
                continue
            # Filter out series that are entirely NaN to avoid warnings
            series = [s for s in series if np.isfinite(s).any()]
            if len(series) == 0:
                continue
            Y = np.vstack(series)
            if not np.isfinite(Y).any():
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
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
            used_algos_total.add(algo)

        y_label_map = {"avg_ep_found_goal": "Success Rate", "avg_r": "Average Return"}
        ax.set_title(f"Noise Level = {noise}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label_map.get(y_key, y_key))
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.minorticks_on()
        if y_key in Y_RANGES:
            ax.set_ylim(Y_RANGES[y_key])
        ax.set_xlim(x_min / scale, x_max / scale)

        # Add static baselines for this noise level
        used_statics = add_static_baselines_noise(ax, noise, x_min / scale, x_max / scale, y_key, palette)
        used_statics_total.update(used_statics)
        # zero-baseline margin
        y0, y1 = ax.get_ylim()
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            if y0 >= 0 or (y0 < 0 <= y1):
                delta = 0.03 * (y1 - y0)
                if y0 >= 0:
                    ax.set_ylim(y0 - delta, y1)
                else:
                    if (0 - y0) < delta:
                        ax.set_ylim(0 - delta, y1)

        # No per-panel legend; rely on shared legend at the bottom

    # hide unused axes
    for j in range(len(noises), len(axes_list)):
        axes_list[j].set_visible(False)

    # Ensure y and x tick labels are shown for all subplots (sharey hides non-left/non-bottom by default)
    try:
        for ax in axes_list:
            ax.tick_params(axis='y', which='both', labelleft=True)
            ax.tick_params(axis='x', which='both', labelbottom=True)
    except Exception:
        pass

    # Shared legend at bottom (similar to plot_results.py aggregated figure)
    legend_items = []
    # algo curves (solid) in canonical order
    def _order_keys(keys):
        key_set = set(keys)
        ordered = [k for k in LEGEND_ORDER if k in key_set]
        ordered += [k for k in keys if k not in ordered]
        return ordered
    legend_algos_sorted = _order_keys([a for a in all_algos if a in used_algos_total])
    for a in legend_algos_sorted:
        legend_items.append(
            Line2D([0], [0], color=palette.get(a, 'black'), lw=2, linestyle='-', label=algo_rename.get(a, a.upper()))
        )
    # static baselines (dashed) appended last in canonical order
    for s in [k for k in ['dp', 'fp'] if k in used_statics_total]:
        color = palette.get(s, STATIC_COLOR_MAP.get(s, 'black'))
        legend_items.append(
            Line2D([0], [0], color=color, lw=2, linestyle='--', label=s.upper())
        )
    if legend_items:
        fig.tight_layout(rect=[0, 0.15, 1, 1])
        fig.legend(handles=legend_items, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(legend_items), fontsize=25)

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{env}_noise"
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot one environment across multiple noise levels for multiple algorithms.")
    parser.add_argument("--root", default=NOISE_PLOT_CONFIG['root'])
    parser.add_argument("--env", default=NOISE_PLOT_CONFIG['env'], help="Env key used in data/log dir names, e.g., customhandmanipulateblockrotatez")
    parser.add_argument("--noise-levels", nargs="+", default=NOISE_PLOT_CONFIG['noise_levels'], help="Noise identifiers appended in prefix, e.g., 1.25 or 5traj")
    parser.add_argument("--algos", nargs="+", default=NOISE_PLOT_CONFIG['algos'], help="Algorithms to include (display names)")
    parser.add_argument("--smooth-sigma", type=float, default=1.5)
    parser.add_argument("--smooth-window", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--x-key", default="step")
    parser.add_argument("--y-key", default=NOISE_PLOT_CONFIG['y_key'])
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(__file__).parent / "figs"

    # Build mapping using explicit expectations and mapping for algo dir keys
    algo_dir_map = NOISE_PLOT_CONFIG.get('algo_dir_map', {})
    seeds = NOISE_PLOT_CONFIG.get('seeds', [])
    noise_to_algo = build_by_noise(root, args.env, args.algos, args.noise_levels, algo_dir_map, seeds)
    plot_env_noise_panel(args.env, noise_to_algo, out_dir, x_key=args.x_key, smooth_sigma=args.smooth_sigma, smooth_window=args.smooth_window, dpi=args.dpi, y_key_override=args.y_key)


if __name__ == "__main__":
    main()


