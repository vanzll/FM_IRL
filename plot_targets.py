#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


# ========== 用户在这里配置目标与路径 ==========
# 例子：
# target_paths = {
#     'targetA': '/abs/path/to/expA',
#     'targetB': '/abs/path/to/expB',
# }
target_paths: Dict[str, str] = {'FM-A2C':'/home/wanzl/project/DRAIL/data/log/maze2d-medium_fm-a2c',
                                'FM-PPO':'/home/wanzl/project/DRAIL/data/log/maze2d-medium_fm-ppo',
                                'FPO':'/home/wanzl/project/DRAIL/data/log/maze2d-medium_fpo',
                                #'PPO':'/home/wanzl/project/DRAIL/data/log/maze2d-medium_ppo',
                                'FM-IRL':'/home/wanzl/project/DRAIL/data/log/maze2d-medium_fmail',
                                #'5':'/home/wanzl/project/DRAIL/data/log/customhandmanipulateblockrotatez_fmail_hyper_5',
                                }

x_range = (0,5000000) # only plot the first 2.5M steps
# ========== 画图与数据处理配置（仿照 plot_results.py 风格） ==========
Y_RANGES: Dict[str, Tuple[float, float]] = {
    'avg_ep_found_goal': (0.0, 1.0),
}
DEFAULT_Y_PREF_ORDER: List[str] = ['avg_ep_found_goal', 'avg_r']
SMOOTH_SIGMA: float = 1.5
SMOOTH_WINDOW: Optional[int] = 50


def load_csv_robust(csv_path: Path) -> Optional[pd.DataFrame]:
    """Robust CSV reader similar to plot_results.py with tolerant fallbacks."""
    # 1) Fast path
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e1:
        print(f"[WARN] pd.read_csv failed for {csv_path}: {e1}")
        # 2) Line-level salvage
        try:
            import csv
            with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                print(f"[ERROR] Empty CSV: {csv_path}")
                return None
            header = rows[0]
            expected = len(header)
            fixed: List[List[str]] = []
            skipped = 0
            for r in rows[1:]:
                if len(r) == expected:
                    fixed.append(r)
                elif len(r) > expected:
                    rr = r[:]
                    while len(rr) > expected and (rr[-1] is None or rr[-1] == '' or rr[-1].isspace()):
                        rr.pop()
                    if len(rr) == expected:
                        fixed.append(rr)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            df = pd.DataFrame(fixed, columns=header)
            print(f"[WARN] Loaded after line-level skip; skipped {skipped} malformed lines: {csv_path}")
            return df
        except Exception as e2:
            # 3) Fallback tolerant pandas
            try:
                df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
                print(f"[WARN] Loaded with engine='python', on_bad_lines='skip': {csv_path}")
                return df
            except Exception as e3:
                print(f"[ERROR] Failed to load {csv_path}: {e3}")
                return None


def smooth_series(y: np.ndarray, sigma: float, window_size: Optional[int] = None) -> np.ndarray:
    """Nan-aware Gaussian + optional moving-average smoothing (matches plot_results.py style)."""
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
            y = np.divide(smooth_vals, smooth_wgts, out=np.full_like(y, np.nan), where=smooth_wgts > 1e-12)
    return y


def interp_to_grid(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    ux, idx = np.unique(x_sorted, return_index=True)
    uy = y_sorted[idx]
    return np.interp(x_grid, ux, uy, left=np.nan, right=np.nan)


def select_y_key_across_seeds(dfs: List[pd.DataFrame]) -> Optional[str]:
    """Prefer keys in DEFAULT_Y_PREF_ORDER if present across all seeds, else any numeric column."""
    for key in DEFAULT_Y_PREF_ORDER:
        if all(key in df.columns for df in dfs):
            return key
    # fallback: first numeric column excluding x fields
    for df in dfs:
        for col in df.columns:
            if col in ("step", "timestamp"):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def scan_seed_dirs(root: Path) -> List[Path]:
    """Return subdirectories that contain a metrics.csv file. Accept names like seed1, seed2, ..."""
    if not root.exists():
        return []
    seed_dirs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        # Prefer seed* dirs but accept any dir with metrics.csv
        if (p / 'metrics.csv').exists():
            seed_dirs.append(p)
    return sorted(seed_dirs)


def load_seed_dataframes(seed_dirs: List[Path]) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for sd in seed_dirs:
        csv_fp = sd / 'metrics.csv'
        if not csv_fp.exists():
            continue
        df = load_csv_robust(csv_fp)
        if df is None:
            continue
        # Prefer step if present; else sort by timestamp if available
        if 'step' in df.columns:
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df = df.dropna(subset=['step'])
            df = df.sort_values('step')
        elif 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        dfs.append(df)
    return dfs


def build_x_grid(all_dfs: List[pd.DataFrame], x_key: str = 'step', x_range: Tuple[float, float] = (0.0, float('inf'))) -> Optional[np.ndarray]:
    """Construct a common x grid limited by available data and optional x_range (min,max)."""
    max_list: List[float] = []
    for df in all_dfs:
        if x_key in df.columns:
            try:
                max_list.append(float(np.nanmax(pd.to_numeric(df[x_key], errors='coerce').values)))
            except Exception:
                pass
    if not max_list:
        return None
    data_x_max = float(np.nanmax(max_list))
    if not np.isfinite(data_x_max) or data_x_max <= 0:
        return None
    xmin = float(x_range[0]) if x_range is not None else 0.0
    xmax_cap = float(x_range[1]) if x_range is not None else float('inf')
    if not np.isfinite(xmax_cap):
        xmax_cap = data_x_max
    x_max = min(data_x_max, xmax_cap)
    if x_max <= xmin:
        return None
    n_pts = 2000
    grid = np.linspace(xmin, x_max, num=n_pts)
    return np.unique(grid)


def aggregate_target_series(dfs: List[pd.DataFrame], y_key: str, x_grid: np.ndarray, x_key: str = 'step') -> Optional[np.ndarray]:
    series: List[np.ndarray] = []
    for df in dfs:
        if y_key not in df.columns or x_key not in df.columns:
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
        yg = interp_to_grid(xv, yv, x_grid)
        yg = smooth_series(yg, SMOOTH_SIGMA, SMOOTH_WINDOW)
        series.append(yg)
    if not series:
        return None
    return np.vstack(series)


def plot_targets_comparison(target_to_path: Dict[str, str], out_dir: Path, x_key: str = 'step', x_range: Tuple[float, float] = (0, float('inf'))) -> None:
    # Style (match plot_results.py)
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2.0,
        'legend.frameon': False,
    })

    # Load all seeds for each target
    target_data: Dict[str, List[pd.DataFrame]] = {}
    for tgt, p in target_to_path.items():
        root = Path(p).expanduser().resolve()
        seed_dirs = scan_seed_dirs(root)
        if not seed_dirs:
            print(f"[INFO] No seed dirs found for {tgt} at {root}")
            continue
        dfs = load_seed_dataframes(seed_dirs)
        if dfs:
            target_data[tgt] = dfs

    if not target_data:
        print("[INFO] No valid data to plot.")
        return

    # Decide y_key: prefer one that all targets share across at least one seed each
    # Start from the first target's first df
    some_dfs = next(iter(target_data.values()))
    y_key = select_y_key_across_seeds(some_dfs)
    if y_key is None:
        print("[INFO] No valid y key found.")
        return

    # Build a common x grid from all dataframes (across all targets)
    all_dfs = [df for dfs in target_data.values() for df in dfs]
    x_grid = build_x_grid(all_dfs, x_key=x_key, x_range=x_range)
    if x_grid is None:
        print("[INFO] No valid x range.")
        return

    # Colors per target
    targets = sorted(target_data.keys())
    palette = {t: c for t, c in zip(targets, sns.color_palette('tab10', n_colors=max(3, len(targets))))}

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

    lines_plotted = 0
    for tgt in targets:
        dfs = target_data[tgt]
        Y = aggregate_target_series(dfs, y_key=y_key, x_grid=x_grid, x_key=x_key)
        if Y is None:
            print(f"[INFO] No valid series for {tgt}")
            continue
        mean = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0)
        n = np.sum(~np.isnan(Y), axis=0)
        sem = np.divide(std, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 0)
        lower = mean - 1.96 * sem
        upper = mean + 1.96 * sem

        ax.plot(x_grid, mean, label=tgt, color=palette[tgt], linewidth=2.0)
        ax.fill_between(x_grid, lower, upper, color=palette[tgt], alpha=0.2)
        lines_plotted += 1

    if lines_plotted == 0:
        print("[INFO] Nothing plotted.")
        plt.close(fig)
        return

    y_label_map = {"avg_ep_found_goal": "Success Rate", "avg_r": "Average Return"}
    ax.set_title("Maze2d")
    ax.set_xlabel('Step')
    ax.set_ylabel(y_label_map.get(y_key, y_key))
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
    ax.minorticks_on()

    if y_key in Y_RANGES:
        ax.set_ylim(Y_RANGES[y_key])

    # Ensure small margin below zero when applicable
    y0, y1 = ax.get_ylim()
    if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
        if y0 >= 0 or (y0 < 0 <= y1):
            delta = 0.03 * (y1 - y0)
            if y0 >= 0:
                ax.set_ylim(y0 - delta, y1)
            else:
                if (0 - y0) < delta:
                    ax.set_ylim(0 - delta, y1)

    # Limit x range to the constructed grid
    ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))

    handles, labels = ax.get_legend_handles_labels()
    labels = [label for label in labels]
    ax.legend(handles, labels, loc='upper left', fontsize=15)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(str(out_dir / f"FM-family.{ext}"))
    plt.close(fig)


def main():
    # 简单直接：使用文件顶部的 target_paths 作为输入
    if not target_paths:
        print("请先在文件顶部的 target_paths 中配置 {target: path} 映射！")
        return
    out_dir = Path(__file__).parent / 'figs'
    plot_targets_comparison(target_paths, out_dir=out_dir, x_key='step', x_range=x_range)


if __name__ == '__main__':
    main()


