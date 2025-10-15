#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from scipy.ndimage import gaussian_filter1d


DEFAULT_Y_KEYS = ["avg_ep_found_goal", "avg_r"]


def load_run(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Robust CSV loader: tries pandas fast path, then line-level skip for malformed
    rows using csv.reader, then pandas python engine with on_bad_lines=skip.
    Returns a DataFrame or None.
    """
    # 1) Fast path
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e1:
        print(f"[WARN] pd.read_csv failed for {csv_path}: {e1}")
        # 2) Line-level strict skip using csv.reader
        try:
            with open(csv_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if not rows:
                print(f"[ERROR] Empty CSV: {csv_path}")
                return None
            header = rows[0]
            expected = len(header)
            good = []
            skipped = 0
            for r in rows[1:]:
                if len(r) == expected:
                    good.append(r)
                elif len(r) > expected:
                    # Trim trailing empties only
                    rr = r[:]
                    while len(rr) > expected and (rr[-1] is None or rr[-1] == '' or rr[-1].isspace()):
                        rr.pop()
                    if len(rr) == expected:
                        good.append(rr)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            df = pd.DataFrame(good, columns=header)
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


def interp_to_grid(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if x.size < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    ux, idx = np.unique(x_sorted, return_index=True)
    uy = y_sorted[idx]
    return np.interp(x_grid, ux, uy, left=np.nan, right=np.nan)


def smooth_series(y: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return y
    y = y.astype(float, copy=False)
    mask = np.isfinite(y)
    if not np.any(mask):
        return y
    values = np.where(mask, y, 0.0)
    weights = mask.astype(float)
    sm_values = gaussian_filter1d(values, sigma=sigma, mode="nearest")
    sm_weights = gaussian_filter1d(weights, sigma=sigma, mode="nearest")
    out = np.full_like(y, np.nan, dtype=float)
    valid = sm_weights > 1e-8
    out[valid] = sm_values[valid] / sm_weights[valid]
    return out


def select_y_keys(df: pd.DataFrame, preferred: List[str]) -> List[str]:
    out = [k for k in preferred if k in df.columns]
    return out if out else [c for c in df.columns if c not in ("step", "timestamp") and pd.api.types.is_numeric_dtype(df[c])]


def plot_seeds(env: str, algo: str, seeds: List[int], root: Path, out_dir: Path, y_keys: List[str], x_key: str = "step", smooth_sigma: float = 2.0, dpi: int = 300, x_max: Optional[float] = None, show_mean: bool = True):
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

    # Load per-seed
    seed_to_df = {}
    for s in seeds:
        csv_fp = root / f"{env}_{algo}" / f"seed{s}" / "metrics.csv"
        if not csv_fp.exists():
            print(f"[WARN] Missing file: {csv_fp}")
            continue
        df = load_run(csv_fp)
        if df is None:
            continue
        # Sort by x
        if x_key in df.columns:
            df[x_key] = pd.to_numeric(df[x_key], errors='coerce')
            df = df.dropna(subset=[x_key])
            df = df.sort_values(x_key)
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        seed_to_df[s] = df

    if not seed_to_df:
        print(f"[INFO] No data for {env}_{algo}")
        return

    # Determine y keys
    some_df = next(iter(seed_to_df.values()))
    keys = [k for k in y_keys if k in some_df.columns]
    if not keys:
        keys = select_y_keys(some_df, DEFAULT_Y_KEYS)
        if not keys:
            print(f"[INFO] No valid y-keys found in {env}_{algo}")
            return

    # X grid
    xmax_list = []
    for df in seed_to_df.values():
        if x_key in df.columns:
            try:
                xmax_list.append(float(np.nanmax(df[x_key].values)))
            except Exception:
                pass
    if not xmax_list:
        print(f"[INFO] No valid x range for {env}_{algo}")
        return
    XMAX = float(x_max) if x_max is not None else float(np.nanmax(xmax_list))
    if not np.isfinite(XMAX) or XMAX <= 0:
        print(f"[INFO] x_max not finite for {env}_{algo}")
        return
    n_pts = 2000
    x_grid = np.linspace(0, XMAX, num=n_pts)
    x_grid = np.unique(x_grid)

    # Colors per seed
    ordered_seeds = sorted(seed_to_df.keys())
    palette = {s: c for s, c in zip(ordered_seeds, sns.color_palette("tab10", n_colors=max(3, len(ordered_seeds))))}

    fig, axes = plt.subplots(1, len(keys), figsize=(7.5 * len(keys), 4.5), constrained_layout=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, y_key in zip(axes, keys):
        series = []
        # plot per-seed
        for s in ordered_seeds:
            df = seed_to_df[s]
            if y_key not in df.columns or x_key not in df.columns:
                continue
            xv = pd.to_numeric(df[x_key], errors='coerce').to_numpy()
            yv = pd.to_numeric(df[y_key], errors='coerce').to_numpy()
            valid = np.isfinite(xv) & np.isfinite(yv)
            xv = xv[valid]
            yv = yv[valid]
            if xv.size == 0:
                continue
            y_grid = interp_to_grid(xv, yv, x_grid)
            y_grid = smooth_series(y_grid, smooth_sigma)
            series.append(y_grid)
            ax.plot(x_grid, y_grid, label=f"seed{s}", color=palette[s], alpha=0.9, linewidth=1.8)

        # mean ± 95% CI
        if show_mean and len(series) > 1:
            Y = np.vstack(series)
            mean = np.nanmean(Y, axis=0)
            std = np.nanstd(Y, axis=0)
            n = np.sum(~np.isnan(Y), axis=0)
            sem = np.divide(std, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 0)
            lower = mean - 1.96 * sem
            upper = mean + 1.96 * sem
            ax.plot(x_grid, mean, color='black', linewidth=2.4, label='mean')
            ax.fill_between(x_grid, lower, upper, color='black', alpha=0.15, label='95% CI')

        y_label_map = {"avg_ep_found_goal": "Success Rate", "avg_r": "Average Return"}
        ax.set_title(f"{env}-{algo} ({y_key})")
        ax.set_xlabel(x_key.capitalize())
        ax.set_ylabel(y_label_map.get(y_key, y_key))
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.4)
        ax.minorticks_on()

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), bbox_to_anchor=(0.5, 1.03))

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{env}_{algo}_seeds"
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot different seeds for the same (env, algo) with per-seed curves and mean±CI.")
    parser.add_argument("--root", default=str(Path(__file__).parent / "data" / "log"))
    parser.add_argument("--env", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", nargs='+', type=int, required=True)
    parser.add_argument("--y-keys", nargs='+', default=DEFAULT_Y_KEYS)
    parser.add_argument("--x-key", default="step")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--no-mean", action='store_true')
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(__file__).parent / "figs"
    plot_seeds(
        env=args.env,
        algo=args.algo,
        seeds=args.seed,
        root=root,
        out_dir=out_dir,
        y_keys=args.y_keys,
        x_key=args.x_key,
        smooth_sigma=args.smooth_sigma,
        dpi=args.dpi,
        x_max=args.x_max,
        show_mean=(not args.no_mean),
    )


if __name__ == "__main__":
    main()
