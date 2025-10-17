# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_k_expert_points_on_circle(k: int = 16, radius: float = 1.0):
    angles = np.linspace(0.0, 2.0 * math.pi, num=k, endpoint=False)
    s = (radius * np.cos(angles)).astype(np.float32)
    a = (radius * np.sin(angles)).astype(np.float32)
    return s, a

def sample_clusters_around_anchors(s_anchors: np.ndarray, a_anchors: np.ndarray, points_per_anchor: int = 200, radial_std: float = 0.05, tangential_std: float = 0.05):
    """
    Sample points near each anchor along radial and tangential directions.
    - radial_std controls deviation along radius
    - tangential_std controls deviation along angle direction
    """
    all_s = []
    all_a = []
    for sc, ac in zip(s_anchors, a_anchors):
        r = math.sqrt(sc*sc + ac*ac) + np.random.randn(points_per_anchor).astype(np.float32) * radial_std
        theta0 = math.atan2(ac, sc)
        theta = theta0 + np.random.randn(points_per_anchor).astype(np.float32) * tangential_std
        all_s.append(r * np.cos(theta))
        all_a.append(r * np.sin(theta))
    s = np.concatenate(all_s).astype(np.float32)
    a = np.concatenate(all_a).astype(np.float32)
    return s, a


def main():
    this_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(this_dir, "../expert_datasets")
    os.makedirs(out_dir, exist_ok=True)

    k = 16
    s_anchor, a_anchor = generate_k_expert_points_on_circle(k=k, radius=1.0)
    # Sample clusters around each anchor to increase expert coverage near ring
    points_per_anchor = 200
    s, a = sample_clusters_around_anchors(s_anchor, a_anchor, points_per_anchor=points_per_anchor, radial_std=0.05, tangential_std=0.05)

    obs = torch.from_numpy(s).view(-1, 1)
    actions = torch.from_numpy(a).view(-1, 1)
    # Next-obs simple shift
    next_obs = torch.cat([obs[1:], obs[0:1]], dim=0)
    # Dones: end every anchor cluster
    cluster_len = points_per_anchor
    num_clusters = k
    dones = torch.zeros_like(obs)
    for i in range(num_clusters):
        idx = (i+1)*cluster_len - 1
        if idx < dones.numel():
            dones[idx] = 1.0

    torch.save(
        {"obs": obs, "next_obs": next_obs, "actions": actions, "done": dones},
        os.path.join(out_dir, "circle_k16_clusters.pt"),
    )

    # Visualization: scatter expert points and unit circle
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(s, a, c="red", s=5, alpha=0.6, label="expert")
    theta = np.linspace(0, 2.0 * math.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), c="black", lw=1.5, label="s^2 + a^2 = 1")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("state s")
    ax.set_ylabel("action a")
    ax.legend()
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(this_dir, "../circle.png"))


if __name__ == "__main__":
    main()


