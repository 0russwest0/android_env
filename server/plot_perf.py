import argparse
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_create_total(df: pd.DataFrame, out_dir: str) -> None:
    data = df[df["test_type"] == "create_total"].sort_values("stage_envs")
    if data.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(data["stage_envs"], data["wall_s"], marker="o")
    plt.xlabel("stage_envs")
    plt.ylabel("total create time (s)")
    plt.title("Create total time vs stage_envs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "create_total_wall.png"), dpi=150)
    plt.close()


def plot_step_curves(df: pd.DataFrame, out_dir: str) -> None:
    data = df[df["test_type"] == "step"]
    if data.empty:
        return
    # avg_req_s vs concurrency per stage
    plt.figure(figsize=(8, 5))
    for stage_envs, g in data.groupby("stage_envs"):
        g2 = g.sort_values("concurrency")
        plt.plot(g2["concurrency"], g2["avg_req_s"], marker="o", label=f"stage={stage_envs}")
    plt.xlabel("concurrency")
    plt.ylabel("avg req time (s)")
    plt.title("Step avg req time vs concurrency")
    plt.legend(title="stage_envs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_avg_req.png"), dpi=150)
    plt.close()

    # wall_s vs concurrency per stage (separate)
    plt.figure(figsize=(8, 5))
    for stage_envs, g in data.groupby("stage_envs"):
        g2 = g.sort_values("concurrency")
        plt.plot(g2["concurrency"], g2["wall_s"], marker="o", label=f"stage={stage_envs}")
    plt.xlabel("concurrency")
    plt.ylabel("wall time (s)")
    plt.title("Step wall time vs concurrency")
    plt.legend(title="stage_envs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_wall.png"), dpi=150)
    plt.close()

    # effective = wall/concurrency vs concurrency per stage (separate)
    plt.figure(figsize=(8, 5))
    for stage_envs, g in data.groupby("stage_envs"):
        g2 = g.sort_values("concurrency")
        effective = g2["wall_s"] / g2["concurrency"].replace(0, pd.NA)
        plt.plot(g2["concurrency"], effective, marker="s", linestyle="--", label=f"stage={stage_envs}")
    plt.xlabel("concurrency")
    plt.ylabel("effective time (s)")
    plt.title("Step effective (wall/concurrency) vs concurrency")
    plt.legend(title="stage_envs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_effective.png"), dpi=150)
    plt.close()


def plot_wall_and_effective(df: pd.DataFrame, test_type: str, out_path: str) -> None:
    data = df[df["test_type"] == test_type]
    if data.empty:
        return
    agg = (
        data.groupby("stage_envs", as_index=False)
        .agg({"wall_s": "mean", "concurrency": "mean"})
        .sort_values("stage_envs")
    )
    agg["effective_s"] = agg["wall_s"] / agg["concurrency"].replace(0, pd.NA)

    plt.figure(figsize=(8, 5))
    plt.plot(agg["stage_envs"], agg["wall_s"], marker="o", label="wall")
    plt.plot(agg["stage_envs"], agg["effective_s"], marker="s", label="wall/concurrency")
    plt.xlabel("stage_envs")
    plt.ylabel("time (s)")
    plt.title(f"{test_type.capitalize()} wall vs effective time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot performance results")
    parser.add_argument(
        "--csv",
        type=str,
        default="/root/android_env/server/perf_results_2.csv",
        help="Path to results CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/android_env/server/perf_plots_2",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    plot_create_total(df, args.out_dir)
    plot_step_curves(df, args.out_dir)
    plot_wall_and_effective(df, "reacquire", os.path.join(args.out_dir, "reacquire_wall_effective.png"))
    plot_wall_and_effective(df, "release", os.path.join(args.out_dir, "release_wall_effective.png"))

    print(f"Saved plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


