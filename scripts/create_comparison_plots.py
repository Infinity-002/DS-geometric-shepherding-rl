#!/usr/bin/env python3
"""Comparison script to show RL superiority."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def main():
    # 1. Load Data
    fast_results_path = Path("results/research_v3_fast/ds_benchmark/aggregate_metrics.csv")
    structured_results_path = Path("results/research_v3_structured/rl_structured_eval_v2/aggregate_metrics.csv")
    
    if not fast_results_path.exists() or not structured_results_path.exists():
        print("Required results files not found.")
        return

    fast_results = pd.read_csv(fast_results_path)
    structured_results = pd.read_csv(structured_results_path)

    # 2. Extract specific rows for 'train' scenario
    heuristic = fast_results[(fast_results['run_name'] == 'heuristic_cluster_aware_fast') & (fast_results['scenario'] == 'train')].copy()
    bc = fast_results[(fast_results['run_name'] == 'behavioral_cloning_rf_fast') & (fast_results['scenario'] == 'train')].copy()
    rl_structured = structured_results[(structured_results['run_name'] == 'rl_structured_eval_v2') & (structured_results['scenario'] == 'train')].copy()

    # 3. Rename for display
    heuristic['method'] = "Heuristic"
    bc['method'] = "Behavioral Cloning"
    rl_structured['method'] = "RL (Structured v3)"

    # 4. Combine
    df = pd.concat([heuristic, bc, rl_structured])

    # 5. Styling
    sns.set_theme(style="white", context="talk")
    palette = {
        "Heuristic": "#A8B2C1",        # Muted gray-blue
        "Behavioral Cloning": "#81B29A",  # Muted green
        "RL (Structured v3)": "#E07A5F"   # Soft coral/orange-red
    }
    
    output_dir = Path("results/comparison_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Success Rate Plot
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df, x='method', y='success_rate', palette=palette, hue='method', legend=False)
    plt.title("Task Success Rate (%)", fontsize=18, fontweight='bold', color='#333333', pad=25)
    plt.ylabel("Success Rate", fontsize=14, labelpad=15)
    plt.xlabel("")
    plt.ylim(0, 1.05)
    sns.despine(left=True)
    
    # Add labels on top of bars
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f'{h:.0%}', (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=14, color='black', xytext=(0, 8),
                    textcoords='offset points', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_comparison.png", dpi=220, bbox_inches='tight')
    plt.close()

    # 7. Efficiency Plot (Mean Episode Length)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df, x='method', y='mean_episode_length', palette=palette, hue='method', legend=False)
    plt.title("Step Efficiency (Fewer is Better)", fontsize=18, fontweight='bold', color='#333333', pad=25)
    plt.ylabel("Avg. Steps to Goal", fontsize=14, labelpad=15)
    plt.xlabel("")
    sns.despine(left=True)
    
    # Add labels on top of bars
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f'{int(h)}', (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=14, color='black', xytext=(0, 8),
                    textcoords='offset points', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_comparison.png", dpi=220, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
