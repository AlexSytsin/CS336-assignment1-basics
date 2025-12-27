#!/usr/bin/env python3
"""
Script to plot training and validation metrics from experiment logs.
Usage:
    python plot_experiments.py --experiment_dir checkpoints/experiment1
    python plot_experiments.py --experiments checkpoints/exp1 checkpoints/exp2 --names "Baseline" "Ablation"
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

def load_metrics(experiment_dir):
    """Load metrics.csv from an experiment directory."""
    metrics_path = os.path.join(experiment_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No metrics.csv found in {experiment_dir}")
    
    df = pd.read_csv(metrics_path)
    return df

def plot_single_experiment(experiment_dir, save_dir=None):
    """Plot metrics for a single experiment."""
    df = load_metrics(experiment_dir)
    
    # Extract experiment name from directory
    exp_name = os.path.basename(experiment_dir)
    
    # Create output directory
    if save_dir is None:
        save_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter rows with training and validation data
    train_df = df[df['train_loss'].notna()].copy()
    val_df = df[df['val_loss'].notna()].copy()
    
    # Normalize elapsed time to hours
    if 'elapsed_time' in train_df.columns:
        train_df['elapsed_hours'] = train_df['elapsed_time'] / 3600
        val_df['elapsed_hours'] = val_df['elapsed_time'] / 3600
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics: {exp_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss vs Steps
    ax = axes[0, 0]
    ax.plot(train_df['step'], train_df['train_loss'], label='Train Loss', alpha=0.7, linewidth=2)
    ax.plot(val_df['step'], val_df['val_loss'], label='Val Loss', alpha=0.7, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs Wall Clock Time
    ax = axes[0, 1]
    if 'elapsed_hours' in train_df.columns:
        ax.plot(train_df['elapsed_hours'], train_df['train_loss'], label='Train Loss', alpha=0.7, linewidth=2)
        ax.plot(val_df['elapsed_hours'], val_df['val_loss'], label='Val Loss', alpha=0.7, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time (hours)')
    else:
        ax.plot(train_df['step'], train_df['train_loss'], label='Train Loss', alpha=0.7, linewidth=2)
        ax.plot(val_df['step'], val_df['val_loss'], label='Val Loss', alpha=0.7, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Loss vs Wall Clock Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Perplexity vs Steps
    ax = axes[0, 2]
    ax.plot(train_df['step'], train_df['train_ppl'], label='Train PPL', alpha=0.7, linewidth=2)
    ax.plot(val_df['step'], val_df['val_ppl'], label='Val PPL', alpha=0.7, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate Schedule
    ax = axes[1, 0]
    ax.plot(train_df['step'], train_df['learning_rate'], linewidth=2, color='green')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 5: Gradient Norm
    ax = axes[1, 1]
    ax.plot(train_df['step'], train_df['grad_norm'], label='Grad Norm', alpha=0.7, linewidth=2)
    ax.plot(train_df['step'], train_df['normalized_grad_norm'], label='Clipped Grad Norm', alpha=0.7, linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Throughput (Tokens/sec)
    ax = axes[1, 2]
    ax.plot(train_df['step'], train_df['tokens_per_sec'], linewidth=2, color='purple', alpha=0.7)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Tokens/sec')
    ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f'{exp_name}_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    plt.close()
    
    return plot_path

def plot_comparison(experiment_dirs, names=None, save_path=None):
    """Plot comparison of multiple experiments."""
    if names is None:
        names = [os.path.basename(d) for d in experiment_dirs]
    
    if len(names) != len(experiment_dirs):
        raise ValueError("Number of names must match number of experiment directories")
    
    # Load all metrics
    all_train_dfs = []
    all_val_dfs = []
    
    for exp_dir in experiment_dirs:
        df = load_metrics(exp_dir)
        train_df = df[df['train_loss'].notna()].copy()
        val_df = df[df['val_loss'].notna()].copy()
        
        # Normalize elapsed time
        if 'elapsed_time' in train_df.columns:
            train_df['elapsed_hours'] = train_df['elapsed_time'] / 3600
            val_df['elapsed_hours'] = val_df['elapsed_time'] / 3600
        
        all_train_dfs.append(train_df)
        all_val_dfs.append(val_df)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_dirs)))
    
    # Plot 1: Training Loss vs Steps
    ax = axes[0, 0]
    for i, (train_df, name) in enumerate(zip(all_train_dfs, names)):
        ax.plot(train_df['step'], train_df['train_loss'], label=name, alpha=0.7, linewidth=2, color=colors[i])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss vs Steps
    ax = axes[0, 1]
    for i, (val_df, name) in enumerate(zip(all_val_dfs, names)):
        ax.plot(val_df['step'], val_df['val_loss'], label=name, alpha=0.7, linewidth=2, marker='o', markersize=3, color=colors[i])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss vs Wall Clock Time
    ax = axes[1, 0]
    if 'elapsed_hours' in all_train_dfs[0].columns:
        for i, (train_df, name) in enumerate(zip(all_train_dfs, names)):
            ax.plot(train_df['elapsed_hours'], train_df['train_loss'], label=name, alpha=0.7, linewidth=2, color=colors[i])
        ax.set_xlabel('Time (hours)')
    else:
        for i, (train_df, name) in enumerate(zip(all_train_dfs, names)):
            ax.plot(train_df['step'], train_df['train_loss'], label=name, alpha=0.7, linewidth=2, color=colors[i])
        ax.set_xlabel('Steps')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Wall Clock Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Validation Perplexity vs Steps
    ax = axes[1, 1]
    for i, (val_df, name) in enumerate(zip(all_val_dfs, names)):
        ax.plot(val_df['step'], val_df['val_ppl'], label=name, alpha=0.7, linewidth=2, marker='o', markersize=3, color=colors[i])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title('Validation Perplexity vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = 'experiment_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {save_path}")
    
    plt.close()
    
    return save_path

def print_summary(experiment_dir):
    """Print summary statistics for an experiment."""
    df = load_metrics(experiment_dir)
    exp_name = os.path.basename(experiment_dir)
    
    train_df = df[df['train_loss'].notna()]
    val_df = df[df['val_loss'].notna()]
    
    print(f"\n{'='*60}")
    print(f"Experiment Summary: {exp_name}")
    print(f"{'='*60}")
    
    if len(train_df) > 0:
        print(f"\nTraining:")
        print(f"  Total steps: {train_df['step'].max()}")
        print(f"  Final train loss: {train_df['train_loss'].iloc[-1]:.4f}")
        print(f"  Final train perplexity: {train_df['train_ppl'].iloc[-1]:.2f}")
        print(f"  Best train loss: {train_df['train_loss'].min():.4f}")
        
        if 'elapsed_time' in train_df.columns:
            total_hours = train_df['elapsed_time'].iloc[-1] / 3600
            print(f"  Total training time: {total_hours:.2f} hours")
        
        if 'tokens_per_sec' in train_df.columns:
            avg_throughput = train_df['tokens_per_sec'].mean()
            print(f"  Average throughput: {avg_throughput:.0f} tokens/sec")
    
    if len(val_df) > 0:
        print(f"\nValidation:")
        print(f"  Final val loss: {val_df['val_loss'].iloc[-1]:.4f}")
        print(f"  Final val perplexity: {val_df['val_ppl'].iloc[-1]:.2f}")
        print(f"  Best val loss: {val_df['val_loss'].min():.4f} (step {val_df.loc[val_df['val_loss'].idxmin(), 'step']:.0f})")
        print(f"  Best val perplexity: {val_df['val_ppl'].min():.2f} (step {val_df.loc[val_df['val_ppl'].idxmin(), 'step']:.0f})")
    
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from experiments')
    parser.add_argument('--experiment_dir', type=str, help='Single experiment directory to plot')
    parser.add_argument('--experiments', nargs='+', help='Multiple experiment directories to compare')
    parser.add_argument('--names', nargs='+', help='Names for experiments (for comparison plots)')
    parser.add_argument('--save_dir', type=str, help='Directory to save plots (default: experiment_dir/plots)')
    parser.add_argument('--output', type=str, help='Output path for comparison plot')
    parser.add_argument('--summary', action='store_true', help='Print summary statistics')
    
    args = parser.parse_args()
    
    if args.experiment_dir:
        # Single experiment plotting
        if args.summary:
            print_summary(args.experiment_dir)
        plot_single_experiment(args.experiment_dir, args.save_dir)
        
    elif args.experiments:
        # Multiple experiment comparison
        if args.summary:
            for exp_dir in args.experiments:
                print_summary(exp_dir)
        plot_comparison(args.experiments, args.names, args.output)
        
    else:
        parser.error("Must provide either --experiment_dir or --experiments")

if __name__ == "__main__":
    main()
