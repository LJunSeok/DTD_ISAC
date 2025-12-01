"""
Plot tradeoff curves comparing RL agent with fixed-rho baselines.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.vehicular_isac_env import load_env_from_config


def evaluate_rl_agent(
    env,
    model,
    vec_normalize_path: str = None,
    n_episodes: int = 20,
    seed: int = 100
):
    """Evaluate RL agent and return performance metrics."""
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    episode_rates = []
    episode_crlbs_theta = []
    episode_crlbs_r = []
    
    for ep in range(n_episodes):
        if vec_normalize_path:
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep)
        
        done = False
        truncated = False
        rates = []
        crlbs_theta = []
        crlbs_r = []
        
        while not (done or truncated):
            if vec_normalize_path:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_vec, info = env.step(action)
                done = done_vec[0]
                info = info[0] if isinstance(info, list) else info
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
            
            rates.append(info['R_sum'])
            crlbs_theta.extend(info['crlbs_theta'].tolist())
            crlbs_r.extend(info['crlbs_r'].tolist())
        
        episode_rates.append(np.mean(rates))
        episode_crlbs_theta.append(np.mean(crlbs_theta))
        episode_crlbs_r.append(np.mean(crlbs_r))
    
    return {
        'mean_rate': np.mean(episode_rates),
        'std_rate': np.std(episode_rates),
        'mean_crlb_theta': np.mean(episode_crlbs_theta),
        'std_crlb_theta': np.std(episode_crlbs_theta),
        'mean_crlb_r': np.mean(episode_crlbs_r),
        'std_crlb_r': np.std(episode_crlbs_r)
    }


def plot_tradeoff_curves(
    baseline_csv: str,
    rl_results: dict,
    save_dir: str = 'plots',
    config_name: str = 'default'
):
    """
    Plot tradeoff curves.
    
    Args:
        baseline_csv: Path to baseline results CSV
        rl_results: Dictionary with RL agent results
        save_dir: Directory to save plots
        config_name: Name of configuration for plot title
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline results
    df_baseline = pd.read_csv(baseline_csv)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Rate vs Angle CRLB
    ax1 = axes[0]
    
    # Baseline points
    ax1.scatter(
        df_baseline['mean_crlb_theta'],
        df_baseline['mean_rate'],
        c=df_baseline['rho_fixed'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='black',
        label='Fixed ρ'
    )
    
    # Add rho labels
    for idx, row in df_baseline.iterrows():
        ax1.annotate(
            f"ρ={row['rho_fixed']:.1f}",
            (row['mean_crlb_theta'], row['mean_rate']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.7
        )
    
    # RL agent point
    ax1.scatter(
        rl_results['mean_crlb_theta'],
        rl_results['mean_rate'],
        c='red',
        marker='*',
        s=400,
        edgecolors='black',
        linewidths=2,
        label='RL Agent',
        zorder=10
    )
    
    # Error bars for RL
    ax1.errorbar(
        rl_results['mean_crlb_theta'],
        rl_results['mean_rate'],
        xerr=rl_results['std_crlb_theta'],
        yerr=rl_results['std_rate'],
        fmt='none',
        ecolor='red',
        alpha=0.5,
        capsize=5,
        zorder=9
    )
    
    ax1.set_xlabel('Mean Angle CRLB (rad²)', fontsize=12)
    ax1.set_ylabel('Mean Sum Rate (bits/s/Hz)', fontsize=12)
    ax1.set_title('Rate vs Angle CRLB Tradeoff', fontsize=14)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: Rate vs Range CRLB
    ax2 = axes[1]
    
    # Baseline points
    ax2.scatter(
        df_baseline['mean_crlb_r'],
        df_baseline['mean_rate'],
        c=df_baseline['rho_fixed'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='black',
        label='Fixed ρ'
    )
    
    # Add rho labels
    for idx, row in df_baseline.iterrows():
        ax2.annotate(
            f"ρ={row['rho_fixed']:.1f}",
            (row['mean_crlb_r'], row['mean_rate']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.7
        )
    
    # RL agent point
    ax2.scatter(
        rl_results['mean_crlb_r'],
        rl_results['mean_rate'],
        c='red',
        marker='*',
        s=400,
        edgecolors='black',
        linewidths=2,
        label='RL Agent',
        zorder=10
    )
    
    # Error bars for RL
    ax2.errorbar(
        rl_results['mean_crlb_r'],
        rl_results['mean_rate'],
        xerr=rl_results['std_crlb_r'],
        yerr=rl_results['std_rate'],
        fmt='none',
        ecolor='red',
        alpha=0.5,
        capsize=5,
        zorder=9
    )
    
    ax2.set_xlabel('Mean Range CRLB (m²)', fontsize=12)
    ax2.set_ylabel('Mean Sum Rate (bits/s/Hz)', fontsize=12)
    ax2.set_title('Rate vs Range CRLB Tradeoff', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.suptitle(f'Performance Tradeoffs - {config_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    fig_path = save_path / f'tradeoff_curves_{config_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    
    plt.close()
    
    # Create CRLB comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df_baseline))
    width = 0.35
    
    # Angle CRLB comparison
    baseline_theta = df_baseline['mean_crlb_theta'].values
    rl_theta = np.array([rl_results['mean_crlb_theta']] * len(df_baseline))
    
    bars1 = ax.bar(x - width/2, baseline_theta, width, label='Fixed ρ', alpha=0.7)
    ax.axhline(y=rl_results['mean_crlb_theta'], color='r', linestyle='--', 
               linewidth=2, label='RL Agent')
    
    ax.set_xlabel('Fixed ρ Value', fontsize=12)
    ax.set_ylabel('Mean Angle CRLB (rad²)', fontsize=12)
    ax.set_title('Angle CRLB Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{rho:.1f}" for rho in df_baseline['rho_fixed']])
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    fig_path = save_path / f'crlb_comparison_{config_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot tradeoff curves')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to environment config'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained RL model'
    )
    parser.add_argument(
        '--vec-normalize',
        type=str,
        default=None,
        help='Path to VecNormalize stats'
    )
    parser.add_argument(
        '--baseline-csv',
        type=str,
        required=True,
        help='Path to baseline results CSV'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=20,
        help='Number of episodes to evaluate RL agent'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=100,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Load environment
    env = load_env_from_config(args.config)
    
    # Load model
    model = PPO.load(args.model)
    
    print("Evaluating RL agent...")
    rl_results = evaluate_rl_agent(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_episodes=args.n_episodes,
        seed=args.seed
    )
    
    print("\nRL Agent Results:")
    print(f"  Mean rate: {rl_results['mean_rate']:.3f} ± {rl_results['std_rate']:.3f}")
    print(f"  Mean CRLB theta: {rl_results['mean_crlb_theta']:.2e}")
    print(f"  Mean CRLB range: {rl_results['mean_crlb_r']:.2e}")
    
    config_name = Path(args.config).stem
    
    print("\nGenerating tradeoff plots...")
    plot_tradeoff_curves(
        baseline_csv=args.baseline_csv,
        rl_results=rl_results,
        save_dir=args.save_dir,
        config_name=config_name
    )
    
    print("\nPlots generated successfully!")


if __name__ == '__main__':
    main()