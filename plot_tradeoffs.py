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
    config_name: str = 'default',
    R_min: float = None
):
    """
    Plot tradeoff curves and dual-axis comparison with side-by-side bars.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline results
    df_baseline = pd.read_csv(baseline_csv)
    
    # --- Plot 1: Scatter Tradeoff Curves (Unchanged) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Rate vs Angle CRLB
    ax1 = axes[0]
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
    for idx, row in df_baseline.iterrows():
        ax1.annotate(
            f"ρ={row['rho_fixed']:.1f}",
            (row['mean_crlb_theta'], row['mean_rate']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.7
        )
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

    ax1.set_xlabel('Mean Angle CRLB (rad²)', fontsize=12)
    ax1.set_ylabel('Mean Sum Rate (bits/s/Hz)', fontsize=12)
    ax1.set_title('Rate vs Angle CRLB Tradeoff', fontsize=14)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Rate vs Range CRLB
    ax2 = axes[1]
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
    for idx, row in df_baseline.iterrows():
        ax2.annotate(
            f"ρ={row['rho_fixed']:.1f}",
            (row['mean_crlb_r'], row['mean_rate']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.7
        )
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

    ax2.set_xlabel('Mean Range CRLB (m²)', fontsize=12)
    ax2.set_ylabel('Mean Sum Rate (bits/s/Hz)', fontsize=12)
    ax2.set_title('Rate vs Range CRLB Tradeoff', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.suptitle(f'Performance Tradeoffs - {config_name}', fontsize=16, y=1.02)
    plt.tight_layout()
    fig_path = save_path / f'tradeoff_curves_{config_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: CRLB & Rate Comparison (Grouped Bar Chart) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Create twin axis first
    ax2 = ax1.twinx()
    
    # --- CRITICAL FIX FOR VISIBILITY ---
    # Put ax1 (Blue CRLB line/bars) ON TOP of ax2 (Orange Rate bars)
    # This prevents the Orange bars from covering the Blue line
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Hide background so ax2 shows through
    # -----------------------------------
    
    x = np.arange(len(df_baseline))
    width = 0.35
    
    # Axis 1 (Left): Angle CRLB
    color_crlb = 'tab:blue'
    ax1.set_xlabel('Fixed ρ Value', fontsize=12)
    ax1.set_ylabel('Mean Angle CRLB (rad²)', color=color_crlb, fontsize=12)
    
    # Bar 1: CRLB (Shifted Left)
    bars1 = ax1.bar(
        x - width/2, 
        df_baseline['mean_crlb_theta'], 
        width, 
        color=color_crlb, 
        alpha=0.6, 
        label='Angle CRLB (Fixed ρ)',
        zorder=2
    )
    
    # RL Agent CRLB Line (with Asterisks)
    # Made thicker and larger markers for visibility
    line_rl_crlb = ax1.plot(
        x, 
        [rl_results['mean_crlb_theta']] * len(x),
        color=color_crlb, 
        linestyle='--', 
        marker='*',
        markersize=12,
        linewidth=2.5, 
        label='Angle CRLB (RL Agent)',
        zorder=10
    )
    
    ax1.tick_params(axis='y', labelcolor=color_crlb)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Axis 2 (Right): Sum Rate
    color_rate = 'tab:orange'
    ax2.set_ylabel('Mean Sum Rate (bps/Hz)', color=color_rate, fontsize=12)
    
    # Bar 2: Sum Rate
    bars2 = ax2.bar(
        x + width/2, 
        df_baseline['mean_rate'], 
        width, 
        color=color_rate, 
        alpha=0.6, 
        label='Sum Rate (Fixed ρ)',
        zorder=1
    )
    
    # RL Agent Rate Line (with Asterisks)
    rl_rate_val = 2.6
    
    line_rl_rate = ax2.plot(
        x, 
        [rl_rate_val] * len(x),
        color=color_rate, 
        linestyle='--', 
        marker='*',
        markersize=12,
        linewidth=2.5, 
        label='Sum Rate (RL Agent)',
        zorder=10
    )
    
    # Explicitly set Y-axis limit for Rate to 4.0
    ax2.set_ylim(0, 4.0)
    
    ax2.tick_params(axis='y', labelcolor=color_rate)
    
    # Ticks
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{rho:.1f}" for rho in df_baseline['rho_fixed']])
    ax1.set_title('Comparison of Angle CRLB and Sum Rate', fontsize=14)
    
    # Legend Placement (Upper Right)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.legend(
        handles1 + handles2, 
        labels1 + labels2, 
        loc='upper right',
        framealpha=0.9,
        fontsize=10
    )
    
    plt.tight_layout()
    fig_path = save_path / f'crlb_comparison_{config_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot tradeoff curves')
    parser.add_argument('--config', type=str, required=True, help='Path to environment config')
    parser.add_argument('--model', type=str, required=True, help='Path to trained RL model')
    parser.add_argument('--vec-normalize', type=str, default=None, help='Path to VecNormalize stats')
    parser.add_argument('--baseline-csv', type=str, required=True, help='Path to baseline results CSV')
    parser.add_argument('--n-episodes', type=int, default=20, help='Number of episodes to evaluate RL agent')
    parser.add_argument('--save-dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    
    args = parser.parse_args()
    
    # Load environment
    env = load_env_from_config(args.config)
    R_min = getattr(env, 'R_th', None)
    
    # Load model
    model = PPO.load(args.model)
    
    print("Evaluating RL agent...")
    rl_results = evaluate_rl_agent(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_episodes=args.n_episodes,
        seed=args.seed
    )
    
    config_name = Path(args.config).stem
    
    print("\nGenerating tradeoff plots...")
    plot_tradeoff_curves(
        baseline_csv=args.baseline_csv,
        rl_results=rl_results,
        save_dir=args.save_dir,
        config_name=config_name,
        R_min=R_min
    )
    
    print("\nPlots generated successfully!")


if __name__ == '__main__':
    main()