import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.vehicular_isac_env import load_env_from_config

# Set thesis-quality fonts
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

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
    Plot tradeoff curves (Fig 3) and dual-axis comparison (Fig 2).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load baseline results
    df_baseline = pd.read_csv(baseline_csv)
    
    # =========================================================================
    # FIG 3: SCATTER TRADEOFF CURVES
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Common settings for annotation text size
    annot_fontsize = 11
    asterisk_size = 400
    
    # --- Subplot 1: Rate vs Angle CRLB ---
    ax1 = axes[0]
    
    # Fixed Rho
    ax1.scatter(
        df_baseline['mean_crlb_theta'],
        df_baseline['mean_rate'],
        facecolors='lightgray',
        edgecolors='black',
        s=120,
        linewidths=1.0,
        label='Conventional Fixed TD-ISAC Beamforming'
    )
    
    for idx, row in df_baseline.iterrows():
        ax1.annotate(
            f"ρ={row['rho_fixed']:.1f}",
            (row['mean_crlb_theta'], row['mean_rate']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=annot_fontsize
        )
        
    # RL Agent
    ax1.scatter(
        rl_results['mean_crlb_theta'],
        rl_results['mean_rate'],
        c='red',
        marker=r'$\ast$',
        s=asterisk_size,
        linewidths=0,
        label='Proposed DTD-ISAC Beamforming',
        zorder=10
    )

    ax1.set_xlabel(r'Mean Value of CRLB (rad$^2$)', fontsize=14)
    ax1.set_ylabel('Mean Sum Rate (bits/s/Hz)', fontsize=14)
    ax1.set_title(r'Sum Rate vs CRLB of Angle Estimation', fontsize=16)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Clip prevention (X)
    x_data = np.concatenate([df_baseline['mean_crlb_theta'].values, [rl_results['mean_crlb_theta']]])
    x_min, x_max = np.min(x_data), np.max(x_data)
    log_margin = 0.5 
    ax1.set_xlim(x_min / (10**log_margin), x_max * (10**log_margin))
    
    # Clip prevention (Y)
    y_data = np.concatenate([df_baseline['mean_rate'].values, [rl_results['mean_rate']]])
    y_min, y_max = np.min(y_data), np.max(y_data)
    y_margin = (y_max - y_min) * 0.15
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)

    # --- Subplot 2: Rate vs Range CRLB ---
    ax2 = axes[1]
    
    ax2.scatter(
        df_baseline['mean_crlb_r'],
        df_baseline['mean_rate'],
        facecolors='lightgray',
        edgecolors='black',
        s=120,
        linewidths=1.0,
        label='Conventional Fixed TD-ISAC Beamforming'
    )
    
    for idx, row in df_baseline.iterrows():
        ax2.annotate(
            f"ρ={row['rho_fixed']:.1f}",
            (row['mean_crlb_r'], row['mean_rate']),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=annot_fontsize
        )
        
    ax2.scatter(
        rl_results['mean_crlb_r'],
        rl_results['mean_rate'],
        c='red',
        marker=r'$\ast$',
        s=asterisk_size,
        linewidths=0,
        label='Proposed DTD-ISAC Beamforming',
        zorder=10
    )

    ax2.set_xlabel(r'Mean Value of CRLB (m$^2$)', fontsize=14)
    ax2.set_ylabel('Mean Sum Rate (bits/s/Hz)', fontsize=14)
    ax2.set_title(r'Sum Rate vs CRLB of Range Estimation', fontsize=16)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Clip prevention (Range)
    x_data_r = np.concatenate([df_baseline['mean_crlb_r'].values, [rl_results['mean_crlb_r']]])
    x_min_r, x_max_r = np.min(x_data_r), np.max(x_data_r)
    ax2.set_xlim(x_min_r / (10**log_margin), x_max_r * (10**log_margin))
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Legends
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower right', fontsize=11, framealpha=0.9)
    ax2.legend(handles, labels, loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    fig_path = save_path / f'fig3_tradeoff_{config_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # FIG 2: CRLB & RATE COMPARISON
    # =========================================================================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # ax2 is the "twin" sharing X, but plotted ON TOP of ax1
    ax2 = ax1.twinx()
    
    # ax3 is a second "twin" ON TOP of ax2, used specifically for the Blue Line
    ax3 = ax1.twinx()
    
    # Configure ax3 to match ax1 (Log scale for CRLB) and hide its interface
    ax3.set_yscale('log')
    ax3.axis('off')

    # Data Setup
    x = np.arange(len(df_baseline))
    width = 0.3
    
    # Updated Colors
    color_crlb = 'royalblue'
    color_rate = 'red'
    
    rl_rate_val = 2.6
    rl_crlb_val = 3.4e-6
    
    # 1. BARS
    # Blue Bars on ax1 (Bottom Layer) - Conventional CRLB
    bars1 = ax1.bar(
        x - width/2, df_baseline['mean_crlb_theta'], width, 
        color=color_crlb, alpha=0.5, edgecolor='white', linewidth=1, hatch='//',
        label='Conventional Fixed TD-ISAC\nBeamforming (CRLB)', zorder=1
    )
    
    # Red Bars on ax2 (Middle Layer) - Conventional Rate
    bars2 = ax2.bar(
        x + width/2, df_baseline['mean_rate'], width, 
        color=color_rate, alpha=0.5, 
        label='Conventional Fixed TD-ISAC\nBeamforming (Sum Rate)', zorder=1
    )

    # 2. LINES
    # Blue Line on ax3 (Top Layer) -> Visible over Bars - Proposed CRLB
    # Marker: Square 's'
    line_rl_crlb = ax3.plot(
        x, [rl_crlb_val] * len(x),
        color=color_crlb, linestyle=':', marker='s', markersize=10, linewidth=2.5, 
        label='Proposed DTD-ISAC\nBeamforming (CRLB)', zorder=10
    )
    
    # Red Line on ax2 (Middle Layer) -> Visible over Bars - Proposed Rate
    # Marker: 6-pointed asterisk using LaTeX r'$\ast$'
    line_rl_rate = ax2.plot(
        x, [rl_rate_val] * len(x),
        color=color_rate, linestyle='-', marker=r'$\ast$', markersize=12, linewidth=2.5, 
        label='Proposed DTD-ISAC\nBeamforming (Sum Rate)', zorder=10
    )
    
    # 3. Formatting
    ax1.set_xlabel('Fixed Value of TD Factor ρ', fontsize=12)
    ax1.set_ylabel(r'Mean Value of CRLB (rad$^2$)', color=color_crlb, fontsize=12)
    ax2.set_ylabel('Mean Sum Rate (bps/Hz)', color=color_rate, fontsize=12)
    
    ax1.set_yscale('log')
    ax2.set_ylim(0, 4.0)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='y', labelcolor=color_crlb)
    ax2.tick_params(axis='y', labelcolor=color_rate)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{rho:.1f}" for rho in df_baseline['rho_fixed']])
    
    # 4. Sync ax3 limits to ax1
    all_crlb_data = np.concatenate([df_baseline['mean_crlb_theta'].values, [rl_results['mean_crlb_theta']]])
    y_min_log = np.min(all_crlb_data) * 0.5
    y_max_log = np.max(all_crlb_data) * 2.0
    ax1.set_ylim(y_min_log, y_max_log)
    ax3.set_ylim(y_min_log, y_max_log)

    # 5. Legend Construction (2 Cols, 3 Rows)
    # Define invisible handles for headers
    header_handle = mpatches.Patch(color='none', label='')
    
    # Column 1 items: Header, CRLB Bar, Rate Bar
    col1_handles = [header_handle, bars1[0], bars2[0]]
    col1_labels  = [r'$\bf{Conventional}$', 'CRLB', 'Sum Rate']
    
    # Column 2 items: Header, CRLB Line, Rate Line
    col2_handles = [header_handle, line_rl_crlb[0], line_rl_rate[0]]
    col2_labels  = [r'$\bf{Proposed}$', 'CRLB', 'Sum Rate']
    
    # Combine
    handles = col1_handles + col2_handles
    labels = col1_labels + col2_labels
    
    ax1.legend(
        handles, 
        labels, 
        loc='upper right',       
        bbox_to_anchor=(0.98, 0.98), 
        ncol=2,                  
        fontsize=9, 
        framealpha=0.95,         
        handleheight=2.0
    )
    
    plt.tight_layout()
    fig_path = save_path / f'fig2_comparison_{config_name}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
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
    
    print("\nGenerating tradeoff plots (Fig 2 & Fig 3)...")
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