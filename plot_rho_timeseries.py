"""
Plot time series of rho values and other metrics for trained RL agents.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.vehicular_isac_env import load_env_from_config


def plot_episode_trajectory(
    env,
    model,
    vec_normalize_path: str = None,
    n_episodes: int = 3,
    save_dir: str = 'plots',
    seed: int = 100
):
    """
    Plot trajectory of rho, rate, and CRLBs over episodes.
    
    Args:
        env: ISAC environment
        model: Trained RL model
        vec_normalize_path: Path to VecNormalize stats
        n_episodes: Number of episodes to plot
        save_dir: Directory to save plots
        seed: Random seed
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Wrap environment if needed
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    for ep in range(n_episodes):
        # Reset environment
        if vec_normalize_path:
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep)
        
        done = False
        truncated = False
        
        # Storage
        rhos = []
        rates = []
        crlbs_theta_mean = []
        crlbs_r_mean = []
        positions_x = []
        
        step = 0
        
        while not (done or truncated):
            # Get action from model
            if vec_normalize_path:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_vec, info = env.step(action)
                done = done_vec[0]
                info = info[0] if isinstance(info, list) else info
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
            
            # Store metrics
            rhos.append(info['rho'])
            rates.append(info['R_sum'])
            crlbs_theta_mean.append(np.mean(info['crlbs_theta']))
            crlbs_r_mean.append(np.mean(info['crlbs_r']))
            positions_x.append(info['positions'][:, 0].mean())
            
            step += 1
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        time_steps = np.arange(len(rhos))
        
        # Plot rho
        axes[0].plot(time_steps, rhos, 'b-', linewidth=2)
        axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='rho=0.5')
        axes[0].set_ylabel('Sensing Fraction (ρ)', fontsize=12)
        axes[0].set_title(f'Episode {ep+1}: Time Division Trajectory', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        
        # Plot sum rate
        axes[1].plot(time_steps, rates, 'g-', linewidth=2)
        axes[1].set_ylabel('Sum Rate (bits/s/Hz)', fontsize=12)
        axes[1].set_title('Communication Performance', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Plot CRLBs (log scale)
        axes[2].semilogy(time_steps, crlbs_theta_mean, 'r-', linewidth=2, label='Angle CRLB')
        axes[2].set_ylabel('CRLB (rad²)', fontsize=12)
        axes[2].set_title('Sensing Performance - Angle', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        axes[3].semilogy(time_steps, crlbs_r_mean, 'orange', linewidth=2, label='Range CRLB')
        axes[3].set_ylabel('CRLB (m²)', fontsize=12)
        axes[3].set_xlabel('Time Step', fontsize=12)
        axes[3].set_title('Sensing Performance - Range', fontsize=12)
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = save_path / f'episode_trajectory_{ep+1}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        
        plt.close()


def plot_rho_histogram(
    env,
    model,
    vec_normalize_path: str = None,
    n_episodes: int = 50,
    save_dir: str = 'plots',
    seed: int = 100
):
    """
    Plot histogram of rho values across multiple episodes.
    
    Args:
        env: ISAC environment
        model: Trained RL model
        vec_normalize_path: Path to VecNormalize stats
        n_episodes: Number of episodes to collect
        save_dir: Directory to save plots
        seed: Random seed
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Wrap environment if needed
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    all_rhos = []
    
    print(f"Collecting rho values from {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        if vec_normalize_path:
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep)
        
        done = False
        truncated = False
        
        while not (done or truncated):
            if vec_normalize_path:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_vec, info = env.step(action)
                done = done_vec[0]
                info = info[0] if isinstance(info, list) else info
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
            
            all_rhos.append(info['rho'])
        
        if (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{n_episodes} episodes")
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_rhos, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=np.mean(all_rhos), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(all_rhos):.3f}')
    ax.axvline(x=np.median(all_rhos), color='g', linestyle='--', linewidth=2,
               label=f'Median: {np.median(all_rhos):.3f}')
    
    ax.set_xlabel('Sensing Fraction (ρ)', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(f'Distribution of ρ Values ({n_episodes} episodes, {len(all_rhos)} steps)', 
                 fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = save_path / 'rho_histogram.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {fig_path}")
    
    # Print statistics
    print("\nRho Statistics:")
    print(f"  Mean: {np.mean(all_rhos):.4f}")
    print(f"  Std: {np.std(all_rhos):.4f}")
    print(f"  Min: {np.min(all_rhos):.4f}")
    print(f"  Max: {np.max(all_rhos):.4f}")
    print(f"  Median: {np.median(all_rhos):.4f}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot RL agent trajectories')
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
        help='Path to trained model'
    )
    parser.add_argument(
        '--vec-normalize',
        type=str,
        default=None,
        help='Path to VecNormalize stats'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=3,
        help='Number of episodes to plot (for trajectories)'
    )
    parser.add_argument(
        '--n-episodes-hist',
        type=int,
        default=50,
        help='Number of episodes for histogram'
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
    
    print("Generating trajectory plots...")
    plot_episode_trajectory(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_episodes=args.n_episodes,
        save_dir=args.save_dir,
        seed=args.seed
    )
    
    print("\nGenerating rho histogram...")
    plot_rho_histogram(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_episodes=args.n_episodes_hist,
        save_dir=args.save_dir,
        seed=args.seed
    )
    
    print("\nAll plots generated successfully!")


if __name__ == '__main__':
    main()