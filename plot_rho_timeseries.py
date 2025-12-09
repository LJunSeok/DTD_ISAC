"""
Plot time series of rho values and other metrics for trained RL agents.
Focus: Single-Episode Visualization (8 Specific Scenarios).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.vehicular_isac_env import load_env_from_config

# --- HELPER: SCENARIO CONFIGURATION ---
def configure_env_for_scenario(env, ep_idx):
    """
    Configures the environment for one of the 8 specific scenarios.
    Returns a descriptive title for the scenario.
    """
    # Reset defaults
    env.mobility_config = {}
    env.x_init_range = [-100.0, -80.0]  # Fly-by mode for all
    
    if ep_idx == 0:
        env.mobility_type = 'constant_velocity'
        env.v_init_range = [10.0, 12.0]
        return "Scenario 1: Const Vel (Low)"
    elif ep_idx == 1:
        env.mobility_type = 'constant_velocity'
        env.v_init_range = [15.0, 17.0]
        return "Scenario 2: Const Vel (Mid)"
    elif ep_idx == 2:
        env.mobility_type = 'constant_velocity'
        env.v_init_range = [20.0, 22.0]
        return "Scenario 3: Const Vel (High)"
        
    elif ep_idx == 3:
        env.mobility_type = 'acceleration'
        env.v_init_range = [10.0, 12.0]
        env.mobility_config = {'ax': 0.5, 'ay': 0.0}
        return "Scenario 4: Const Accel (Low)"
    elif ep_idx == 4:
        env.mobility_type = 'acceleration'
        env.v_init_range = [10.0, 12.0]
        env.mobility_config = {'ax': 1.0, 'ay': 0.0}
        return "Scenario 5: Const Accel (Mid)"
    elif ep_idx == 5:
        env.mobility_type = 'acceleration'
        env.v_init_range = [10.0, 12.0]
        env.mobility_config = {'ax': 2.0, 'ay': 0.0}
        return "Scenario 6: Const Accel (High)"
        
    elif ep_idx == 6:
        env.mobility_type = 'lane_change'
        env.v_init_range = [15.0, 17.0]
        env.mobility_config = {
            'lane_positions': np.array([3.0, 6.0, 9.0]),
            'change_probability': 0.05,
            'change_duration': 2.0
        }
        return "Scenario 7: Lane Change (Seldom)"
    elif ep_idx == 7:
        env.mobility_type = 'lane_change'
        env.v_init_range = [15.0, 17.0]
        env.mobility_config = {
            'lane_positions': np.array([3.0, 6.0, 9.0]),
            'change_probability': 0.8,
            'change_duration': 2.0
        }
        return "Scenario 8: Lane Change (Frequent)"
    return f"Scenario {ep_idx+1}"


def plot_episode_trajectory(
    env,
    model,
    vec_normalize_path: str = None,
    n_scenarios: int = 8,
    save_dir: str = 'plots/trajectories',
    seed: int = 100
):
    """
    Run 1 episode for each scenario and plot the detailed trajectory.
    Saves 8 separate files.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    raw_env = env
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    print(f"\n--- Generating Single-Episode Trajectories ---")

    for ep_idx in range(n_scenarios):
        title = configure_env_for_scenario(raw_env, ep_idx)
        print(f"Processing {title}...")
        
        # Run Episode
        if vec_normalize_path:
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep_idx)
            
        done = False
        truncated = False
        
        # Storage
        rhos = []
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
            
            rhos.append(info['rho'])
            rates.append(info['R_sum'])
            crlbs_theta.append(np.mean(info['crlbs_theta']))
            crlbs_r.append(np.mean(info['crlbs_r']))

        # --- Plotting ---
        time_steps = np.arange(len(rhos))
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # 1. Rho
        axes[0].plot(time_steps, rhos, 'b-', linewidth=2, label='ρ')
        axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.3)
        axes[0].set_ylabel('Sensing Fraction (ρ)')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title(f'{title} (Single Run)', fontsize=14)
        
        # 2. Rate
        axes[1].plot(time_steps, rates, 'g-', linewidth=2, label='Rate')
        axes[1].set_ylabel('Sum Rate (bps/Hz)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Angle CRLB
        axes[2].semilogy(time_steps, crlbs_theta, 'r-', linewidth=2, label='Angle CRLB')
        axes[2].set_ylabel('CRLB (rad²)')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Range CRLB
        axes[3].semilogy(time_steps, crlbs_r, 'orange', linewidth=2, label='Range CRLB')
        axes[3].set_ylabel('CRLB (m²)')
        axes[3].set_xlabel('Time Step')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'trajectory_scenario_{ep_idx+1}.png'
        plt.savefig(save_path / filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")


def plot_rho_histogram_per_episode(
    env,
    model,
    vec_normalize_path: str = None,
    n_scenarios: int = 8,
    save_dir: str = 'plots/histograms',
    seed: int = 100
):
    """
    Run 1 episode for each scenario and plot the histogram of rho values.
    Saves 8 separate files.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    raw_env = env
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    print(f"\n--- Generating Single-Episode Histograms ---")

    for ep_idx in range(n_scenarios):
        title = configure_env_for_scenario(raw_env, ep_idx)
        print(f"Processing {title}...")
        
        # Run Episode
        if vec_normalize_path:
            obs = env.reset()
        else:
            obs, _ = env.reset(seed=seed + ep_idx)
            
        rhos = []
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
            
            rhos.append(info['rho'])

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 6))
        
        counts, bins, patches = ax.hist(rhos, bins=20, range=(0,1), 
                                      color='royalblue', edgecolor='black', alpha=0.7)
        
        mean_val = np.mean(rhos)
        median_val = np.median(rhos)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='lime', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'ρ Distribution: {title}\n(Single Run, {len(rhos)} steps)', fontsize=14)
        ax.set_xlabel('Sensing Fraction (ρ)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        filename = f'histogram_scenario_{ep_idx+1}.png'
        plt.savefig(save_path / filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description='Plot Single-Episode Analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to environment config')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--vec-normalize', type=str, default=None, help='Path to VecNormalize stats')
    parser.add_argument('--save-dir', type=str, default='plots', help='Root directory to save plots')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    
    args = parser.parse_args()
    
    # Load environment
    env = load_env_from_config(args.config)
    
    # Load model
    model_path = args.model
    if not model_path.endswith('.zip') and os.path.exists(model_path + '.zip'):
        model_path += '.zip'
    
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # 1. Trajectories (Single Run per Scenario)
    plot_episode_trajectory(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_scenarios=8,
        save_dir=os.path.join(args.save_dir, 'trajectories_single'),
        seed=args.seed
    )
    
    # 2. Histograms (Single Run per Scenario)
    plot_rho_histogram_per_episode(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_scenarios=8,
        save_dir=os.path.join(args.save_dir, 'histograms_single'),
        seed=args.seed
    )
    
    print("\nAll Single-Episode Plots Generated Successfully!")

if __name__ == '__main__':
    main()