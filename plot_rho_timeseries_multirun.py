"""
Plot time series of rho values and other metrics for trained RL agents.
Updated with Multi-Run Statistical Analysis.
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

# --- NEW 1: MULTIPLE TRAJECTORIES ---
def plot_trajectory_multiple(
    env,
    model,
    vec_normalize_path: str = None,
    n_scenarios: int = 8,
    n_runs: int = 100,
    save_dir: str = 'plots/trajectories',
    seed: int = 100
):
    """
    Run each scenario n_runs times.
    Plot Mean (Solid Line) +/- Std Dev (Shaded Region).
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    raw_env = env
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    print(f"\n--- Generating Multi-Run Trajectories ({n_runs} runs per scenario) ---")

    for ep_idx in range(n_scenarios):
        title = configure_env_for_scenario(raw_env, ep_idx)
        print(f"Processing {title}...")
        
        # Storage for [Run, Step]
        # Assuming 100 steps per episode
        n_steps = 100
        rho_data = np.zeros((n_runs, n_steps))
        rate_data = np.zeros((n_runs, n_steps))
        theta_crlb_data = np.zeros((n_runs, n_steps))
        range_crlb_data = np.zeros((n_runs, n_steps))
        
        for run in range(n_runs):
            # Unique seed for each run to ensure variety
            run_seed = seed + (ep_idx * 1000) + run
            
            if vec_normalize_path:
                obs = env.reset()
            else:
                obs, _ = env.reset(seed=run_seed)
                
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < n_steps:
                if vec_normalize_path:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done_vec, info = env.step(action)
                    done = done_vec[0]
                    info = info[0] if isinstance(info, list) else info
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                
                # Record metrics
                rho_data[run, step] = info['rho']
                rate_data[run, step] = info['R_sum']
                theta_crlb_data[run, step] = np.mean(info['crlbs_theta'])
                range_crlb_data[run, step] = np.mean(info['crlbs_r'])
                
                step += 1
                
        # --- Plotting Smartly (Mean + Shaded Std) ---
        time_steps = np.arange(n_steps)
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # Helper to plot with shading
        def plot_with_shade(ax, data, color, label, ylabel, log=False):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            
            ax.plot(time_steps, mean, color=color, linewidth=2, label=f'Mean {label}')
            ax.fill_between(time_steps, mean - std, mean + std, color=color, alpha=0.2, label='±1 Std Dev')
            
            ax.set_ylabel(ylabel, fontsize=10)
            if log:
                ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        # 1. Rho
        plot_with_shade(axes[0], rho_data, 'blue', 'ρ', 'Sensing Fraction (ρ)')
        axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f'{title} (Avg of {n_runs} runs)', fontsize=14)
        
        # 2. Rate
        plot_with_shade(axes[1], rate_data, 'green', 'Rate', 'Sum Rate (bps/Hz)')
        
        # 3. Angle CRLB
        plot_with_shade(axes[2], theta_crlb_data, 'red', 'Angle CRLB', 'CRLB (rad²)', log=True)
        
        # 4. Range CRLB
        plot_with_shade(axes[3], range_crlb_data, 'orange', 'Range CRLB', 'CRLB (m²)', log=True)
        axes[3].set_xlabel('Time Step (0.1s)', fontsize=12)
        
        plt.tight_layout()
        filename = f'episode_trajectory_{ep_idx+1}_multiple.png'
        plt.savefig(save_path / filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

# --- NEW 2: MULTIPLE HISTOGRAMS ---
def plot_histogram_multiple(
    env,
    model,
    vec_normalize_path: str = None,
    n_scenarios: int = 8,
    n_runs: int = 100,
    save_dir: str = 'plots/histograms',
    seed: int = 100
):
    """
    Run each scenario n_runs times.
    Aggregate all rho values (n_runs * n_steps).
    Plot distribution.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    raw_env = env
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    print(f"\n--- Generating Multi-Run Histograms ({n_runs} runs per scenario) ---")

    for ep_idx in range(n_scenarios):
        title = configure_env_for_scenario(raw_env, ep_idx)
        print(f"Processing {title}...")
        
        all_rhos = []
        
        for run in range(n_runs):
            run_seed = seed + (ep_idx * 1000) + run
            
            if vec_normalize_path:
                obs = env.reset()
            else:
                obs, _ = env.reset(seed=run_seed)
                
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
        
        # --- Plotting Smartly (Aggregated Distribution) ---
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Main Histogram
        counts, bins, patches = ax.hist(all_rhos, bins=30, range=(0,1), 
                                      color='royalblue', edgecolor='black', 
                                      alpha=0.7, density=True, label='Observed Density')
        
        # Statistics Lines
        mean_val = np.mean(all_rhos)
        median_val = np.median(all_rhos)
        std_val = np.std(all_rhos)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='lime', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Add a text box with stats
        stats_text = f"N Samples: {len(all_rhos)}\nMean: {mean_val:.3f}\nStd Dev: {std_val:.3f}"
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_title(f'ρ Distribution: {title}\n({n_runs} episodes aggregated)', fontsize=14)
        ax.set_xlabel('Sensing Fraction (ρ)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        filename = f'rho_hist_scenario_{ep_idx+1}_multiple.png'
        plt.savefig(save_path / filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description='Plot Multi-Run Analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to environment config')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--vec-normalize', type=str, default=None, help='Path to VecNormalize stats')
    parser.add_argument('--n-runs', type=int, default=100, help='Number of runs per scenario')
    parser.add_argument('--save-dir', type=str, default='plots', help='Root directory to save plots')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')
    
    args = parser.parse_args()
    
    # Load environment
    env = load_env_from_config(args.config)
    
    # Load model
    # Handle .zip extension automatically
    model_path = args.model
    if not model_path.endswith('.zip') and os.path.exists(model_path + '.zip'):
        model_path += '.zip'
        
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    # 1. Generate Multiple Trajectories (New 1)
    plot_trajectory_multiple(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_scenarios=8,
        n_runs=args.n_runs,
        save_dir=os.path.join(args.save_dir, 'trajectories'),
        seed=args.seed
    )
    
    # 2. Generate Multiple Histograms (Fix 1 + New 2)
    plot_histogram_multiple(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_scenarios=8,
        n_runs=args.n_runs,
        save_dir=os.path.join(args.save_dir, 'histograms'),
        seed=args.seed
    )
    
    print("\nAll Multi-Run Plots Generated Successfully!")

if __name__ == '__main__':
    main()