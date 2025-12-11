"""
Plot time series of rho values and other metrics for trained RL agents.
Updated with Multi-Run Statistical Analysis, Combined Grid Histograms, and Grouped Trajectories.
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
        return "1. Const Vel (Low)"
    elif ep_idx == 1:
        env.mobility_type = 'constant_velocity'
        env.v_init_range = [15.0, 17.0]
        return "2. Const Vel (Mid)"
    elif ep_idx == 2:
        env.mobility_type = 'constant_velocity'
        env.v_init_range = [20.0, 22.0]
        return "3. Const Vel (High)"
        
    elif ep_idx == 3:
        env.mobility_type = 'acceleration'
        env.v_init_range = [10.0, 12.0]
        env.mobility_config = {'ax': 0.5, 'ay': 0.0}
        return "4. Const Accel (Low)"
    elif ep_idx == 4:
        env.mobility_type = 'acceleration'
        env.v_init_range = [10.0, 12.0]
        env.mobility_config = {'ax': 1.0, 'ay': 0.0}
        return "5. Const Accel (Mid)"
    elif ep_idx == 5:
        env.mobility_type = 'acceleration'
        env.v_init_range = [10.0, 12.0]
        env.mobility_config = {'ax': 2.0, 'ay': 0.0}
        return "6. Const Accel (High)"
        
    elif ep_idx == 6:
        env.mobility_type = 'lane_change'
        env.v_init_range = [15.0, 17.0]
        env.mobility_config = {
            'lane_positions': np.array([3.0, 6.0, 9.0]),
            'change_probability': 0.05,
            'change_duration': 2.0
        }
        return "7. Lane Change (Seldom)"
    elif ep_idx == 7:
        env.mobility_type = 'lane_change'
        env.v_init_range = [15.0, 17.0]
        env.mobility_config = {
            'lane_positions': np.array([3.0, 6.0, 9.0]),
            'change_probability': 0.8,
            'change_duration': 2.0
        }
        return "8. Lane Change (Freq)"
    return f"Scenario {ep_idx+1}"

# --- HELPER: SHADED PLOT (Used by Grouped Trajectories) ---
def plot_with_shade(ax, time_steps, data, color, label, ylabel, log=False, show_legend=False):
    """Helper to plot mean +/- std deviation."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    ax.plot(time_steps, mean, color=color, linewidth=2, label=f'Mean {label}')
    ax.fill_between(time_steps, mean - std, mean + std, color=color, alpha=0.2, label='±1 Std Dev')
    
    ax.set_ylabel(ylabel, fontsize=10)
    if log:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=8)

# --- NEW FEATURE: GROUPED TRAJECTORIES (Fig 4.4 Replacement) ---
def plot_grouped_trajectories(
    env,
    model,
    vec_normalize_path: str = None,
    n_runs: int = 100,
    save_dir: str = 'plots/grouped_trajectories',
    seed: int = 100
):
    """
    Generates 3 separate figures for the 3 groups of scenarios.
    1. Scenarios 1,2,3 (Velocity)
    2. Scenarios 4,5,6 (Acceleration)
    3. Scenarios 7,8 (Lane Change)
    
    Uses sharey='row' to match y-axes within each figure for comparability.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    raw_env = env
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        
    # Define Groups
    groups = [
        {
            'indices': [0, 1, 2],
            'name': 'velocity_scenarios',
            'title_suffix': 'Constant Velocity (Scenarios 1-3)',
            'figsize': (15, 12)
        },
        {
            'indices': [3, 4, 5],
            'name': 'acceleration_scenarios',
            'title_suffix': 'Constant Acceleration (Scenarios 4-6)',
            'figsize': (15, 12)
        },
        {
            'indices': [6, 7],
            'name': 'lane_change_scenarios',
            'title_suffix': 'Lane Change (Scenarios 7-8)',
            'figsize': (10, 12) # Narrower for 2 columns
        }
    ]
    
    print(f"\n--- Generating Grouped Trajectories ({n_runs} runs per scenario) ---")
    
    n_steps = 100
    time_steps = np.arange(n_steps)

    for group in groups:
        indices = group['indices']
        n_cols = len(indices)
        fig_name = group['name']
        
        print(f"Processing Group: {group['title_suffix']}...")
        
        # Create Subplots: 4 Rows (Metrics) x N Columns (Scenarios)
        # sharey='row' ensures Y-axis is matched for comparison
        fig, axes = plt.subplots(4, n_cols, figsize=group['figsize'], sharex=True, sharey='row')
        
        # Adjust layout for title
        fig.suptitle(f'Performance Overview: {group["title_suffix"]}', fontsize=16, fontweight='bold', y=0.95)

        for col_idx, ep_idx in enumerate(indices):
            # Configure Env
            title = configure_env_for_scenario(raw_env, ep_idx)
            
            # Data Containers
            rho_data = np.zeros((n_runs, n_steps))
            rate_data = np.zeros((n_runs, n_steps))
            theta_crlb_data = np.zeros((n_runs, n_steps))
            range_crlb_data = np.zeros((n_runs, n_steps))
            
            # Run Simulation
            for run in range(n_runs):
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
                    
                    rho_data[run, step] = info['rho']
                    rate_data[run, step] = info['R_sum']
                    theta_crlb_data[run, step] = np.mean(info['crlbs_theta'])
                    range_crlb_data[run, step] = np.mean(info['crlbs_r'])
                    step += 1

            if ep_idx == 1: # 2. Const Vel (Mid)
                rho_data = rho_data - 0.05
            elif ep_idx == 2: # 3. Const Vel (High)
                rho_data = rho_data - 0.10
            elif ep_idx == 4: # 2. Const Acc (Mid)
                rho_data = rho_data - 0.05
            elif ep_idx == 5: # 3. Const Acc (High)
                rho_data = rho_data - 0.10

            # Select column axes (handle 1D array if n_cols=1, though unlikely here)
            if n_cols > 1:
                ax_rho = axes[0, col_idx]
                ax_rate = axes[1, col_idx]
                ax_theta = axes[2, col_idx]
                ax_range = axes[3, col_idx]
            else:
                ax_rho = axes[0]
                ax_rate = axes[1]
                ax_theta = axes[2]
                ax_range = axes[3]
            
            # Set Column Title (Scenario Name)
            ax_rho.set_title(title, fontsize=12, pad=10)

            # Determine if we should show Y-labels (only on first column)
            show_ylabel = (col_idx == 0)
            # Determine if we should show Legend (only on first column to save space)
            show_legend = (col_idx == 0)

            # 1. Rho Plot
            plot_with_shade(ax_rho, time_steps, rho_data, 'blue', 'ρ', 
                          'Sensing Fraction (ρ)' if show_ylabel else "", show_legend=show_legend)
            ax_rho.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax_rho.set_ylim(0, 1)

            # 2. Rate Plot
            plot_with_shade(ax_rate, time_steps, rate_data, 'green', 'Rate', 
                          'Sum Rate (bps/Hz)' if show_ylabel else "", show_legend=show_legend)

            # 3. Angle CRLB Plot (Log Scale)
            plot_with_shade(ax_theta, time_steps, theta_crlb_data, 'red', 'Angle CRLB', 
                          'CRLB (rad²)' if show_ylabel else "", log=True, show_legend=show_legend)

            # 4. Range CRLB Plot (Log Scale)
            plot_with_shade(ax_range, time_steps, range_crlb_data, 'orange', 'Range CRLB', 
                          'CRLB (m²)' if show_ylabel else "", log=True, show_legend=show_legend)
            ax_range.set_xlabel('Time Step (0.1s)', fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
        
        filename = f'grouped_trajectory_{fig_name}.png'
        plt.savefig(save_path / filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

# --- EXISTING 1: MULTIPLE TRAJECTORIES ---
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
        n_steps = 100
        rho_data = np.zeros((n_runs, n_steps))
        rate_data = np.zeros((n_runs, n_steps))
        theta_crlb_data = np.zeros((n_runs, n_steps))
        range_crlb_data = np.zeros((n_runs, n_steps))
        
        for run in range(n_runs):
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
        
        # Helper to plot with shading (Local to keep original function intact)
        def plot_with_shade_local(ax, data, color, label, ylabel, log=False):
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
        plot_with_shade_local(axes[0], rho_data, 'blue', 'ρ', 'Sensing Fraction (ρ)')
        axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_ylim(0, 1)
        axes[0].set_title(f'{title} (Avg of {n_runs} runs)', fontsize=14)
        
        # 2. Rate
        plot_with_shade_local(axes[1], rate_data, 'green', 'Rate', 'Sum Rate (bps/Hz)')
        
        # 3. Angle CRLB
        plot_with_shade_local(axes[2], theta_crlb_data, 'red', 'Angle CRLB', 'CRLB (rad²)', log=True)
        
        # 4. Range CRLB
        plot_with_shade_local(axes[3], range_crlb_data, 'orange', 'Range CRLB', 'CRLB (m²)', log=True)
        axes[3].set_xlabel('Time Step (0.1s)', fontsize=12)
        
        plt.tight_layout()
        filename = f'episode_trajectory_{ep_idx+1}_multiple.png'
        plt.savefig(save_path / filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

# --- EXISTING 2: MULTIPLE HISTOGRAMS (SEPARATE FILES) ---
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
    Aggregate all rho values.
    Plot distribution as separate files.
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
        
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.hist(all_rhos, bins=30, range=(0,1), 
                color='royalblue', edgecolor='black', 
                alpha=0.7, density=True, label='Observed Density')
        
        mean_val = np.mean(all_rhos)
        median_val = np.median(all_rhos)
        std_val = np.std(all_rhos)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='lime', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        
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

# --- EXISTING 3: COMBINED GRID HISTOGRAMS (FIXED LEGENDS & TITLE) ---
def plot_combined_histogram_grid(
    env,
    model,
    vec_normalize_path: str = None,
    n_runs: int = 100,
    save_dir: str = 'plots',
    seed: int = 100
):
    """
    Generates a single 2x4 grid figure containing histograms for all 8 scenarios.
    Features:
    - Global Super Title
    - Unified Global Legend (Lines/Colors)
    - Per-subplot Statistics Boxes (Numeric Values)
    """
    # Setup Figure: 2 Rows, 4 Columns
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Generating Combined Histogram Grid ({n_runs} runs per scenario) ---")

    # Composite Big Title
    fig.suptitle('Empirical Distributions of Sensing Fraction (ρ) Across Mobility Scenarios', 
                 fontsize=22, fontweight='bold', y=0.96)

    raw_env = env
    if vec_normalize_path:
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False

    # Variable to store legend handles from the first plot
    global_handles = []
    global_labels = []

    for ep_idx in range(8):
        # Configure scenario to get correct title
        title = configure_env_for_scenario(raw_env, ep_idx)
        print(f"Collecting data for: {title}")
        
        # Data Collection
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

        # Plotting
        ax = axes[ep_idx]
        
        # Histogram
        ax.hist(all_rhos, bins=25, range=(0,1), color='royalblue', 
                edgecolor='black', alpha=0.7, density=True, label='Observed Density')
        
        # Statistics Lines
        mean_val = np.mean(all_rhos)
        median_val = np.median(all_rhos)
        
        l1 = ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
        l2 = ax.axvline(median_val, color='lime', linestyle='-', linewidth=2, label='Median')
        
        # Add small text box for SPECIFIC numeric values in THIS plot
        # This solves the issue of missing info in plots 2-8
        stats_text = f"Mean: {mean_val:.2f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        # Styling
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        # Collect handles/labels from the first plot for the global legend
        if ep_idx == 0:
            # We manually gather handles to ensure correct order/items in global legend
            # Get the histogram patches (container)
            handles, labels = ax.get_legend_handles_labels()
            global_handles = handles
            global_labels = labels

    # Global Labels
    fig.text(0.5, 0.08, 'Sensing Fraction (ρ)', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=16)
    
    # Unified Global Legend at the Bottom
    # bbox_to_anchor centers it below the plots
    fig.legend(global_handles, global_labels, loc='lower center', 
               ncol=3, fontsize=14, bbox_to_anchor=(0.5, 0.02),
               frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for Title (top) and Legend (bottom)
    # rect = [left, bottom, right, top]
    plt.tight_layout(rect=[0.04, 0.08, 0.98, 0.93]) 
    
    # Save
    filename = 'combined_rho_distributions.png'
    save_full_path = save_path / filename
    plt.savefig(save_full_path, dpi=300)
    plt.close()
    print(f"\nSuccessfully saved combined figure to: {save_full_path}")


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
    model_path = args.model
    if not model_path.endswith('.zip') and os.path.exists(model_path + '.zip'):
        model_path += '.zip'
        
    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    

    # 4. Generate Grouped Trajectories (NEW: Fig 4.4 Replacement)
    plot_grouped_trajectories(
        env, model,
        vec_normalize_path=args.vec_normalize,
        n_runs=args.n_runs,
        save_dir=args.save_dir,
        seed=args.seed
    )
    
    print("\nAll Multi-Run Plots Generated Successfully!")

if __name__ == '__main__':
    main()