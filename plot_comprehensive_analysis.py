"""
Comprehensive ISAC Analysis Plots
Generates detailed visualizations for P1 and P2 under various mobility scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from envs.vehicular_isac_env import VehicularISACEnv


def create_env(config: Dict, mobility_type: str = None, 
               speed_range: Tuple[float, float] = None):
    """Create environment with specific mobility settings."""
    config_copy = config.copy()
    
    # Override mobility if specified
    if mobility_type is not None:
        config_copy['mobility_type'] = mobility_type
        config_copy['mobility_types'] = None  # Disable random selection
    
    # Override speed range if specified
    if speed_range is not None:
        config_copy['v_init_range'] = list(speed_range)
    
    def _make_env():
        return VehicularISACEnv(config_copy)
    
    env = DummyVecEnv([_make_env])
    return env


def run_evaluation_episodes(model, env, vec_normalize, n_episodes: int = 50,
                           seed: int = None) -> Dict[str, List]:
    """Run evaluation episodes and collect detailed metrics."""
    
    if seed is not None:
        np.random.seed(seed)
    
    all_data = {
        'episodes': [],
        'rho': [],
        'rates': [],
        'crlb_angle': [],
        'crlb_range': [],
        'positions': [],
        'distances': [],  # Changed from 'distances_to_rsu'
        'timesteps': []
    }
    
    for ep in range(n_episodes):
        obs = env.reset()
        # Note: VecNormalize is now part of env, no need to normalize separately
        
        episode_data = {
            'rho': [],
            'rates': [],
            'crlb_angle': [],
            'crlb_range': [],
            'positions': [],
            'distances': [],
            'timesteps': []
        }
        
        done = False
        step = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Note: obs is already normalized if VecNormalize is in env
            
            # Extract metrics from info
            info_dict = info[0] if isinstance(info, list) else info
            
            episode_data['rho'].append(info_dict.get('rho', 0.5))
            episode_data['rates'].append(info_dict.get('R_sum', 0.0))
            
            crlbs_theta = info_dict.get('crlbs_theta', [])
            crlbs_r = info_dict.get('crlbs_r', [])
            
            episode_data['crlb_angle'].append(np.mean(crlbs_theta) if len(crlbs_theta) > 0 else 1e-4)
            episode_data['crlb_range'].append(np.mean(crlbs_r) if len(crlbs_r) > 0 else 1e-3)
            
            positions = info_dict.get('positions', np.zeros((3, 2)))
            episode_data['positions'].append(positions.copy())
            
            # Calculate distance to RSU (assumed at origin)
            distances = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
            episode_data['distances'].append(np.mean(distances))
            
            episode_data['timesteps'].append(step)
            step += 1
            
            if done:
                break
        
        # Store episode data
        for key in ['rho', 'rates', 'crlb_angle', 'crlb_range', 'distances', 'timesteps']:
            all_data[key].append(np.array(episode_data[key]))
        all_data['positions'].append(episode_data['positions'])
        all_data['episodes'].append(ep)
    
    return all_data


# ============================================================================
# Plot Category A: Constant Velocity Scenarios
# ============================================================================

def plot_A1_time_evolution(data_dict: Dict[str, Dict], save_path: Path):
    """Plot A1: Time-division and KPIs vs distance for different speeds."""
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    speeds = ['low', 'mid', 'high']
    colors = ['blue', 'green', 'red']
    
    for speed, color in zip(speeds, colors):
        if speed not in data_dict:
            continue
        
        data = data_dict[speed]
        
        # Average across episodes
        n_steps = min([len(d) for d in data['distances']])
        
        avg_distances = np.mean([d[:n_steps] for d in data['distances']], axis=0)
        avg_rho = np.mean([r[:n_steps] for r in data['rho']], axis=0)
        avg_rate = np.mean([r[:n_steps] for r in data['rates']], axis=0)
        avg_crlb_angle = np.mean([c[:n_steps] for c in data['crlb_angle']], axis=0)
        
        # Plot sensing fraction
        axes[0].plot(avg_distances, avg_rho, color=color, label=f'{speed.capitalize()} speed', linewidth=2)
        axes[0].set_ylabel('Sensing Fraction (ρ)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot throughput
        axes[1].plot(avg_distances, avg_rate, color=color, label=f'{speed.capitalize()} speed', linewidth=2)
        axes[1].set_ylabel('Sum Rate (bits/s/Hz)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot CRLB angle
        axes[2].semilogy(avg_distances, avg_crlb_angle, color=color, label=f'{speed.capitalize()} speed', linewidth=2)
        axes[2].set_ylabel('Angle CRLB (rad²)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot comm fraction (1-ρ)
        axes[3].plot(avg_distances, 1 - avg_rho, color=color, label=f'{speed.capitalize()} speed', linewidth=2)
        axes[3].set_ylabel('Comm Fraction (1-ρ)')
        axes[3].set_xlabel('Distance to RSU (m)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
    
    plt.suptitle('Time Evolution: Adaptation vs Distance (Constant Velocity)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'A1_time_evolution_vs_distance.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_A2_tradeoff_curves(data_dict: Dict[str, Dict], save_path: Path):
    """Plot A2: Communication-Sensing tradeoff at each speed."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    speeds = ['low', 'mid', 'high']
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for speed, color, marker in zip(speeds, colors, markers):
        if speed not in data_dict:
            continue
        
        data = data_dict[speed]
        
        # Compute average metrics per episode
        avg_rates = [np.mean(r) for r in data['rates']]
        avg_crlb_angle = [np.mean(c) for c in data['crlb_angle']]
        avg_crlb_range = [np.mean(c) for c in data['crlb_range']]
        
        # Rate vs Angle CRLB
        ax1.scatter(avg_crlb_angle, avg_rates, c=color, marker=marker, 
                   label=f'{speed.capitalize()} speed', alpha=0.6, s=100)
        ax1.set_xlabel('Mean Angle CRLB (rad²)')
        ax1.set_ylabel('Mean Sum Rate (bits/s/Hz)')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Rate vs Angle CRLB Tradeoff')
        
        # Rate vs Range CRLB
        ax2.scatter(avg_crlb_range, avg_rates, c=color, marker=marker,
                   label=f'{speed.capitalize()} speed', alpha=0.6, s=100)
        ax2.set_xlabel('Mean Range CRLB (m²)')
        ax2.set_ylabel('Mean Sum Rate (bits/s/Hz)')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('Rate vs Range CRLB Tradeoff')
    
    plt.suptitle('Communication-Sensing Tradeoff Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'A2_tradeoff_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_A3_metrics_vs_velocity(data_dict: Dict[str, Dict], velocity_values: Dict[str, float],
                                save_path: Path):
    """Plot A3: Key metrics vs velocity."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    speeds = ['low', 'mid', 'high']
    velocities = [velocity_values[s] for s in speeds if s in data_dict]
    
    # Collect metrics
    avg_throughput = []
    avg_crlb_angle = []
    avg_crlb_range = []
    avg_rho = []
    
    for speed in speeds:
        if speed not in data_dict:
            continue
        
        data = data_dict[speed]
        avg_throughput.append(np.mean([np.mean(r) for r in data['rates']]))
        avg_crlb_angle.append(np.mean([np.mean(c) for c in data['crlb_angle']]))
        avg_crlb_range.append(np.mean([np.mean(c) for c in data['crlb_range']]))
        avg_rho.append(np.mean([np.mean(r) for r in data['rho']]))
    
    # Plot 1: Average Throughput
    axes[0].plot(velocities, avg_throughput, 'o-', linewidth=2, markersize=10, color='blue')
    axes[0].set_xlabel('Velocity (m/s)')
    axes[0].set_ylabel('Avg Sum Rate (bits/s/Hz)')
    axes[0].set_title('Average Throughput vs Velocity')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Average Angle CRLB
    axes[1].semilogy(velocities, avg_crlb_angle, 's-', linewidth=2, markersize=10, color='red')
    axes[1].set_xlabel('Velocity (m/s)')
    axes[1].set_ylabel('Avg Angle CRLB (rad²)')
    axes[1].set_title('Average Angle CRLB vs Velocity')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Average Range CRLB
    axes[2].semilogy(velocities, avg_crlb_range, '^-', linewidth=2, markersize=10, color='green')
    axes[2].set_xlabel('Velocity (m/s)')
    axes[2].set_ylabel('Avg Range CRLB (m²)')
    axes[2].set_title('Average Range CRLB vs Velocity')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Average Sensing Fraction
    axes[3].plot(velocities, avg_rho, 'D-', linewidth=2, markersize=10, color='purple')
    axes[3].set_xlabel('Velocity (m/s)')
    axes[3].set_ylabel('Avg Sensing Fraction (ρ)')
    axes[3].set_title('Average Sensing Allocation vs Velocity')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Performance Metrics vs Velocity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'A3_metrics_vs_velocity.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_A4_heatmap_distance_speed(data_dict: Dict[str, Dict], save_path: Path):
    """Plot A4: Heatmap over (distance, speed)."""
    
    speeds = ['low', 'mid', 'high']
    
    # Create distance bins
    distance_bins = np.linspace(0, 200, 20)
    
    # Initialize arrays for heatmaps
    throughput_matrix = np.zeros((len(speeds), len(distance_bins) - 1))
    crlb_matrix = np.zeros((len(speeds), len(distance_bins) - 1))
    
    for speed_idx, speed in enumerate(speeds):
        if speed not in data_dict:
            continue
        
        data = data_dict[speed]
        
        # Concatenate all episodes
        all_distances = np.concatenate(data['distances'])
        all_rates = np.concatenate(data['rates'])
        all_crlb = np.concatenate(data['crlb_angle'])
        
        # Bin by distance
        for bin_idx in range(len(distance_bins) - 1):
            bin_mask = (all_distances >= distance_bins[bin_idx]) & \
                      (all_distances < distance_bins[bin_idx + 1])
            
            if np.sum(bin_mask) > 0:
                throughput_matrix[speed_idx, bin_idx] = np.mean(all_rates[bin_mask])
                crlb_matrix[speed_idx, bin_idx] = np.mean(all_crlb[bin_mask])
    
    # Create heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Throughput heatmap
    im1 = ax1.imshow(throughput_matrix, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_xlabel('Distance to RSU (m)')
    ax1.set_ylabel('Speed Category')
    ax1.set_yticks(range(len(speeds)))
    ax1.set_yticklabels([s.capitalize() for s in speeds])
    ax1.set_xticks(np.arange(0, len(distance_bins) - 1, 3))
    ax1.set_xticklabels([f'{int(d)}' for d in distance_bins[::3]])
    ax1.set_title('Average Throughput (bits/s/Hz)')
    plt.colorbar(im1, ax=ax1)
    
    # CRLB heatmap
    im2 = ax2.imshow(np.log10(crlb_matrix), aspect='auto', origin='lower', cmap='RdYlGn_r')
    ax2.set_xlabel('Distance to RSU (m)')
    ax2.set_ylabel('Speed Category')
    ax2.set_yticks(range(len(speeds)))
    ax2.set_yticklabels([s.capitalize() for s in speeds])
    ax2.set_xticks(np.arange(0, len(distance_bins) - 1, 3))
    ax2.set_xticklabels([f'{int(d)}' for d in distance_bins[::3]])
    ax2.set_title('log₁₀(Angle CRLB)')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle('Operating Region: Distance × Speed', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'A4_heatmap_distance_speed.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Plot Category B: Acceleration Scenarios
# ============================================================================

def plot_B1_acceleration_response(data: Dict, save_path: Path):
    """Plot B1: Response to acceleration events."""
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # Select 3 representative episodes
    n_episodes = min(3, len(data['rho']))
    
    colors = ['blue', 'green', 'red']
    
    for ep_idx in range(n_episodes):
        color = colors[ep_idx]
        timesteps = data['timesteps'][ep_idx]
        
        # Estimate velocity changes (acceleration proxy)
        positions = data['positions'][ep_idx]
        velocities = np.diff([p[0, 0] for p in positions]) / 0.1  # dt = 0.1s
        velocities = np.concatenate([[velocities[0]], velocities])  # Pad
        
        # Plot velocity profile
        axes[0].plot(timesteps, velocities, color=color, label=f'Episode {ep_idx+1}', linewidth=2)
        axes[0].set_ylabel('Velocity (m/s)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title('Velocity Profile')
        
        # Plot sensing fraction
        axes[1].plot(timesteps, data['rho'][ep_idx], color=color, linewidth=2)
        axes[1].set_ylabel('Sensing Fraction (ρ)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title('Sensing Allocation Response')
        
        # Plot CRLB angle
        axes[2].semilogy(timesteps, data['crlb_angle'][ep_idx], color=color, linewidth=2)
        axes[2].set_ylabel('Angle CRLB (rad²)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('Sensing Performance')
        
        # Plot CRLB range
        axes[3].semilogy(timesteps, data['crlb_range'][ep_idx], color=color, linewidth=2)
        axes[3].set_ylabel('Range CRLB (m²)')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title('Range Tracking Performance')
        
        # Plot throughput
        axes[4].plot(timesteps, data['rates'][ep_idx], color=color, linewidth=2)
        axes[4].set_ylabel('Sum Rate (bits/s/Hz)')
        axes[4].set_xlabel('Time Step')
        axes[4].grid(True, alpha=0.3)
        axes[4].set_title('Communication Performance')
    
    plt.suptitle('Response to Acceleration Events', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'B1_acceleration_response.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_B2_tracking_vs_acceleration(data_dict: Dict[str, Dict], save_path: Path):
    """Plot B2: Tracking quality vs acceleration magnitude."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    accel_levels = ['low', 'mid', 'high']
    accel_values = [0.5, 1.0, 2.0]  # Example values
    
    # Collect metrics
    avg_throughput = []
    avg_crlb_angle = []
    avg_crlb_range = []
    avg_rho = []
    
    for accel in accel_levels:
        if accel not in data_dict:
            continue
        
        data = data_dict[accel]
        avg_throughput.append(np.mean([np.mean(r) for r in data['rates']]))
        avg_crlb_angle.append(np.mean([np.mean(c) for c in data['crlb_angle']]))
        avg_crlb_range.append(np.mean([np.mean(c) for c in data['crlb_range']]))
        avg_rho.append(np.mean([np.mean(r) for r in data['rho']]))
    
    # Plot metrics
    axes[0].plot(accel_values[:len(avg_throughput)], avg_throughput, 'o-', linewidth=2, markersize=10)
    axes[0].set_xlabel('Acceleration (m/s²)')
    axes[0].set_ylabel('Avg Sum Rate (bits/s/Hz)')
    axes[0].set_title('Throughput vs Acceleration')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(accel_values[:len(avg_crlb_angle)], avg_crlb_angle, 's-', linewidth=2, markersize=10)
    axes[1].set_xlabel('Acceleration (m/s²)')
    axes[1].set_ylabel('Avg Angle CRLB (rad²)')
    axes[1].set_title('Angle CRLB vs Acceleration')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogy(accel_values[:len(avg_crlb_range)], avg_crlb_range, '^-', linewidth=2, markersize=10)
    axes[2].set_xlabel('Acceleration (m/s²)')
    axes[2].set_ylabel('Avg Range CRLB (m²)')
    axes[2].set_title('Range CRLB vs Acceleration')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(accel_values[:len(avg_rho)], avg_rho, 'D-', linewidth=2, markersize=10)
    axes[3].set_xlabel('Acceleration (m/s²)')
    axes[3].set_ylabel('Avg Sensing Fraction (ρ)')
    axes[3].set_title('Sensing Allocation vs Acceleration')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Tracking Quality vs Acceleration Magnitude', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'B2_tracking_vs_acceleration.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Plot Category C: Lane Change Scenarios
# ============================================================================

def plot_C1_lane_change_timeline(data: Dict, save_path: Path):
    """Plot C1: Lane change event timeline."""
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # Select representative episode
    ep_idx = 0
    timesteps = data['timesteps'][ep_idx]
    
    # Detect lane changes (y-position changes)
    positions = data['positions'][ep_idx]
    y_positions = [p[0, 1] for p in positions]
    
    # Simple lane change detection: large y-change
    lane_changes = []
    for i in range(1, len(y_positions)):
        if abs(y_positions[i] - y_positions[i-1]) > 0.5:  # Threshold
            lane_changes.append(i)
    
    # Plot sensing fraction
    axes[0].plot(timesteps, data['rho'][ep_idx], linewidth=2, color='blue')
    axes[0].set_ylabel('Sensing Fraction (ρ)')
    axes[0].set_title('Time-Division Adaptation')
    for lc in lane_changes:
        axes[0].axvline(timesteps[lc], color='red', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    
    # Plot throughput
    axes[1].plot(timesteps, data['rates'][ep_idx], linewidth=2, color='green')
    axes[1].set_ylabel('Sum Rate (bits/s/Hz)')
    axes[1].set_title('Communication Performance')
    for lc in lane_changes:
        axes[1].axvline(timesteps[lc], color='red', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # Plot CRLB angle
    axes[2].semilogy(timesteps, data['crlb_angle'][ep_idx], linewidth=2, color='purple')
    axes[2].set_ylabel('Angle CRLB (rad²)')
    axes[2].set_title('Angle Tracking Performance')
    for lc in lane_changes:
        axes[2].axvline(timesteps[lc], color='red', linestyle='--', alpha=0.5, label='Lane Change' if lc == lane_changes[0] else '')
    axes[2].grid(True, alpha=0.3)
    if lane_changes:
        axes[2].legend()
    
    # Plot CRLB range
    axes[3].semilogy(timesteps, data['crlb_range'][ep_idx], linewidth=2, color='orange')
    axes[3].set_ylabel('Range CRLB (m²)')
    axes[3].set_title('Range Tracking Performance')
    for lc in lane_changes:
        axes[3].axvline(timesteps[lc], color='red', linestyle='--', alpha=0.5)
    axes[3].grid(True, alpha=0.3)
    
    # Plot y-position (lane)
    axes[4].plot(timesteps, y_positions, linewidth=2, color='brown')
    axes[4].set_ylabel('Y Position (m)')
    axes[4].set_xlabel('Time Step')
    axes[4].set_title('Lateral Position (Lane)')
    for lc in lane_changes:
        axes[4].axvline(timesteps[lc], color='red', linestyle='--', alpha=0.5)
    axes[4].grid(True, alpha=0.3)
    
    plt.suptitle('Lane Change Event Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'C1_lane_change_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Plot Category D: Cross-Scenario Summaries
# ============================================================================

def plot_D1_scenario_comparison(all_scenarios_data: Dict[str, Dict], save_path: Path):
    """Plot D1: Bar chart comparing scenarios."""
    
    scenarios = list(all_scenarios_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Collect metrics for each scenario
    metrics = {
        'throughput': [],
        'crlb_angle': [],
        'crlb_range': [],
        'rho': []
    }
    
    for scenario in scenarios:
        data = all_scenarios_data[scenario]
        metrics['throughput'].append(np.mean([np.mean(r) for r in data['rates']]))
        metrics['crlb_angle'].append(np.mean([np.mean(c) for c in data['crlb_angle']]))
        metrics['crlb_range'].append(np.mean([np.mean(c) for c in data['crlb_range']]))
        metrics['rho'].append(np.mean([np.mean(r) for r in data['rho']]))
    
    # Plot bars
    x = np.arange(len(scenarios))
    
    axes[0].bar(x, metrics['throughput'], color='steelblue', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[0].set_ylabel('Avg Sum Rate (bits/s/Hz)')
    axes[0].set_title('Average Throughput')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, metrics['crlb_angle'], color='coral', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[1].set_ylabel('Avg Angle CRLB (rad²)')
    axes[1].set_yscale('log')
    axes[1].set_title('Average Angle CRLB')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(x, metrics['crlb_range'], color='lightgreen', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[2].set_ylabel('Avg Range CRLB (m²)')
    axes[2].set_yscale('log')
    axes[2].set_title('Average Range CRLB')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    axes[3].bar(x, metrics['rho'], color='mediumpurple', alpha=0.8)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(scenarios, rotation=45, ha='right')
    axes[3].set_ylabel('Avg Sensing Fraction (ρ)')
    axes[3].set_title('Average Sensing Allocation')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Cross-Scenario Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'D1_scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_D2_cdf_analysis(all_data: Dict, save_path: Path):
    """Plot D2: CDFs of performance metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Collect all per-episode metrics
    all_throughput = [np.mean(r) for r in all_data['rates']]
    all_crlb_angle = [np.mean(c) for c in all_data['crlb_angle']]
    all_crlb_range = [np.mean(c) for c in all_data['crlb_range']]
    all_rho = [np.mean(r) for r in all_data['rho']]
    
    # Compute reliability metrics
    R_min = 2.0  # Example threshold
    CRLB_angle_max = 1e-4
    CRLB_range_max = 1e-3
    
    comm_reliability = np.mean([r >= R_min for r in all_throughput])
    sens_angle_reliability = np.mean([c <= CRLB_angle_max for c in all_crlb_angle])
    sens_range_reliability = np.mean([c <= CRLB_range_max for c in all_crlb_range])
    
    # Plot CDFs
    axes[0].hist(all_throughput, bins=50, cumulative=True, density=True, alpha=0.7, color='blue')
    axes[0].axvline(R_min, color='red', linestyle='--', label=f'Threshold ({R_min} bits/s/Hz)')
    axes[0].set_xlabel('Sum Rate (bits/s/Hz)')
    axes[0].set_ylabel('CDF')
    axes[0].set_title(f'Throughput CDF\n(Reliability: {comm_reliability:.1%})')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].hist(all_crlb_angle, bins=50, cumulative=True, density=True, alpha=0.7, color='red')
    axes[1].axvline(CRLB_angle_max, color='darkred', linestyle='--', label=f'Threshold ({CRLB_angle_max})')
    axes[1].set_xlabel('Angle CRLB (rad²)')
    axes[1].set_xscale('log')
    axes[1].set_ylabel('CDF')
    axes[1].set_title(f'Angle CRLB CDF\n(Reliability: {sens_angle_reliability:.1%})')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].hist(all_crlb_range, bins=50, cumulative=True, density=True, alpha=0.7, color='green')
    axes[2].axvline(CRLB_range_max, color='darkgreen', linestyle='--', label=f'Threshold ({CRLB_range_max})')
    axes[2].set_xlabel('Range CRLB (m²)')
    axes[2].set_xscale('log')
    axes[2].set_ylabel('CDF')
    axes[2].set_title(f'Range CRLB CDF\n(Reliability: {sens_range_reliability:.1%})')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    axes[3].hist(all_rho, bins=50, cumulative=True, density=True, alpha=0.7, color='purple')
    axes[3].set_xlabel('Sensing Fraction (ρ)')
    axes[3].set_ylabel('CDF')
    axes[3].set_title('Sensing Allocation CDF')
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Performance Analysis (CDFs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'D2_cdf_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_D3_p1_vs_p2_comparison(p1_data: Dict, p2_data: Dict, save_path: Path):
    """Plot D3: Direct P1 vs P2 comparison."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Rho distribution
    all_rho_p1 = np.concatenate([r for r in p1_data['rho']])
    all_rho_p2 = np.concatenate([r for r in p2_data['rho']])
    
    axes[0, 0].hist(all_rho_p1, bins=50, alpha=0.6, label='P1 (Sensing)', color='blue', density=True)
    axes[0, 0].hist(all_rho_p2, bins=50, alpha=0.6, label='P2 (Comm)', color='red', density=True)
    axes[0, 0].set_xlabel('Sensing Fraction (ρ)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('ρ Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Rate distribution
    all_rate_p1 = np.concatenate([r for r in p1_data['rates']])
    all_rate_p2 = np.concatenate([r for r in p2_data['rates']])
    
    axes[0, 1].hist(all_rate_p1, bins=50, alpha=0.6, label='P1 (Sensing)', color='blue', density=True)
    axes[0, 1].hist(all_rate_p2, bins=50, alpha=0.6, label='P2 (Comm)', color='red', density=True)
    axes[0, 1].set_xlabel('Sum Rate (bits/s/Hz)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Rate Distribution Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: CRLB angle distribution
    all_crlb_p1 = np.concatenate([c for c in p1_data['crlb_angle']])
    all_crlb_p2 = np.concatenate([c for c in p2_data['crlb_angle']])
    
    axes[0, 2].hist(np.log10(all_crlb_p1), bins=50, alpha=0.6, label='P1 (Sensing)', color='blue', density=True)
    axes[0, 2].hist(np.log10(all_crlb_p2), bins=50, alpha=0.6, label='P2 (Comm)', color='red', density=True)
    axes[0, 2].set_xlabel('log₁₀(Angle CRLB)')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Angle CRLB Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Tradeoff scatter
    avg_rate_p1 = [np.mean(r) for r in p1_data['rates']]
    avg_crlb_p1 = [np.mean(c) for c in p1_data['crlb_angle']]
    avg_rate_p2 = [np.mean(r) for r in p2_data['rates']]
    avg_crlb_p2 = [np.mean(c) for c in p2_data['crlb_angle']]
    
    axes[1, 0].scatter(avg_crlb_p1, avg_rate_p1, alpha=0.5, label='P1 (Sensing)', color='blue', s=50)
    axes[1, 0].scatter(avg_crlb_p2, avg_rate_p2, alpha=0.5, label='P2 (Comm)', color='red', s=50)
    axes[1, 0].set_xlabel('Mean Angle CRLB (rad²)')
    axes[1, 0].set_ylabel('Mean Sum Rate (bits/s/Hz)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title('Rate-CRLB Tradeoff')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Mean metrics comparison
    metrics_p1 = [
        np.mean(avg_rate_p1),
        np.mean(avg_crlb_p1) * 1e5,  # Scale for visibility
        np.mean([np.mean(r) for r in p1_data['rho']]) * 10
    ]
    metrics_p2 = [
        np.mean(avg_rate_p2),
        np.mean(avg_crlb_p2) * 1e5,
        np.mean([np.mean(r) for r in p2_data['rho']]) * 10
    ]
    
    x = np.arange(3)
    width = 0.35
    
    axes[1, 1].bar(x - width/2, metrics_p1, width, label='P1 (Sensing)', color='blue', alpha=0.8)
    axes[1, 1].bar(x + width/2, metrics_p2, width, label='P2 (Comm)', color='red', alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Rate', 'CRLB×10⁵', 'ρ×10'])
    axes[1, 1].set_ylabel('Metric Value (scaled)')
    axes[1, 1].set_title('Average Metrics Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Episode trajectory comparison
    ep_idx = 0
    axes[1, 2].plot(p1_data['timesteps'][ep_idx], p1_data['rho'][ep_idx], 
                   label='P1 (Sensing)', color='blue', linewidth=2)
    axes[1, 2].plot(p2_data['timesteps'][ep_idx], p2_data['rho'][ep_idx],
                   label='P2 (Comm)', color='red', linewidth=2)
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Sensing Fraction (ρ)')
    axes[1, 2].set_title('Example Trajectory Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Problem 1 vs Problem 2: Comprehensive Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'D3_p1_vs_p2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive ISAC analysis plots')
    parser.add_argument('--p1-config', type=str, required=True, help='Path to P1 config')
    parser.add_argument('--p1-model', type=str, required=True, help='Path to P1 model')
    parser.add_argument('--p1-vec-normalize', type=str, default=None, help='Path to P1 vec_normalize')
    parser.add_argument('--p2-config', type=str, required=True, help='Path to P2 config')
    parser.add_argument('--p2-model', type=str, required=True, help='Path to P2 model')
    parser.add_argument('--p2-vec-normalize', type=str, default=None, help='Path to P2 vec_normalize')
    parser.add_argument('--save-dir', type=str, default='experiments/comprehensive_plots', help='Directory to save plots')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per scenario')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create save directory
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for P1 and P2
    (save_path / 'p1').mkdir(exist_ok=True)
    (save_path / 'p2').mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Comprehensive ISAC Analysis")
    print("=" * 80)
    
    # Load configs
    with open(args.p1_config, 'r') as f:
        p1_config = yaml.safe_load(f)
    with open(args.p2_config, 'r') as f:
        p2_config = yaml.safe_load(f)
    
    # Load models
    print("\nLoading models...")
    # Remove .zip extension if present (stable_baselines3 adds it automatically)
    p1_model_path = args.p1_model.replace('.zip', '') if args.p1_model.endswith('.zip') else args.p1_model
    p2_model_path = args.p2_model.replace('.zip', '') if args.p2_model.endswith('.zip') else args.p2_model
    
    p1_model = PPO.load(p1_model_path)
    p2_model = PPO.load(p2_model_path)
    
    # Note: VecNormalize will be loaded per environment during evaluation
    # We store the paths for now
    p1_vec_norm_path = args.p1_vec_normalize
    p2_vec_norm_path = args.p2_vec_normalize
    
    # ========================================================================
    # Category A: Constant Velocity
    # ========================================================================
    print("\n" + "=" * 80)
    print("Category A: Constant Velocity Scenarios")
    print("=" * 80)
    
    velocity_configs = {
        'low': (10.0, 12.0),
        'mid': (15.0, 17.0),
        'high': (20.0, 22.0)
    }
    
    p1_velocity_data = {}
    p2_velocity_data = {}
    
    for speed, v_range in velocity_configs.items():
        print(f"\nEvaluating {speed} velocity...")
        
        # P1 evaluation
        env = create_env(p1_config, mobility_type='constant_velocity', speed_range=v_range)
        if p1_vec_norm_path:
            env = VecNormalize.load(p1_vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        p1_velocity_data[speed] = run_evaluation_episodes(p1_model, env, None, args.episodes, args.seed)
        env.close()
        
        # P2 evaluation
        env = create_env(p2_config, mobility_type='constant_velocity', speed_range=v_range)
        if p2_vec_norm_path:
            env = VecNormalize.load(p2_vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        p2_velocity_data[speed] = run_evaluation_episodes(p2_model, env, None, args.episodes, args.seed)
        env.close()
    
    print("\nGenerating Category A plots...")
    plot_A1_time_evolution(p1_velocity_data, save_path / 'p1')
    plot_A1_time_evolution(p2_velocity_data, save_path / 'p2')
    
    plot_A2_tradeoff_curves(p1_velocity_data, save_path / 'p1')
    plot_A2_tradeoff_curves(p2_velocity_data, save_path / 'p2')
    
    velocity_values = {'low': 11, 'mid': 16, 'high': 21}
    plot_A3_metrics_vs_velocity(p1_velocity_data, velocity_values, save_path / 'p1')
    plot_A3_metrics_vs_velocity(p2_velocity_data, velocity_values, save_path / 'p2')
    
    plot_A4_heatmap_distance_speed(p1_velocity_data, save_path / 'p1')
    plot_A4_heatmap_distance_speed(p2_velocity_data, save_path / 'p2')
    
    # ========================================================================
    # Category B: Acceleration
    # ========================================================================
    print("\n" + "=" * 80)
    print("Category B: Acceleration Scenarios")
    print("=" * 80)
    
    accel_configs = {
        'low': {'a_min': -0.5, 'a_max': 0.5},
        'mid': {'a_min': -1.0, 'a_max': 1.0},
        'high': {'a_min': -2.0, 'a_max': 2.0}
    }
    
    p1_accel_data = {}
    p2_accel_data = {}
    
    for accel, params in accel_configs.items():
        print(f"\nEvaluating {accel} acceleration...")
        
        # Update config with acceleration params
        p1_config_accel = p1_config.copy()
        p1_config_accel['mobility_config'] = params
        p2_config_accel = p2_config.copy()
        p2_config_accel['mobility_config'] = params
        
        # P1 evaluation
        env = create_env(p1_config_accel, mobility_type='acceleration')
        if p1_vec_norm_path:
            env = VecNormalize.load(p1_vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        p1_accel_data[accel] = run_evaluation_episodes(p1_model, env, None, args.episodes, args.seed)
        env.close()
        
        # P2 evaluation
        env = create_env(p2_config_accel, mobility_type='acceleration')
        if p2_vec_norm_path:
            env = VecNormalize.load(p2_vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        p2_accel_data[accel] = run_evaluation_episodes(p2_model, env, None, args.episodes, args.seed)
        env.close()
    
    print("\nGenerating Category B plots...")
    plot_B1_acceleration_response(p1_accel_data['high'], save_path / 'p1')
    plot_B1_acceleration_response(p2_accel_data['high'], save_path / 'p2')
    
    plot_B2_tracking_vs_acceleration(p1_accel_data, save_path / 'p1')
    plot_B2_tracking_vs_acceleration(p2_accel_data, save_path / 'p2')
    
    # ========================================================================
    # Category C: Lane Changes
    # ========================================================================
    print("\n" + "=" * 80)
    print("Category C: Lane Change Scenarios")
    print("=" * 80)
    
    print("\nEvaluating lane changes...")
    
    # P1 evaluation
    env = create_env(p1_config, mobility_type='lane_change')
    if p1_vec_norm_path:
        env = VecNormalize.load(p1_vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    p1_lane_data = run_evaluation_episodes(p1_model, env, None, args.episodes, args.seed)
    env.close()
    
    # P2 evaluation
    env = create_env(p2_config, mobility_type='lane_change')
    if p2_vec_norm_path:
        env = VecNormalize.load(p2_vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    p2_lane_data = run_evaluation_episodes(p2_model, env, None, args.episodes, args.seed)
    env.close()
    
    print("\nGenerating Category C plots...")
    plot_C1_lane_change_timeline(p1_lane_data, save_path / 'p1')
    plot_C1_lane_change_timeline(p2_lane_data, save_path / 'p2')
    
    # ========================================================================
    # Category D: Cross-Scenario Summaries
    # ========================================================================
    print("\n" + "=" * 80)
    print("Category D: Cross-Scenario Summaries")
    print("=" * 80)
    
    # Combine all scenarios for P1 and P2
    p1_all_scenarios = {
        'Const Vel (Low)': p1_velocity_data['low'],
        'Const Vel (Mid)': p1_velocity_data['mid'],
        'Const Vel (High)': p1_velocity_data['high'],
        'Accel (Low)': p1_accel_data['low'],
        'Accel (Mid)': p1_accel_data['mid'],
        'Accel (High)': p1_accel_data['high'],
        'Lane Change': p1_lane_data
    }
    
    p2_all_scenarios = {
        'Const Vel (Low)': p2_velocity_data['low'],
        'Const Vel (Mid)': p2_velocity_data['mid'],
        'Const Vel (High)': p2_velocity_data['high'],
        'Accel (Low)': p2_accel_data['low'],
        'Accel (Mid)': p2_accel_data['mid'],
        'Accel (High)': p2_accel_data['high'],
        'Lane Change': p2_lane_data
    }
    
    print("\nGenerating Category D plots...")
    plot_D1_scenario_comparison(p1_all_scenarios, save_path / 'p1')
    plot_D1_scenario_comparison(p2_all_scenarios, save_path / 'p2')
    
    # Combine all data for CDF analysis
    p1_all_data = {
        'rho': [],
        'rates': [],
        'crlb_angle': [],
        'crlb_range': []
    }
    p2_all_data = {
        'rho': [],
        'rates': [],
        'crlb_angle': [],
        'crlb_range': []
    }
    
    for scenario_data in p1_all_scenarios.values():
        for key in p1_all_data.keys():
            p1_all_data[key].extend(scenario_data[key])
    
    for scenario_data in p2_all_scenarios.values():
        for key in p2_all_data.keys():
            p2_all_data[key].extend(scenario_data[key])
    
    plot_D2_cdf_analysis(p1_all_data, save_path / 'p1')
    plot_D2_cdf_analysis(p2_all_data, save_path / 'p2')
    
    # P1 vs P2 direct comparison
    plot_D3_p1_vs_p2_comparison(p1_all_data, p2_all_data, save_path)
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {save_path}")
    print("=" * 80)
    print("\nPlot Summary:")
    print("  Category A (Constant Velocity): 4 plots × 2 problems = 8 plots")
    print("  Category B (Acceleration): 2 plots × 2 problems = 4 plots")
    print("  Category C (Lane Changes): 1 plot × 2 problems = 2 plots")
    print("  Category D (Cross-Scenario): 2 plots × 2 problems + 1 comparison = 5 plots")
    print("  Total: 19 comprehensive analysis plots")
    print("\nDone!")


if __name__ == '__main__':
    main()