"""
Calibration script to gather statistics for reward normalization.
Runs random policy to collect percentiles of sum-rate and CRLBs.
"""
import numpy as np
import yaml
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.vehicular_isac_env import load_env_from_config


def calibrate_metrics(
    config_path: str,
    n_episodes: int = 50,
    seed: int = 42
) -> dict:
    """
    Run calibration to gather metric statistics.
    
    Args:
        config_path: Path to environment config file
        n_episodes: Number of episodes to run
        seed: Random seed
        
    Returns:
        Dictionary with percentile statistics
    """
    # Load environment
    env = load_env_from_config(config_path)
    
    # Storage for metrics
    R_sums = []
    crlbs_theta_all = []
    crlbs_r_all = []
    
    print(f"Running calibration with {n_episodes} episodes...")
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        
        while not (done or truncated):
            # Random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # Collect metrics
            R_sums.append(info['R_sum'])
            crlbs_theta_all.extend(info['crlbs_theta'].tolist())
            crlbs_r_all.extend(info['crlbs_r'].tolist())
        
        if (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{n_episodes} episodes")
    
    # Convert to arrays
    R_sums = np.array(R_sums)
    crlbs_theta_all = np.array(crlbs_theta_all)
    crlbs_r_all = np.array(crlbs_r_all)
    
    # Compute log-scale CRLBs
    log_crlbs_theta = np.log10(crlbs_theta_all + 1e-12)
    log_crlbs_r = np.log10(crlbs_r_all + 1e-12)
    
    # Compute percentiles
    q_R_10 = float(np.percentile(R_sums, 10))
    q_R_90 = float(np.percentile(R_sums, 90))
    q_theta_10 = float(np.percentile(log_crlbs_theta, 10))
    q_theta_90 = float(np.percentile(log_crlbs_theta, 90))
    q_r_10 = float(np.percentile(log_crlbs_r, 10))
    q_r_90 = float(np.percentile(log_crlbs_r, 90))
    
    stats = {
        'q_R_10': q_R_10,
        'q_R_90': q_R_90,
        'q_theta_10': q_theta_10,
        'q_theta_90': q_theta_90,
        'q_r_10': q_r_10,
        'q_r_90': q_r_90
    }
    
    print("\nCalibration Statistics:")
    print(f"  Sum Rate - 10th: {q_R_10:.3f}, 90th: {q_R_90:.3f}")
    print(f"  Log CRLB Theta - 10th: {q_theta_10:.3f}, 90th: {q_theta_90:.3f}")
    print(f"  Log CRLB Range - 10th: {q_r_10:.3f}, 90th: {q_r_90:.3f}")
    
    return stats


def update_config_with_stats(config_path: str, stats: dict) -> None:
    """Update config file with calibration statistics."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['calibration_stats'] = stats
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nUpdated config file: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate metric normalization')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of calibration episodes'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--all-configs',
        action='store_true',
        help='Calibrate all config files in configs/ directory'
    )
    
    args = parser.parse_args()
    
    if args.all_configs:
        # Calibrate all config files
        config_dir = Path('configs')
        config_files = list(config_dir.glob('*.yaml'))
        
        for config_file in config_files:
            print(f"\n{'='*60}")
            print(f"Calibrating: {config_file}")
            print(f"{'='*60}")
            
            stats = calibrate_metrics(
                str(config_file),
                n_episodes=args.episodes,
                seed=args.seed
            )
            update_config_with_stats(str(config_file), stats)
    else:
        # Calibrate single config
        stats = calibrate_metrics(
            args.config,
            n_episodes=args.episodes,
            seed=args.seed
        )
        update_config_with_stats(args.config, stats)
    
    print("\nCalibration complete!")


if __name__ == '__main__':
    main()