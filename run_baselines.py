"""
Run baseline evaluations with fixed-rho policies.
"""
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.vehicular_isac_env import load_env_from_config
from baselines.fixed_rho_policies import evaluate_fixed_rho_policy


def run_baseline_sweep(
    config_path: str,
    rho_values: list = None,
    n_episodes: int = 20,
    seed: int = 42,
    save_dir: str = 'experiments/baselines'
):
    """
    Run baseline evaluation across multiple fixed rho values.
    
    Args:
        config_path: Path to environment config
        rho_values: List of fixed rho values to test
        n_episodes: Number of episodes per rho value
        seed: Random seed
        save_dir: Directory to save results
    """
    if rho_values is None:
        rho_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load environment
    env = load_env_from_config(config_path)
    
    print(f"Running baseline evaluation on {config_path}")
    print(f"Testing rho values: {rho_values}")
    print(f"Episodes per rho: {n_episodes}")
    
    results_list = []
    
    for rho in rho_values:
        print(f"\nEvaluating rho = {rho:.2f}...")
        
        results = evaluate_fixed_rho_policy(
            env, rho, n_episodes=n_episodes, seed=seed
        )
        
        print(f"  Mean return: {results['mean_return']:.3f} ± {results['std_return']:.3f}")
        print(f"  Mean rate: {results['mean_rate']:.3f} ± {results['std_rate']:.3f}")
        print(f"  Mean CRLB theta: {results['mean_crlb_theta']:.2e}")
        print(f"  Mean CRLB range: {results['mean_crlb_r']:.2e}")
        
        results_list.append(results)
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Save results
    config_name = Path(config_path).stem
    csv_path = save_path / f'baseline_results_{config_name}.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Find best fixed rho for each metric
    best_rate_idx = df['mean_rate'].idxmax()
    best_crlb_theta_idx = df['mean_crlb_theta'].idxmin()
    best_crlb_r_idx = df['mean_crlb_r'].idxmin()
    
    print("\n" + "="*80)
    print("BEST FIXED RHO VALUES")
    print("="*80)
    print(f"Best for rate: rho = {df.loc[best_rate_idx, 'rho_fixed']:.2f} "
          f"(rate = {df.loc[best_rate_idx, 'mean_rate']:.3f})")
    print(f"Best for angle CRLB: rho = {df.loc[best_crlb_theta_idx, 'rho_fixed']:.2f} "
          f"(CRLB = {df.loc[best_crlb_theta_idx, 'mean_crlb_theta']:.2e})")
    print(f"Best for range CRLB: rho = {df.loc[best_crlb_r_idx, 'rho_fixed']:.2f} "
          f"(CRLB = {df.loc[best_crlb_r_idx, 'mean_crlb_r']:.2e})")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run baseline evaluations')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--rho-values',
        type=float,
        nargs='+',
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help='List of fixed rho values to test'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of episodes per rho value'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='experiments/baselines',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    run_baseline_sweep(
        config_path=args.config,
        rho_values=args.rho_values,
        n_episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()