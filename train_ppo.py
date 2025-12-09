"""
PPO training script for ISAC RL agent.
"""
import numpy as np
import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.vehicular_isac_env import load_env_from_config


def make_env(config_path: str, seed: int = 0):
    """Create and wrap environment."""
    def _init():
        env = load_env_from_config(config_path)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_ppo(
    config_path: str,
    total_timesteps: int = 100000,
    n_steps: int = 1024,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    n_epochs: int = 10,
    gamma: float = 0.99,
    ent_coef: float = 1e-3,
    clip_range: float = 0.2,
    save_dir: str = 'experiments/ppo_results',
    seed: int = 42,
    device: str = 'cpu'
):
    """
    Train PPO agent on ISAC environment.
    
    Args:
        config_path: Path to environment config
        total_timesteps: Total training steps
        n_steps: Steps per rollout
        batch_size: Minibatch size
        learning_rate: Learning rate
        n_epochs: Number of epochs per update
        gamma: Discount factor
        ent_coef: Entropy coefficient
        clip_range: PPO clip range
        save_dir: Directory to save results
        seed: Random seed
        device: Device to use ('cpu' or 'cuda')
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create save directory and clean up any existing problematic files
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories (remove if they exist as files)
    for subdir in ['checkpoints', 'best_model', 'eval_logs', 'tensorboard']:
        subdir_path = save_path / subdir
        # If it exists as a file (not directory), remove it
        if subdir_path.exists() and not subdir_path.is_dir():
            subdir_path.unlink()
        subdir_path.mkdir(exist_ok=True)
    
    # Load config to get objective mode
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    objective_mode = config.get('objective_mode', 1)
    
    print(f"Training PPO for Objective Mode {objective_mode}")
    print(f"Config: {config_path}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {device}")
    print(f"Save directory: {save_path}")
    
    # Create training environment
    env = DummyVecEnv([make_env(config_path, seed)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(config_path, seed + 1000)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards during evaluation
        clip_obs=10.0,
        training=False
    )
    
    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(save_path / 'checkpoints'),
        name_prefix='ppo_model'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / 'best_model'),
        log_path=str(save_path / 'eval_logs'),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Create PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log=str(save_path / 'tensorboard'),
        device=device,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        )
    )
    
    print("\nStarting training...")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=f'ppo_mode{objective_mode}'
    )
    
    # Save final model
    model.save(str(save_path / 'final_model'))
    env.save(str(save_path / 'vec_normalize.pkl'))
    
    print(f"\nTraining complete! Model saved to {save_path}")
    
    return model, env


def main():
    parser = argparse.ArgumentParser(description='Train PPO for ISAC')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100000,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=1024,
        help='Steps per rollout'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Minibatch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='experiments/ppo_results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    train_ppo(
        config_path=args.config,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        save_dir=args.save_dir,
        seed=args.seed,
        device=args.device
    )


if __name__ == '__main__':
    main()