"""
Fixed-rho baseline policies for comparison.
"""
import numpy as np
from typing import Tuple


class FixedRhoPolicy:
    """
    Baseline policy with fixed time division ratio.
    Uses geometric beamforming and equal power allocation.
    """
    
    def __init__(
        self,
        rho_fixed: float,
        N_t: int,
        K: int,
        P_tot: float = 1.0,
        p_sens: float = 0.3
    ):
        """
        Args:
            rho_fixed: Fixed sensing time fraction
            N_t: Number of transmit antennas
            K: Number of users
            P_tot: Total power
            p_sens: Fixed sensing power fraction
        """
        self.rho_fixed = rho_fixed
        self.N_t = N_t
        self.K = K
        self.P_tot = P_tot
        self.p_sens = p_sens
    
    def get_action(self, observation: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """
        Generate action based on fixed policy.
        
        Args:
            observation: Current observation (not used in fixed policy)
            thetas: Geometric angles to vehicles
            
        Returns:
            Action vector
        """
        # Action format: [raw_rho, raw_p_sens, delta_phi_1, ..., delta_phi_K, u_1, ..., u_K]
        action_dim = 2 + 2 * self.K
        action = np.zeros(action_dim)
        
        # Map rho_fixed to raw_rho (inverse sigmoid)
        # rho = rho_min + (rho_max - rho_min) / (1 + exp(-raw_rho))
        # Assume rho_min=0.05, rho_max=0.95
        rho_min, rho_max = 0.05, 0.95
        rho_norm = (self.rho_fixed - rho_min) / (rho_max - rho_min)
        rho_norm = np.clip(rho_norm, 0.01, 0.99)
        raw_rho = -np.log((1.0 / rho_norm) - 1.0)
        action[0] = raw_rho
        
        # Map p_sens to raw_p_sens (inverse sigmoid)
        p_sens_clip = np.clip(self.p_sens, 0.01, 0.99)
        raw_p_sens = -np.log((1.0 / p_sens_clip) - 1.0)
        action[1] = raw_p_sens
        
        # No angle offsets (geometric beamforming)
        action[2:2+self.K] = 0.0
        
        # Equal power allocation (uniform logits)
        action[2+self.K:2+2*self.K] = 0.0
        
        return action


def evaluate_fixed_rho_policy(
    env,
    rho_fixed: float,
    n_episodes: int = 10,
    seed: int = 42
) -> dict:
    """
    Evaluate a fixed-rho policy on the environment.
    
    Args:
        env: ISAC environment
        rho_fixed: Fixed sensing time fraction
        n_episodes: Number of evaluation episodes
        seed: Random seed
        
    Returns:
        Dictionary with evaluation metrics
    """
    policy = FixedRhoPolicy(
        rho_fixed=rho_fixed,
        N_t=env.N_t,
        K=env.K,
        P_tot=env.P_tot
    )
    
    episode_returns = []
    episode_rates = []
    episode_crlbs_theta = []
    episode_crlbs_r = []
    constraint_violations_R = []
    constraint_violations_theta = []
    constraint_violations_r = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        truncated = False
        episode_return = 0.0
        
        rates = []
        crlbs_theta = []
        crlbs_r = []
        
        while not (done or truncated):
            # Get thetas from current positions
            from envs.channel_models import generate_channels_batch
            _, thetas, _, _ = generate_channels_batch(
                env.positions, env.N_t, env.N_r,
                env.r0, env.beta0, env.alpha
            )
            
            action = policy.get_action(obs, thetas)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_return += reward
            rates.append(info['R_sum'])
            crlbs_theta.extend(info['crlbs_theta'].tolist())
            crlbs_r.extend(info['crlbs_r'].tolist())
        
        episode_returns.append(episode_return)
        episode_rates.append(np.mean(rates))
        episode_crlbs_theta.append(np.mean(crlbs_theta))
        episode_crlbs_r.append(np.mean(crlbs_r))
        
        # Check constraint violations
        avg_rate = np.mean(rates)
        avg_crlb_theta = np.mean(crlbs_theta)
        avg_crlb_r = np.mean(crlbs_r)
        
        constraint_violations_R.append(1 if avg_rate < env.R_th else 0)
        constraint_violations_theta.append(1 if avg_crlb_theta > env.eps_theta else 0)
        constraint_violations_r.append(1 if avg_crlb_r > env.eps_r else 0)
    
    results = {
        'rho_fixed': rho_fixed,
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_rate': np.mean(episode_rates),
        'std_rate': np.std(episode_rates),
        'mean_crlb_theta': np.mean(episode_crlbs_theta),
        'std_crlb_theta': np.std(episode_crlbs_theta),
        'mean_crlb_r': np.mean(episode_crlbs_r),
        'std_crlb_r': np.std(episode_crlbs_r),
        'violation_rate_R': np.mean(constraint_violations_R),
        'violation_rate_theta': np.mean(constraint_violations_theta),
        'violation_rate_r': np.mean(constraint_violations_r)
    }
    
    return results