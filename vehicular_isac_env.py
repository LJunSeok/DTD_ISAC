"""
Vehicular ISAC RL Environment.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import yaml

from .channel_models import generate_channels_batch
from .mobility_models import create_mobility_model, MobilityModel
from .beamforming import (
    create_sensing_beam,
    create_communication_beams,
    compute_sum_rate,
    parse_action_to_beamformers
)
from .sensing_metrics import compute_sensing_metrics_batch


class VehicularISACEnv(gym.Env):
    """
    Gymnasium environment for ISAC vehicular networks with dynamic time division.
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ISAC environment.
        
        Args:
            config: Configuration dictionary with all parameters
        """
        super().__init__()
        
        # System parameters
        self.N_t = config.get('N_t', 8)
        self.N_r = config.get('N_r', 4)
        self.K = config.get('K', 2)
        self.M = config.get('M', 1)  # RX antennas per vehicle (start with 1)
        
        # Time parameters
        self.T = config.get('T', 10.0)  # Total time (seconds)
        self.N = config.get('N', 100)  # Number of slots
        self.dt = self.T / self.N  # Slot duration
        
        # Power and noise
        self.P_tot = float(config.get('P_tot', 1.0))
        self.sigma_sens_sq = float(config.get('sigma_sens_sq', 1e-10))
        self.sigma_comm_sq = float(config.get('sigma_comm_sq', 1e-10))
        
        # Sensing parameters
        self.tau = config.get('tau', 4)  # Number of sensing echoes
        self.D_ap = float(config.get('D_ap', self.N_t * 0.5))  # Effective aperture
        self.B = float(config.get('B', 1e6))  # Bandwidth (Hz) - ensure float
        self.c_snr = float(config.get('c_snr', 1.0))
        self.c_theta = float(config.get('c_theta', 1.0))
        self.c_r = float(config.get('c_r', 1e-6))
        
        # Channel parameters
        self.r0 = float(config.get('r0', 1.0))
        self.beta0 = float(config.get('beta0', 1.0))
        self.alpha = float(config.get('alpha', 2.5))
        
        # Time division constraints
        self.rho_min = config.get('rho_min', 0.05)
        self.rho_max = config.get('rho_max', 0.95)
        
        # Mobility
        self.mobility_type = config.get('mobility_type', 'constant_velocity')
        self.mobility_config = config.get('mobility_config', {})
        self.lane_positions = np.array(config.get('lane_positions', [3.0, 6.0, 9.0]))
        
        # Initial positions
        self.x_init_range = config.get('x_init_range', [-30.0, -50.0])
        self.v_init_range = config.get('v_init_range', [10.0, 20.0])
        
        # Objective mode
        self.objective_mode = config.get('objective_mode', 1)  # 1 or 2
        
        # Thresholds and weights
        self.R_th = float(config.get('R_th', 5.0))  # Rate threshold for Problem 1
        self.eps_theta = float(config.get('eps_theta', 1e-4))  # Angle CRLB threshold
        self.eps_r = float(config.get('eps_r', 1e-3))  # Range CRLB threshold
        
        # Reward weights
        self.alpha_theta = float(config.get('alpha_theta', 0.5))
        self.alpha_r = float(config.get('alpha_r', 0.5))
        self.lambda_R = float(config.get('lambda_R', 3.0))
        self.lambda_theta = float(config.get('lambda_theta', 3.0))
        self.lambda_r = float(config.get('lambda_r', 3.0))
        
        # Normalization statistics (loaded from calibration)
        calib_stats = config.get('calibration_stats', {})
        self.q_R_10 = float(calib_stats.get('q_R_10', 0.0))
        self.q_R_90 = float(calib_stats.get('q_R_90', 10.0))
        self.q_theta_10 = float(calib_stats.get('q_theta_10', -5.0))
        self.q_theta_90 = float(calib_stats.get('q_theta_90', -2.0))
        self.q_r_10 = float(calib_stats.get('q_r_10', -4.0))
        self.q_r_90 = float(calib_stats.get('q_r_90', -1.0))
        
        # Action space: [raw_rho, raw_p_sens, delta_phi_1, ..., delta_phi_K, u_1, ..., u_K]
        action_dim = 2 + 2 * self.K
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Observation space: per vehicle (x, y, vx) + aggregated metrics + previous actions
        # Per vehicle: 3 features
        # Global: prev_rho, prev_p_sens, prev_R_sum, prev_CRLB_theta_mean, prev_CRLB_r_mean
        obs_dim = 3 * self.K + 5
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.positions = np.zeros((self.K, 2))  # (x, y) for each vehicle
        self.velocities = np.zeros((self.K, 2))  # (vx, vy) for each vehicle
        self.mobility_models: list[MobilityModel] = []
        
        # Previous metrics for observation
        self.prev_rho = 0.5
        self.prev_p_sens = 0.5
        self.prev_R_sum = 0.0
        self.prev_crlb_theta_mean = 0.0
        self.prev_crlb_r_mean = 0.0
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_rhos = []
        self.episode_rates = []
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Initialize vehicle positions
        for k in range(self.K):
            x_init = self.np_random.uniform(*self.x_init_range)
            lane_idx = self.np_random.integers(0, len(self.lane_positions))
            y_init = self.lane_positions[lane_idx]
            self.positions[k] = [x_init, y_init]
            
            vx_init = self.np_random.uniform(*self.v_init_range)
            self.velocities[k] = [vx_init, 0.0]
        
        # Initialize mobility models
        self.mobility_models = []
        for k in range(self.K):
            model = create_mobility_model(
                self.mobility_type,
                self.dt,
                **self.mobility_config
            )
            # For lane change model, initialize lane
            if self.mobility_type == 'lane_change':
                model.initialize_lane(self.positions[k, 1])
            self.mobility_models.append(model)
        
        # Reset previous metrics
        self.prev_rho = 0.5
        self.prev_p_sens = 0.5
        self.prev_R_sum = 0.0
        self.prev_crlb_theta_mean = 0.0
        self.prev_crlb_r_mean = 0.0
        
        # Reset episode tracking
        self.episode_rewards = []
        self.episode_rhos = []
        self.episode_rates = []
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Generate channels
        H_batch, thetas, ranges, betas = generate_channels_batch(
            self.positions, self.N_t, self.N_r,
            self.r0, self.beta0, self.alpha
        )
        
        # Parse action into beamforming parameters
        rho, p_sens, phi_sens, phi_comm, p_comm = parse_action_to_beamformers(
            action, thetas, self.P_tot, self.N_t,
            self.rho_min, self.rho_max
        )
        
        # Create beamformers
        f_sens = create_sensing_beam(phi_sens, p_sens, self.P_tot, self.N_t)
        F_comm = create_communication_beams(phi_comm, p_comm, self.P_tot, self.N_t)
        
        # Compute sensing metrics
        snrs, crlbs_theta, crlbs_r = compute_sensing_metrics_batch(
            thetas, betas, f_sens, rho, self.dt, self.tau,
            self.P_tot, self.sigma_sens_sq, self.N_t, self.N_r,
            self.D_ap, self.B, self.c_snr, self.c_theta, self.c_r
        )
        
        # Compute communication metrics
        R_sum, rates = compute_sum_rate(H_batch, F_comm, rho, self.sigma_comm_sq)
        
        # Compute reward
        reward = self._compute_reward(R_sum, crlbs_theta, crlbs_r)
        
        # Update state
        self.prev_rho = rho
        self.prev_p_sens = p_sens
        self.prev_R_sum = R_sum
        self.prev_crlb_theta_mean = np.mean(crlbs_theta)
        self.prev_crlb_r_mean = np.mean(crlbs_r)
        
        # Update vehicle positions
        for k in range(self.K):
            x, y, vx, vy = self.mobility_models[k].update_position(
                self.positions[k, 0], self.positions[k, 1],
                self.velocities[k, 0], self.velocities[k, 1]
            )
            self.positions[k] = [x, y]
            self.velocities[k] = [vx, vy]
        
        # Episode tracking
        self.episode_rewards.append(reward)
        self.episode_rhos.append(rho)
        self.episode_rates.append(R_sum)
        
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.N
        truncated = False
        
        # Check if vehicles left simulation area
        if np.any(self.positions[:, 0] > 250.0) or np.any(self.positions[:, 0] < -250.0):
            truncated = True
        
        obs = self._get_observation()
        info = {
            'rho': rho,
            'R_sum': R_sum,
            'crlbs_theta': crlbs_theta,
            'crlbs_r': crlbs_r,
            'positions': self.positions.copy(),
            'episode_length': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs_list = []
        
        # Per-vehicle features (normalized)
        for k in range(self.K):
            x_norm = self.positions[k, 0] / 100.0  # Normalize by typical max distance
            y_norm = self.positions[k, 1] / 15.0   # Normalize by max lane position
            vx_norm = self.velocities[k, 0] / 30.0  # Normalize by typical max velocity
            obs_list.extend([x_norm, y_norm, vx_norm])
        
        # Global features
        obs_list.append(self.prev_rho)
        obs_list.append(self.prev_p_sens)
        obs_list.append(self.prev_R_sum / 20.0)  # Normalize
        obs_list.append(np.log10(self.prev_crlb_theta_mean + 1e-12) / 5.0)
        obs_list.append(np.log10(self.prev_crlb_r_mean + 1e-12) / 5.0)
        
        obs = np.array(obs_list, dtype=np.float32)
        obs = np.clip(obs, -10.0, 10.0)
        
        return obs
    
    def _compute_reward(
        self,
        R_sum: float,
        crlbs_theta: np.ndarray,
        crlbs_r: np.ndarray
    ) -> float:
        """Compute reward based on objective mode."""
        eps = 1e-6
        
        # Normalize sum rate
        R_hat = np.clip(
            (R_sum - self.q_R_10) / (self.q_R_90 - self.q_R_10 + eps),
            0.0, 1.5
        )
        
        # Normalize CRLBs
        c_theta_hat = np.zeros(self.K)
        c_r_hat = np.zeros(self.K)
        
        for k in range(self.K):
            log_crlb_theta = np.log10(crlbs_theta[k] + eps)
            log_eps_theta = np.log10(self.eps_theta)
            c_theta_hat[k] = np.clip(
                (log_crlb_theta - log_eps_theta) / (self.q_theta_90 - log_eps_theta + eps),
                0.0, 1.5
            )
            
            log_crlb_r = np.log10(crlbs_r[k] + eps)
            log_eps_r = np.log10(self.eps_r)
            c_r_hat[k] = np.clip(
                (log_crlb_r - log_eps_r) / (self.q_r_90 - log_eps_r + eps),
                0.0, 1.5
            )
        
        C_theta_hat = np.mean(c_theta_hat)
        C_r_hat = np.mean(c_r_hat)
        
        # Constraint violations
        v_R = max(0.0, (self.R_th - R_sum) / (self.R_th + eps))
        v_theta = max(0.0, C_theta_hat - 1.0)
        v_r = max(0.0, C_r_hat - 1.0)
        
        # Exploration bonus: encourage mid-range rho (helps avoid getting stuck at boundaries)
        rho_exploration_bonus = 0.1 * (1.0 - abs(self.prev_rho - 0.5) * 2.0)
        
        # Compute reward based on mode
        if self.objective_mode == 1:
            # Problem 1: minimize CRLB under rate constraint
            # Also add soft constraints on CRLB to prevent degenerate solutions
            reward = -(self.alpha_theta * C_theta_hat + self.alpha_r * C_r_hat) \
                     - self.lambda_R * v_R \
                     - 0.5 * self.lambda_theta * v_theta \
                     - 0.5 * self.lambda_r * v_r \
                     + rho_exploration_bonus
        else:
            # Problem 2: maximize rate under CRLB constraints
            reward = R_hat - (C_theta_hat +  C_r_hat) - self.lambda_theta * v_theta - self.lambda_r * v_r \
                     + rho_exploration_bonus
        
        # Clip reward
        reward = np.clip(reward, -5.0, 5.0)
        
        return float(reward)


def load_env_from_config(config_path: str) -> VehicularISACEnv:
    """Load environment from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return VehicularISACEnv(config)