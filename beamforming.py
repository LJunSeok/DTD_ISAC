"""
Beamforming utilities for ISAC.
Implements parametric sensing and communication beamformers.
"""
import numpy as np
from typing import Tuple
from .channel_models import array_response_tx


def create_sensing_beam(
    phi_sens: float,
    p_sens: float,
    P_tot: float,
    N_t: int
) -> np.ndarray:
    """
    Create sensing beamformer vector.
    
    Args:
        phi_sens: Sensing beam angle (radians)
        p_sens: Sensing power fraction [0, 1]
        P_tot: Total power budget
        N_t: Number of transmit antennas
        
    Returns:
        Sensing beamformer vector of shape (N_t, 1)
    """
    a_t = array_response_tx(phi_sens, N_t)
    f_sens = np.sqrt(P_tot * p_sens) * a_t
    return f_sens.reshape(-1, 1)


def create_communication_beams(
    phi_comm: np.ndarray,
    p_comm: np.ndarray,
    P_tot: float,
    N_t: int
) -> np.ndarray:
    """
    Create communication beamformer matrix.
    
    Args:
        phi_comm: Communication beam angles for each user, shape (K,)
        p_comm: Communication power fractions for each user, shape (K,)
        P_tot: Total power budget
        N_t: Number of transmit antennas
        
    Returns:
        Communication beamformer matrix of shape (N_t, K)
    """
    K = len(phi_comm)
    F_comm = np.zeros((N_t, K), dtype=np.complex128)
    
    for k in range(K):
        a_t = array_response_tx(phi_comm[k], N_t)
        F_comm[:, k] = np.sqrt(P_tot * p_comm[k]) * a_t
    
    return F_comm


def compute_sinr(
    H: np.ndarray,
    F_comm: np.ndarray,
    sigma_comm_sq: float
) -> np.ndarray:
    """
    Compute SINR for each user.
    
    Args:
        H: Channel matrix for one user, shape (N_r, N_t) or (N_t,) for M=1
        F_comm: Communication beamformer matrix, shape (N_t, K)
        sigma_comm_sq: Noise power
        
    Returns:
        SINR values for all K users, shape (K,)
    """
    K = F_comm.shape[1]
    sinr = np.zeros(K)
    
    for k in range(K):
        # Effective channel for user k
        if H.ndim == 1:
            # Scalar receiver (M=1)
            h_eff = H
        else:
            # Vector receiver (M>1), use first receive antenna for simplicity
            h_eff = H[0, :]
        
        # Desired signal power
        S_k = np.abs(np.vdot(h_eff, F_comm[:, k]))**2
        
        # Interference power
        I_k = 0.0
        for j in range(K):
            if j != k:
                I_k += np.abs(np.vdot(h_eff, F_comm[:, j]))**2
        
        # SINR
        sinr[k] = S_k / (I_k + sigma_comm_sq)
    
    return sinr


def compute_sum_rate(
    H_batch: np.ndarray,
    F_comm: np.ndarray,
    rho: float,
    sigma_comm_sq: float
) -> Tuple[float, np.ndarray]:
    """
    Compute sum rate for all users.
    
    Args:
        H_batch: Channel matrices for all users, shape (K, N_r, N_t) or (K, N_t)
        F_comm: Communication beamformer matrix, shape (N_t, K)
        rho: Sensing time fraction
        sigma_comm_sq: Noise power
        
    Returns:
        Tuple of:
            - Sum rate across all users
            - Individual rates for each user, shape (K,)
    """
    K = H_batch.shape[0]
    rates = np.zeros(K)
    
    for k in range(K):
        # Compute SINR for user k
        sinr_k = compute_sinr(H_batch[k], F_comm, sigma_comm_sq)[k]
        
        # Rate accounting for communication time fraction
        rates[k] = (1 - rho) * np.log2(1 + sinr_k)
    
    sum_rate = np.sum(rates)
    return sum_rate, rates


def parse_action_to_beamformers(
    action: np.ndarray,
    thetas: np.ndarray,
    P_tot: float,
    N_t: int,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
    delta_phi_max: float = np.pi / 6  # 30 degrees
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Parse RL action vector into beamforming parameters.
    
    Action format: [raw_rho, raw_p_sens, delta_phi_1, ..., delta_phi_K, u_1, ..., u_K]
    
    Args:
        action: Raw action from RL agent
        thetas: Geometric angles to vehicles, shape (K,)
        P_tot: Total power budget
        N_t: Number of transmit antennas
        rho_min: Minimum sensing time fraction
        rho_max: Maximum sensing time fraction
        delta_phi_max: Maximum angle offset (radians)
        
    Returns:
        Tuple of:
            - rho: Sensing time fraction
            - p_sens: Sensing power fraction
            - phi_sens: Sensing beam angle
            - phi_comm: Communication beam angles, shape (K,)
            - p_comm: Communication power fractions, shape (K,)
    """
    K = len(thetas)
    
    # Parse action components
    raw_rho = action[0]
    raw_p_sens = action[1]
    delta_phis = action[2:2+K]
    logits = action[2+K:2+2*K]
    
    # Map rho via sigmoid to [rho_min, rho_max]
    rho = rho_min + (rho_max - rho_min) / (1 + np.exp(-raw_rho))
    
    # Map p_sens via sigmoid to [0, 1]
    p_sens = 1.0 / (1 + np.exp(-raw_p_sens))
    
    # Clip delta_phis and add to geometric angles
    delta_phis = np.clip(delta_phis, -delta_phi_max, delta_phi_max)
    phi_comm = thetas + delta_phis
    
    # Sensing angle: weighted average of vehicle angles (simple heuristic)
    phi_sens = np.mean(thetas)
    
    # Convert logits to power fractions via softmax
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    p_comm_raw = exp_logits / np.sum(exp_logits)
    
    # Scale by remaining power after sensing
    p_comm = (1 - p_sens) * p_comm_raw
    
    return rho, p_sens, phi_sens, phi_comm, p_comm