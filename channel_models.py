"""
Channel models for ISAC vehicular networks.
Implements LoS channel generation with pathloss and array responses.
"""
import numpy as np
from typing import Tuple


def array_response_tx(theta: float, N_t: int) -> np.ndarray:
    """
    Compute transmit array response vector for ULA with half-wavelength spacing.
    
    Args:
        theta: Angle in radians
        N_t: Number of transmit antennas
        
    Returns:
        Complex array response vector of shape (N_t,)
    """
    n = np.arange(N_t)
    return (1.0 / np.sqrt(N_t)) * np.exp(1j * np.pi * n * np.sin(theta))


def array_response_rx(theta: float, N_r: int) -> np.ndarray:
    """
    Compute receive array response vector for ULA with half-wavelength spacing.
    
    Args:
        theta: Angle in radians
        N_r: Number of receive antennas
        
    Returns:
        Complex array response vector of shape (N_r,)
    """
    n = np.arange(N_r)
    return (1.0 / np.sqrt(N_r)) * np.exp(1j * np.pi * n * np.sin(theta))


def compute_pathloss(r: float, r0: float, beta0: float, alpha: float) -> float:
    """
    Compute pathloss coefficient.
    
    Args:
        r: Distance to vehicle (meters)
        r0: Reference distance (meters)
        beta0: Pathloss at reference distance
        alpha: Pathloss exponent
        
    Returns:
        Pathloss coefficient beta
    """
    if r < 1.0:  # Avoid division by very small numbers
        r = 1.0
    return beta0 * (r0 / r) ** alpha


def generate_channel(
    x: float, 
    y: float, 
    N_t: int, 
    N_r: int,
    r0: float = 1.0,
    beta0: float = 1.0,
    alpha: float = 2.5
) -> Tuple[np.ndarray, float, float, float]:
    """
    Generate LoS channel for a vehicle at position (x, y).
    
    Args:
        x: Vehicle x-coordinate (meters)
        y: Vehicle y-coordinate (meters)
        N_t: Number of transmit antennas
        N_r: Number of receive antennas
        r0: Reference distance for pathloss
        beta0: Pathloss coefficient at reference distance
        alpha: Pathloss exponent
        
    Returns:
        Tuple of:
            - H: Channel matrix of shape (N_r, N_t)
            - theta: Angle of arrival/departure (radians)
            - r: Distance (meters)
            - beta: Pathloss coefficient
    """
    # Compute geometry
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    
    # Compute pathloss
    beta = compute_pathloss(r, r0, beta0, alpha)
    
    # Generate channel matrix
    a_t = array_response_tx(theta, N_t)
    a_r = array_response_rx(theta, N_r)
    
    H = np.sqrt(beta) * np.outer(a_r, a_t.conj())
    
    return H, theta, r, beta


def generate_channels_batch(
    positions: np.ndarray,
    N_t: int,
    N_r: int,
    r0: float = 1.0,
    beta0: float = 1.0,
    alpha: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate channels for multiple vehicles.
    
    Args:
        positions: Vehicle positions, shape (K, 2) where positions[k] = [x_k, y_k]
        N_t: Number of transmit antennas
        N_r: Number of receive antennas
        r0: Reference distance for pathloss
        beta0: Pathloss coefficient at reference distance
        alpha: Pathloss exponent
        
    Returns:
        Tuple of:
            - H_batch: Channel matrices, shape (K, N_r, N_t)
            - thetas: Angles, shape (K,)
            - ranges: Distances, shape (K,)
            - betas: Pathloss coefficients, shape (K,)
    """
    K = positions.shape[0]
    H_batch = np.zeros((K, N_r, N_t), dtype=np.complex128)
    thetas = np.zeros(K)
    ranges = np.zeros(K)
    betas = np.zeros(K)
    
    for k in range(K):
        H_batch[k], thetas[k], ranges[k], betas[k] = generate_channel(
            positions[k, 0], positions[k, 1], N_t, N_r, r0, beta0, alpha
        )
    
    return H_batch, thetas, ranges, betas