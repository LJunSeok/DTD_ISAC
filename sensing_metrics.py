"""
Sensing metrics and CRLB surrogate computations.
"""
import numpy as np
from typing import Tuple
from .channel_models import array_response_tx, array_response_rx


def compute_sensing_snr(
    theta_k: float,
    beta_k: float,
    f_sens: np.ndarray,
    rho: float,
    dt: float,
    tau: int,
    P_tot: float,
    sigma_sens_sq: float,
    N_t: int,
    N_r: int,
    c_snr: float = 1.0
) -> float:
    """
    Compute effective sensing SNR for a vehicle.
    
    Args:
        theta_k: Vehicle angle (radians)
        beta_k: Pathloss coefficient
        f_sens: Sensing beamformer, shape (N_t, 1) or (N_t,)
        rho: Sensing time fraction
        dt: Slot duration
        tau: Number of sensing echoes
        P_tot: Total power
        sigma_sens_sq: Sensing noise power
        N_t: Number of TX antennas
        N_r: Number of RX antennas
        c_snr: SNR scaling constant
        
    Returns:
        Effective sensing SNR
    """
    # Normalize sensing beam
    f_sens_flat = f_sens.flatten()
    f_sens_norm = f_sens_flat / (np.linalg.norm(f_sens_flat) + 1e-12)
    
    # Array responses
    a_t = array_response_tx(theta_k, N_t)
    a_r = array_response_rx(theta_k, N_r)
    
    # Array gain (simplified monostatic approximation)
    tx_gain = np.abs(np.vdot(a_t, f_sens_norm))**2
    rx_gain = 1.0  # Simplified, can use |a_r^H * a_r| = 1 for ULA
    
    # Effective SNR
    snr_sens = (c_snr * rho * dt * tau * beta_k * P_tot * tx_gain * rx_gain) / sigma_sens_sq
    
    return snr_sens


def compute_crlb_angle(
    snr_sens: float,
    D_ap: float,
    c_theta: float = 1.0
) -> float:
    """
    Compute CRLB surrogate for angle estimation.
    
    Args:
        snr_sens: Effective sensing SNR
        D_ap: Effective aperture (proportional to N_t * lambda)
        c_theta: Angle CRLB scaling constant
        
    Returns:
        CRLB for angle (radians^2)
    """
    if snr_sens < 1e-12:
        return 1e6  # Large penalty for very low SNR
    
    crlb_theta = c_theta / (snr_sens * D_ap**2)
    return crlb_theta


def compute_crlb_range(
    snr_sens: float,
    B: float,
    c_r: float = 1.0
) -> float:
    """
    Compute CRLB surrogate for range estimation.
    
    Args:
        snr_sens: Effective sensing SNR
        B: System bandwidth (Hz)
        c_r: Range CRLB scaling constant
        
    Returns:
        CRLB for range (meters^2)
    """
    if snr_sens < 1e-12:
        return 1e6  # Large penalty for very low SNR
    
    crlb_r = c_r / (snr_sens * B**2)
    return crlb_r


def compute_sensing_metrics_batch(
    thetas: np.ndarray,
    betas: np.ndarray,
    f_sens: np.ndarray,
    rho: float,
    dt: float,
    tau: int,
    P_tot: float,
    sigma_sens_sq: float,
    N_t: int,
    N_r: int,
    D_ap: float,
    B: float,
    c_snr: float = 1.0,
    c_theta: float = 1.0,
    c_r: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sensing metrics for all vehicles.
    
    Args:
        thetas: Vehicle angles, shape (K,)
        betas: Pathloss coefficients, shape (K,)
        f_sens: Sensing beamformer
        rho: Sensing time fraction
        dt: Slot duration
        tau: Number of sensing echoes
        P_tot: Total power
        sigma_sens_sq: Sensing noise power
        N_t: Number of TX antennas
        N_r: Number of RX antennas
        D_ap: Effective aperture
        B: Bandwidth
        c_snr: SNR scaling constant
        c_theta: Angle CRLB scaling constant
        c_r: Range CRLB scaling constant
        
    Returns:
        Tuple of:
            - SNRs for each vehicle, shape (K,)
            - Angle CRLBs for each vehicle, shape (K,)
            - Range CRLBs for each vehicle, shape (K,)
    """
    K = len(thetas)
    snrs = np.zeros(K)
    crlbs_theta = np.zeros(K)
    crlbs_r = np.zeros(K)
    
    for k in range(K):
        snrs[k] = compute_sensing_snr(
            thetas[k], betas[k], f_sens, rho, dt, tau,
            P_tot, sigma_sens_sq, N_t, N_r, c_snr
        )
        crlbs_theta[k] = compute_crlb_angle(snrs[k], D_ap, c_theta)
        crlbs_r[k] = compute_crlb_range(snrs[k], B, c_r)
    
    return snrs, crlbs_theta, crlbs_r