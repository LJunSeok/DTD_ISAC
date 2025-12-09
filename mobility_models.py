"""
Mobility models for vehicular scenarios.
Implements constant velocity, acceleration, and lane-change dynamics.
"""
import numpy as np
from typing import Tuple, Optional


class MobilityModel:
    """Base class for vehicle mobility models."""
    
    def __init__(self, dt: float):
        """
        Args:
            dt: Time step duration (seconds)
        """
        self.dt = dt
    
    def update_position(
        self, 
        x: float, 
        y: float, 
        vx: float, 
        vy: float,
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """
        Update vehicle position and velocity.
        
        Args:
            x: Current x position
            y: Current y position
            vx: Current x velocity
            vy: Current y velocity
            **kwargs: Additional state variables
            
        Returns:
            Tuple of (x_new, y_new, vx_new, vy_new)
        """
        raise NotImplementedError


class ConstantVelocityModel(MobilityModel):
    """Case A: Constant velocity motion."""
    
    def update_position(
        self, 
        x: float, 
        y: float, 
        vx: float, 
        vy: float,
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """Update position with constant velocity."""
        x_new = x + vx * self.dt
        y_new = y + vy * self.dt
        return x_new, y_new, vx, vy


class AccelerationModel(MobilityModel):
    """Case B: Constant acceleration motion."""
    
    def __init__(self, dt: float, ax: float = 0.0, ay: float = 0.0):
        """
        Args:
            dt: Time step duration
            ax: X-axis acceleration (m/s^2)
            ay: Y-axis acceleration (m/s^2)
        """
        super().__init__(dt)
        self.ax = ax
        self.ay = ay
    
    def update_position(
        self, 
        x: float, 
        y: float, 
        vx: float, 
        vy: float,
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """Update position and velocity with constant acceleration."""
        x_new = x + vx * self.dt + 0.5 * self.ax * self.dt**2
        y_new = y + vy * self.dt + 0.5 * self.ay * self.dt**2
        vx_new = vx + self.ax * self.dt
        vy_new = vy + self.ay * self.dt
        return x_new, y_new, vx_new, vy_new


class LaneChangeModel(MobilityModel):
    """Case C: Lane change dynamics."""
    
    def __init__(
        self, 
        dt: float, 
        lane_positions: np.ndarray,
        change_probability: float = 0.05,
        change_duration: float = 2.0
    ):
        """
        Args:
            dt: Time step duration
            lane_positions: Y-coordinates of lane centerlines
            change_probability: Probability of initiating lane change per step
            change_duration: Duration of lane change maneuver (seconds)
        """
        super().__init__(dt)
        self.lane_positions = lane_positions
        self.change_probability = change_probability
        self.change_duration = change_duration
        self.n_change_steps = int(change_duration / dt)
        
        # State tracking
        self.current_lane_idx: Optional[int] = None
        self.target_lane_idx: Optional[int] = None
        self.change_progress = 0
        self.is_changing = False
    
    def initialize_lane(self, y: float) -> None:
        """Initialize current lane based on y position."""
        distances = np.abs(self.lane_positions - y)
        self.current_lane_idx = np.argmin(distances)
        self.is_changing = False
        self.change_progress = 0
    
    def update_position(
        self, 
        x: float, 
        y: float, 
        vx: float, 
        vy: float,
        **kwargs
    ) -> Tuple[float, float, float, float]:
        """Update position with potential lane changes."""
        # Update x position (constant velocity in x)
        x_new = x + vx * self.dt
        
        # Initialize lane if not set
        if self.current_lane_idx is None:
            self.initialize_lane(y)
        
        # Handle lane change logic
        if not self.is_changing:
            # Check if we should initiate a lane change
            if np.random.rand() < self.change_probability:
                # Choose a random adjacent lane
                possible_lanes = []
                if self.current_lane_idx > 0:
                    possible_lanes.append(self.current_lane_idx - 1)
                if self.current_lane_idx < len(self.lane_positions) - 1:
                    possible_lanes.append(self.current_lane_idx + 1)
                
                if possible_lanes:
                    self.target_lane_idx = np.random.choice(possible_lanes)
                    self.is_changing = True
                    self.change_progress = 0
        
        # Execute lane change
        if self.is_changing:
            self.change_progress += 1
            
            # Smooth lane transition (sigmoid-like interpolation)
            progress_frac = self.change_progress / self.n_change_steps
            if progress_frac >= 1.0:
                # Lane change complete
                y_new = self.lane_positions[self.target_lane_idx]
                self.current_lane_idx = self.target_lane_idx
                self.is_changing = False
                self.change_progress = 0
            else:
                # Interpolate between lanes
                y_start = self.lane_positions[self.current_lane_idx]
                y_end = self.lane_positions[self.target_lane_idx]
                # Smooth S-curve
                smooth_progress = 3 * progress_frac**2 - 2 * progress_frac**3
                y_new = y_start + smooth_progress * (y_end - y_start)
        else:
            # Stay in current lane
            y_new = self.lane_positions[self.current_lane_idx]
        
        vy_new = 0.0  # Velocity in y is implicitly handled by lane position
        
        return x_new, y_new, vx, vy_new


def create_mobility_model(
    model_type: str,
    dt: float,
    **kwargs
) -> MobilityModel:
    """
    Factory function to create mobility models.
    
    Args:
        model_type: One of 'constant_velocity', 'acceleration', 'lane_change'
        dt: Time step duration
        **kwargs: Additional parameters for specific models
        
    Returns:
        MobilityModel instance
    """
    if model_type == 'constant_velocity':
        return ConstantVelocityModel(dt)
    elif model_type == 'acceleration':
        ax = kwargs.get('ax', 0.0)
        ay = kwargs.get('ay', 0.0)
        return AccelerationModel(dt, ax, ay)
    elif model_type == 'lane_change':
        lane_positions = kwargs.get('lane_positions', np.array([3.0, 6.0, 9.0]))
        change_prob = kwargs.get('change_probability', 0.05)
        change_dur = kwargs.get('change_duration', 2.0)
        return LaneChangeModel(dt, lane_positions, change_prob, change_dur)
    else:
        raise ValueError(f"Unknown mobility model type: {model_type}")