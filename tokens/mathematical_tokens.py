"""
Mathematical Token Definitions for Embodied Intelligence
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from torch import nn

@dataclass
class PhysicalDynamicsToken:
    """Token representing physical system dynamics"""
    # Kinematic state
    positions: torch.Tensor          # [batch, n_joints]
    velocities: torch.Tensor         # [batch, n_joints]  
    accelerations: torch.Tensor      # [batch, n_joints]
    
    # Dynamic properties
    masses: torch.Tensor             # [batch, n_links]
    inertias: torch.Tensor           # [batch, n_links, 3, 3]
    center_of_mass: torch.Tensor     # [batch, 3]
    
    # Constraints
    joint_limits: torch.Tensor       # [batch, n_joints, 2]
    contact_constraints: torch.Tensor # [batch, n_contacts, 4]
    
    # Energy
    kinetic_energy: torch.Tensor     # [batch, 1]
    potential_energy: torch.Tensor   # [batch, 1]
    power_consumption: torch.Tensor  # [batch, 1]
    
    # Metadata
    timestamp: float
    control_mode: str
    safety_critical: bool

@dataclass  
class ElectromagneticToken:
    """Token representing electromagnetic perception"""
    # Spectral data
    spectral_bands: torch.Tensor     # [batch, bands, height, width]
    intensity_maps: torch.Tensor     # [batch, bands, height, width]
    wavelength_ranges: torch.Tensor  # [batch, bands, 2]
    
    # Field measurements
    magnetic_field: torch.Tensor     # [batch, 3]  # B_x, B_y, B_z
    electric_field: torch.Tensor     # [batch, 3]  # E_x, E_y, E_z
    field_gradients: torch.Tensor    # [batch, 3, 3]
    
    # Polarization
    stokes_parameters: torch.Tensor  # [batch, 4]  # I, Q, U, V
    polarization_angles: torch.Tensor # [batch, 2] # theta, phi
    
    # Navigation
    geomagnetic_vector: torch.Tensor # [batch, 3]
    inclination: torch.Tensor        # [batch, 1]
    declination: torch.Tensor        # [batch, 1]
    
    # Material properties
    material_signatures: torch.Tensor # [batch, n_materials, signature_dim]
    reflectance_profiles: torch.Tensor # [batch, n_materials, bands]

@dataclass
class ControlTheoryToken:
    """Token representing control systems and feedback"""
    # System dynamics
    state_matrix: torch.Tensor       # [batch, n_states, n_states]
    control_matrix: torch.Tensor     # [batch, n_states, n_inputs]
    output_matrix: torch.Tensor      # [batch, n_outputs, n_states]
    
    # Controller parameters
    pid_gains: torch.Tensor          # [batch, 3]  # Kp, Ki, Kd
    adaptive_parameters: torch.Tensor # [batch, n_adaptive_params]
    robust_margins: torch.Tensor     # [batch, 3]  # gain, phase, delay
    
    # Performance metrics
    tracking_error: torch.Tensor     # [batch, n_states]
    control_effort: torch.Tensor     # [batch, n_inputs]
    stability_margins: torch.Tensor  # [batch, 2]  # gain, phase
    
    # Constraints
    input_constraints: torch.Tensor  # [batch, n_inputs, 2]
    state_constraints: torch.Tensor  # [batch, n_states, 2]
    safety_filters: torch.Tensor     # [batch, n_safety_rules]

@dataclass
class SensorimotorToken:
    """Token representing biological sensorimotor integration"""
    # Sensory inputs
    proprioception: torch.Tensor     # [batch, n_joints, 6]  # pos, vel, force
    vision: torch.Tensor             # [batch, channels, height, width]
    vestibular: torch.Tensor         # [batch, 6]  # accel, gyro
    tactile: torch.Tensor            # [batch, n_tactile_sensors]
    
    # Motor commands
    desired_trajectory: torch.Tensor # [batch, horizon, n_joints]
    current_command: torch.Tensor    # [batch, n_joints]
    impedance_parameters: torch.Tensor # [batch, n_joints, 2]  # stiffness, damping
    
    # Predictive models
    forward_prediction: torch.Tensor # [batch, horizon, n_states]
    sensory_prediction: torch.Tensor # [batch, n_sensors]
    prediction_error: torch.Tensor   # [batch, n_sensors]
    
    # Learning state
    adaptation_rates: torch.Tensor   # [batch, n_adaptive_params]
    learning_signals: torch.Tensor   # [batch, n_learning_signals]
