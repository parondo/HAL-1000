"""
Demo: HALL 1000 Piloting a Humanoid Robot
"""

import torch
import yaml
from core import EmbodiedHALL1000
from tokens import MathematicalTokenFactory
from embodied import HumanoidPilotingSystem

def main():
    # Load configuration
    with open('configs/robotics_piloting.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create piloting system
    pilot = HumanoidPilotingSystem(config)
    
    # Example robot state (simulated data)
    robot_state = {
        'joint_positions': torch.randn(1, 20),  # 20 joints
        'joint_velocities': torch.randn(1, 20),
        'base_pose': torch.tensor([0, 0, 1, 0, 0, 0, 1]),  # xyz + quaternion
        'contact_forces': torch.randn(1, 4, 3),  # 4 feet, 3D forces
        'inertial_properties': torch.randn(1, 10, 10),  # Mass matrix
        'desired_velocity': torch.tensor([0.5, 0.0, 0.0]),  # Desired walking direction
    }
    
    # Create mathematical tokens
    dynamics_token = MathematicalTokenFactory.create_physical_dynamics(robot_state)
    
    # Process through HALL 1000
    with torch.no_grad():
        output = pilot({'dynamics': dynamics_token})
    
    # Extract motor commands
    joint_commands = output['joint_commands']
    balance_adjustments = output['balance_commands']
    
    print(f"Generated joint commands: {joint_commands.shape}")
    print(f"Balance adjustments: {balance_adjustments.shape}")
    print(f"Coherence metrics: {output['coherence_metrics']}")

if __name__ == "__main__":
    main()
