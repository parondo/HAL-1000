"""
Tests for core modules
"""

import torch
from hall1000 import create_hal_1000_core

def test_core_forward():
    model = create_hal_1000_core()
    inputs = {
        'linguistic': torch.randn(2, 128, 1024),
        'visual': torch.randn(2, 128, 2048),
    }
    outputs = model(inputs)
    assert outputs['unified_output'].shape == (2, 128, 1024)
