"""
Attention module - Determines which inputs deserve processing.

Attention filtering is NOT prioritization.
It is pre-salience filtering that decides what enters the cognitive system.

Components:
- filtering.py: Signal/noise classification, input filtering
"""

from cognitive_loop.attention.filtering import (
    InputSignal,
    InputSource,
    FilterConfig,
    FilteredInputs,
    filter_inputs,
    compute_input_features
)

__all__ = [
    'InputSignal',
    'InputSource',
    'FilterConfig',
    'FilteredInputs',
    'filter_inputs',
    'compute_input_features'
]