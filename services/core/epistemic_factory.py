"""
Epistemic Factory — singleton factories for epistemic kernel, bridge, CPE, PHE.

Usage:
    from epistemic_factory import (
        get_epistemic_kernel,
        get_causality_bridge,
        get_causal_policy_engine,
        get_policy_horizon_engine,
        reset_all,
    )
"""

from typing import Optional
from threading import RLock

from epistemic_kernel import EpistemicKernel
from causal_bridge import CausalityBridge
from causal_policy import CausalPolicyEngine
from causal_policy.phe import PolicyHorizonEngine

_lock = RLock()  # Must be reentrant: get_causality_bridge() calls get_epistemic_kernel() while holding _lock

_ek: Optional[EpistemicKernel] = None
_bridge: Optional[CausalityBridge] = None
_cpe: Optional[CausalPolicyEngine] = None
_phe: Optional[PolicyHorizonEngine] = None


def get_epistemic_kernel() -> EpistemicKernel:
    global _ek
    if _ek is None:
        with _lock:
            if _ek is None:
                _ek = EpistemicKernel()
    return _ek


def get_causality_bridge() -> CausalityBridge:
    global _bridge
    if _bridge is None:
        with _lock:
            if _bridge is None:
                ek = get_epistemic_kernel()
                _bridge = CausalityBridge(execution_kernel=None, epistemic_kernel=ek)
    return _bridge


def get_causal_policy_engine() -> CausalPolicyEngine:
    global _cpe
    if _cpe is None:
        with _lock:
            if _cpe is None:
                bridge = get_causality_bridge()
                _cpe = CausalPolicyEngine(bridge=bridge)
    return _cpe


def get_policy_horizon_engine() -> PolicyHorizonEngine:
    global _phe
    if _phe is None:
        with _lock:
            if _phe is None:
                cpe = get_causal_policy_engine()
                _phe = PolicyHorizonEngine(cpe)
    return _phe


def reset_all():
    """Reset all singletons (for testing)."""
    global _ek, _bridge, _cpe, _phe
    with _lock:
        _ek = None
        _bridge = None
        _cpe = None
        _phe = None
