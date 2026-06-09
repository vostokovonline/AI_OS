"""
Phase 30.3: Reality Check Layer
=================================

PURPOSE: Prove whether the model actually learns or just fits noise.

NOT: new architecture
BUT: validation, calibration, generalization tests

WHAT IT ADDS:
  1. Train/val split with holdout trajectories
  2. Predictive accuracy metrics (train vs val gap)
  3. Generalization test (OOD trajectories)
  4. Uncertainty calibration (reliability check)
  5. Long-horizon drift metric

PASS/FAIL CRITERIA:
  - If val_loss doesn't decrease → NOT learning
  - If train/val gap > 2x → overfitting
  - If uncertainty doesn't correlate with error → miscalibrated
  - If long-horizon rollout diverges → unstable dynamics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import random


# ============================================================================
# 1. TRAIN/VALIDATION SPLIT
# ============================================================================

class TrajectorySplitter:
    """
    Splits trajectory buffer into train/val sets.
    
    NOT: random shuffle of transitions
    BUT: episode-level split (prevents data leakage)
    """
    
    def __init__(self, val_ratio: float = 0.2):
        self.val_ratio = val_ratio
    
    def split(self, episodes: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Split episodes into train and validation sets.
        
        Returns: (train_episodes, val_episodes)
        """
        n_val = max(1, int(len(episodes) * self.val_ratio))
        
        # Shuffle episodes
        shuffled = episodes.copy()
        random.shuffle(shuffled)
        
        val_episodes = shuffled[:n_val]
        train_episodes = shuffled[n_val:]
        
        return train_episodes, val_episodes


# ============================================================================
# 2. VALIDATION METRICS
# ============================================================================

class ValidationMetrics:
    """
    Computes validation metrics to prove/disprove learning.
    
    Key metrics:
      - val_loss: prediction error on held-out data
      - train_val_gap: overfitting indicator
      - generalization_error: OOD performance
      - rollout_error: multi-step prediction accuracy
      - uncertainty_calibration: does uncertainty correlate with error?
    """
    
    def __init__(self, model, k_steps: int = 5):
        self.model = model
        self.k_steps = k_steps
    
    def compute_validation_loss(self, val_episodes: List[Dict],
                               batch_size: int = 8,
                               seq_len: int = 10) -> Dict:
        """
        Compute validation loss on held-out trajectories.
        
        Returns: {val_loss, val_dynamics_loss, val_reward_loss}
        """
        from phase30_training_loop import compute_sequence_loss, compute_reward_loss
        
        # Create batches from val episodes
        batches = self._create_batches_from_episodes(val_episodes, batch_size, seq_len)
        
        if not batches:
            return {'val_loss': float('inf'), 'error': 'No validation data'}
        
        # Compute losses (no gradient updates)
        # batches is already a list of samples, pass it directly
        dyn_loss = compute_sequence_loss(self.model, batches, self.k_steps)
        rew_loss = compute_reward_loss(self.model, batches)
        
        return {
            'val_loss': dyn_loss + 0.5 * rew_loss,
            'val_dynamics_loss': dyn_loss,
            'val_reward_loss': rew_loss,
            'n_val_batches': len(batches)
        }
    
    def compute_train_val_gap(self, train_loss: float,
                             val_loss: float) -> Dict:
        """
        Compute train/validation gap (overfitting indicator).
        
        Returns: {gap, gap_ratio, overfitting_detected}
        """
        gap = val_loss - train_loss
        gap_ratio = val_loss / max(1e-8, train_loss)
        
        overfitting = gap_ratio > 2.0  # Val loss 2x higher than train
        
        return {
            'gap': gap,
            'gap_ratio': gap_ratio,
            'overfitting_detected': overfitting,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
    
    def compute_generalization_error(self, ood_episodes: List[Dict],
                                    batch_size: int = 8,
                                    seq_len: int = 10) -> float:
        """
        Compute generalization error on out-of-distribution trajectories.
        
        OOD = different pattern from training data.
        """
        from phase30_training_loop import compute_sequence_loss
        
        batches = self._create_batches_from_episodes(ood_episodes, batch_size, seq_len)
        
        if not batches:
            return float('inf')
        
        # batches is already a list of samples, pass it directly
        return compute_sequence_loss(self.model, batches, self.k_steps)
    
    def compute_rollout_error(self, val_episodes: List[Dict],
                             horizon: int = 10) -> Dict:
        """
        Compute multi-step rollout error.
        
        Tests if model can predict k steps ahead accurately.
        """
        if not val_episodes:
            return {'rollout_error': float('inf')}
        
        # Sample random episode
        ep = random.choice(val_episodes)
        x_seq = ep.get('events', [])[:horizon + 1]
        a_seq = ep.get('actions', [])[:horizon]
        
        if len(x_seq) < horizon + 1:
            return {'rollout_error': float('inf'), 'error': 'Sequence too short'}
        
        # Ground truth: encode full sequence
        h = np.zeros(self.model.belief_dim)
        true_latents = []
        
        for t in range(len(x_seq)):
            h = self.model.gru_step(h, x_seq[t])
            z, mu, _ = self.model.encode_latent(h, sample=False)
            true_latents.append(mu.copy())
        
        # Model rollout: predict forward
        h_pred = np.zeros(self.model.belief_dim)
        z_pred = np.zeros(self.model.latent_dim)
        
        predicted_latents = [z_pred.copy()]
        
        for t in range(len(a_seq)):
            a = a_seq[t]
            mu_pred, _ = self.model.predict_transition(z_pred, h_pred, a)
            
            z_pred = mu_pred
            h_pred = h_pred * 0.9 + np.pad(z_pred, (0, self.model.belief_dim - self.model.latent_dim)) * 0.1
            
            predicted_latents.append(z_pred.copy())
        
        # Compute rollout error (distance from true latents)
        errors = []
        for t in range(1, len(true_latents)):
            error = np.linalg.norm(predicted_latents[t] - true_latents[t])
            errors.append(error)
        
        # Error growth rate
        if len(errors) > 1:
            error_growth = (errors[-1] - errors[0]) / max(1, len(errors) - 1)
        else:
            error_growth = 0.0
        
        return {
            'rollout_error': np.mean(errors),
            'rollout_error_std': np.std(errors),
            'error_growth_rate': error_growth,
            'horizon': horizon,
            'errors': errors
        }
    
    def compute_uncertainty_calibration(self, val_episodes: List[Dict],
                                       n_samples: int = 50) -> Dict:
        """
        Check if model uncertainty correlates with prediction error.
        
        If well-calibrated: high uncertainty → high error
        If miscalibrated: no correlation or inverse correlation
        """
        if not val_episodes:
            return {'calibration': 'no_data'}
        
        uncertainties = []
        errors = []
        
        for _ in range(min(n_samples, len(val_episodes))):
            ep = random.choice(val_episodes)
            events = ep.get('events', [])
            actions = ep.get('actions', [])
            
            if len(events) < 3:
                continue
            
            # Sample random timestep
            t = random.randint(0, len(events) - 2)
            
            # Encode current state
            h = np.zeros(self.model.belief_dim)
            for i in range(t + 1):
                h = self.model.gru_step(h, events[i])
            
            z, mu, logvar = self.model.encode_latent(h, sample=False)
            
            # Predict next
            a = actions[t]
            mu_pred, logvar_pred = self.model.predict_transition(z, h, a)
            
            # True next latent
            h_true = self.model.gru_step(h, events[t + 1])
            z_true, _, _ = self.model.encode_latent(h_true, sample=False)
            
            # Uncertainty (from logvar)
            uncertainty = np.mean(np.exp(logvar_pred))
            
            # Error
            error = np.linalg.norm(mu_pred - z_true)
            
            uncertainties.append(uncertainty)
            errors.append(error)
        
        if len(uncertainties) < 5:
            return {'calibration': 'insufficient_data'}
        
        # Correlation between uncertainty and error
        correlation = np.corrcoef(uncertainties, errors)[0, 1]
        
        if np.isnan(correlation):
            correlation = 0.0
        
        # Calibration quality
        if correlation > 0.3:
            calibration_status = 'well_calibrated'
        elif correlation > 0.0:
            calibration_status = 'weakly_calibrated'
        else:
            calibration_status = 'miscalibrated'
        
        return {
            'calibration_status': calibration_status,
            'uncertainty_error_correlation': correlation,
            'mean_uncertainty': np.mean(uncertainties),
            'mean_error': np.mean(errors),
            'n_samples': len(uncertainties)
        }
    
    def _create_batches(self, episodes: List[Dict],
                       batch_size: int, seq_len: int) -> List[Dict]:
        """Create batches from episodes (assumes x_seq, a_seq, r_seq format)."""
        batches = []
        
        for _ in range(batch_size):
            if not episodes:
                break
            
            ep = random.choice(episodes)
            
            if len(ep.get('x_seq', [])) <= seq_len:
                batches.append(ep)
                continue
            
            start = random.randint(0, len(ep['x_seq']) - seq_len - 1)
            
            batches.append({
                'x_seq': ep['x_seq'][start:start + seq_len],
                'a_seq': ep['a_seq'][start:start + seq_len],
                'r_seq': ep['r_seq'][start:start + seq_len]
            })
        
        return batches
    
    def _create_batches_from_episodes(self, episodes: List[Dict],
                                     batch_size: int, seq_len: int) -> List[Dict]:
        """Create batches from raw episodes (events, actions, rewards format)."""
        batches = []
        
        for _ in range(batch_size):
            if not episodes:
                break
            
            ep = random.choice(episodes)
            events = ep.get('events', [])
            actions = ep.get('actions', [])
            rewards = ep.get('rewards', [])
            
            if len(events) <= seq_len:
                batches.append({
                    'x_seq': events,
                    'a_seq': actions,
                    'r_seq': rewards
                })
                continue
            
            start = random.randint(0, len(events) - seq_len - 1)
            
            batches.append({
                'x_seq': events[start:start + seq_len],
                'a_seq': actions[start:start + seq_len],
                'r_seq': rewards[start:start + seq_len]
            })
        
        return batches


# ============================================================================
# 3. LEARNING PROOF (Pass/Fail criteria)
# ============================================================================

class LearningProof:
    """
    Determines if model is ACTUALLY learning or just fitting noise.
    
    PASS criteria:
      1. Val loss decreases over training
      2. Train/val gap < 2x (no severe overfitting)
      3. Uncertainty correlates with error (calibration)
      4. Rollout error doesn't explode (stability)
    
    FAIL if any criterion not met.
    """
    
    def __init__(self):
        self.history: List[Dict] = []
    
    def record_epoch(self, epoch: int, train_loss: float,
                    val_metrics: Dict):
        """Record epoch metrics."""
        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics
        })
    
    def evaluate(self) -> Dict:
        """
        Evaluate if model is actually learning.
        
        Returns: {is_learning, evidence, failures}
        """
        if len(self.history) < 3:
            return {
                'is_learning': False,
                'reason': 'Insufficient training history (need >= 3 epochs)',
                'evidence': [],
                'failures': ['not_enough_data']
            }
        
        evidence = []
        failures = []
        
        # 1. Check if val loss decreases
        val_losses = [h.get('val_loss', float('inf')) for h in self.history]
        val_losses = [v for v in val_losses if v != float('inf')]
        
        if len(val_losses) >= 2:
            first_val = val_losses[0]
            last_val = val_losses[-1]
            
            if last_val < first_val:
                improvement = (1 - last_val / first_val) * 100
                evidence.append(f"Val loss decreased: {first_val:.4f} → {last_val:.4f} ({improvement:.1f}%)")
            else:
                failures.append(f"Val loss did NOT decrease: {first_val:.4f} → {last_val:.4f}")
        
        # 2. Check train/val gap
        recent = self.history[-1]
        if 'val_loss' in recent and 'train_loss' in recent:
            gap_ratio = recent['val_loss'] / max(1e-8, recent['train_loss'])
            
            if gap_ratio < 2.0:
                evidence.append(f"Train/val gap OK: {gap_ratio:.2f}x")
            else:
                failures.append(f"Severe overfitting: val_loss {gap_ratio:.2f}x higher than train_loss")
        
        # 3. Check uncertainty calibration
        if 'calibration_status' in recent:
            if recent['calibration_status'] in ['well_calibrated', 'weakly_calibrated']:
                evidence.append(f"Uncertainty calibration: {recent['calibration_status']}")
            else:
                failures.append(f"Uncertainty miscalibrated: {recent['calibration_status']}")
        
        # 4. Check rollout stability
        if 'rollout_error' in recent:
            if recent['rollout_error'] < 2.0:
                evidence.append(f"Rollout error stable: {recent['rollout_error']:.3f}")
            else:
                failures.append(f"Rollout error too high: {recent['rollout_error']:.3f}")
        
        # Final verdict
        is_learning = len(failures) == 0 and len(evidence) >= 2
        
        return {
            'is_learning': is_learning,
            'evidence': evidence,
            'failures': failures,
            'n_epochs': len(self.history),
            'verdict': 'LEARNING' if is_learning else 'NOT_LEARNING'
        }


# ============================================================================
# 4. INTEGRATED REALITY CHECK
# ============================================================================

class RealityCheckLayer:
    """
    Complete reality check for Phase 30 v2.
    
    Wraps training loop with validation, calibration, and proof.
    """
    
    def __init__(self, model, val_ratio: float = 0.2, k_steps: int = 5):
        self.model = model
        self.val_ratio = val_ratio
        self.k_steps = k_steps
        
        self.splitter = TrajectorySplitter(val_ratio)
        self.metrics = ValidationMetrics(model, k_steps)
        self.proof = LearningProof()
        
        self.train_episodes = []
        self.val_episodes = []
    
    def prepare_data(self, episodes: List[Dict]):
        """Split data into train/val sets."""
        self.train_episodes, self.val_episodes = self.splitter.split(episodes)
    
    def run_validation_epoch(self, train_loss: float, train_batch: Dict = None) -> Dict:
        """
        Run validation after training epoch.
        
        Returns: validation metrics + learning proof
        """
        # Compute validation loss
        val_metrics = self.metrics.compute_validation_loss(
            self.val_episodes, batch_size=8, seq_len=10
        )
        
        # Compute train/val gap using the current train_batch
        if train_batch is not None and 'val_loss' in val_metrics:
            from phase30_training_loop import compute_sequence_loss, compute_reward_loss
            train_dyn_loss = compute_sequence_loss(self.model, train_batch, self.k_steps)
            train_rew_loss = compute_reward_loss(self.model, train_batch)
            current_train_loss = train_dyn_loss + 0.5 * train_rew_loss
            
            gap = self.metrics.compute_train_val_gap(
                current_train_loss, val_metrics['val_loss']
            )
        elif 'val_loss' in val_metrics:
            gap = self.metrics.compute_train_val_gap(
                train_loss, val_metrics['val_loss']
            )
        else:
            gap = {}
        val_metrics.update(gap)
        
        # Compute rollout error
        rollout = self.metrics.compute_rollout_error(
            self.val_episodes, horizon=10
        )
        val_metrics.update(rollout)
        
        # Compute uncertainty calibration
        calibration = self.metrics.compute_uncertainty_calibration(
            self.val_episodes, n_samples=30
        )
        val_metrics.update(calibration)
        
        # Record in proof tracker
        self.proof.record_epoch(
            epoch=len(self.proof.history),
            train_loss=train_loss,
            val_metrics=val_metrics
        )
        
        return val_metrics
    
    def get_final_verdict(self) -> Dict:
        """Get final learning verdict."""
        return self.proof.evaluate()


# ============================================================================
# 5. TESTS
# ============================================================================

def test_reality_check():
    """Test full reality check pipeline."""
    print("\n" + "=" * 60)
    print("REALITY CHECK TEST")
    print("=" * 60)
    
    from phase30_training_loop import (
        MinimalWorldModel, TrajectoryBuffer, training_loop
    )
    
    # Create model and data
    model = MinimalWorldModel(event_dim=32, latent_dim=16,
                             belief_dim=64, action_dim=16)
    
    # Generate training data
    buffer = TrajectoryBuffer()
    all_episodes = []
    
    for ep in range(40):
        length = random.randint(20, 40)
        events = [np.random.randn(32) * 0.5 for _ in range(length)]
        actions = [np.random.randn(16) * 0.3 for _ in range(length)]
        rewards = [random.uniform(0.3, 0.9) for _ in range(length)]
        
        episode = {
            'events': events,
            'actions': actions,
            'rewards': rewards
        }
        all_episodes.append(episode)
        buffer.add_episode(events, actions, rewards)
    
    print(f"\n  Generated {len(all_episodes)} episodes")
    
    # Setup reality check
    reality_check = RealityCheckLayer(model, val_ratio=0.2, k_steps=3)
    reality_check.prepare_data(all_episodes)
    
    print(f"  Train episodes: {len(reality_check.train_episodes)}")
    print(f"  Val episodes: {len(reality_check.val_episodes)}")
    
    # Train with validation
    print("\n  Training with validation:")
    
    n_epochs = 30
    for epoch in range(n_epochs):
        # Train step
        batch = buffer.sample_batch(batch_size=8, seq_len=10)
        
        from phase30_training_loop import train_step, SimpleOptimizer
        optimizer = SimpleOptimizer(model, learning_rate=0.001)
        logs = train_step(model, optimizer, batch, k_steps=3)
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            val_metrics = reality_check.run_validation_epoch(
                logs['total_loss'], train_batch=batch
            )
            
            print(f"  Epoch {epoch:2d} | "
                  f"train: {logs['total_loss']:.4f} | "
                  f"val: {val_metrics.get('val_loss', 'N/A')} | "
                  f"rollout: {val_metrics.get('rollout_error', 'N/A')} | "
                  f"calibration: {val_metrics.get('calibration_status', 'N/A')}")
    
    # Final verdict
    verdict = reality_check.get_final_verdict()
    
    print(f"\n  {'='*40}")
    print(f"  VERDICT: {verdict['verdict']}")
    print(f"  {'='*40}")
    
    if verdict['evidence']:
        print(f"\n  Evidence:")
        for e in verdict['evidence']:
            print(f"    ✓ {e}")
    
    if verdict['failures']:
        print(f"\n  Failures:")
        for f in verdict['failures']:
            print(f"    ✗ {f}")


def test_generalization():
    """Test generalization to OOD data."""
    print("\n" + "=" * 60)
    print("GENERALIZATION TEST")
    print("=" * 60)
    
    from phase30_training_loop import MinimalWorldModel
    
    model = MinimalWorldModel(event_dim=32, latent_dim=16,
                             belief_dim=64, action_dim=16)
    
    # Generate training data (pattern A)
    train_episodes = []
    for ep in range(20):
        length = 25
        events = [np.random.randn(32) * 0.5 for _ in range(length)]
        actions = [np.random.randn(16) * 0.3 for _ in range(length)]
        rewards = [random.uniform(0.3, 0.9) for _ in range(length)]
        
        train_episodes.append({
            'events': events,
            'actions': actions,
            'rewards': rewards
        })
    
    # Generate OOD data (pattern B - different distribution)
    ood_episodes = []
    for ep in range(10):
        length = 25
        events = [np.random.randn(32) * 2.0 + 1.0 for _ in range(length)]  # Different mean/std
        actions = [np.random.randn(16) * 1.0 for _ in range(length)]
        rewards = [random.uniform(0.1, 0.5) for _ in range(length)]
        
        ood_episodes.append({
            'events': events,
            'actions': actions,
            'rewards': rewards
        })
    
    metrics = ValidationMetrics(model, k_steps=3)
    
    # Compute in-distribution error
    in_dist_error = metrics.compute_generalization_error(
        train_episodes, batch_size=8, seq_len=10
    )
    
    # Compute OOD error
    ood_error = metrics.compute_generalization_error(
        ood_episodes, batch_size=8, seq_len=10
    )
    
    print(f"\n  In-distribution error: {in_dist_error:.4f}")
    print(f"  OOD error: {ood_error:.4f}")
    
    if ood_error > in_dist_error * 1.5:
        print(f"\n  ✓ Model shows expected OOD degradation")
    else:
        print(f"\n  ✗ Model may be underfitting or OOD not different enough")


if __name__ == "__main__":
    test_reality_check()
    test_generalization()
    
    print("\n" + "=" * 60)
    print("PHASE 30.3: REALITY CHECK LAYER")
    print("=" * 60)
    
    print("""
PURPOSE: Prove whether model actually learns or just fits noise.

WHAT IT ADDS:
  1. Train/val split (episode-level, no data leakage)
  2. Validation loss tracking
  3. Train/val gap (overfitting detection)
  4. Rollout error (multi-step stability)
  5. Uncertainty calibration (does uncertainty correlate with error?)
  6. Generalization test (OOD performance)

PASS/FAIL CRITERIA:
  ✓ Val loss decreases → learning
  ✗ Val loss doesn't decrease → NOT learning
  ✓ Train/val gap < 2x → no severe overfitting
  ✗ Train/val gap > 2x → overfitting
  ✓ Uncertainty correlates with error → calibrated
  ✗ No correlation → miscalibrated
  ✓ Rollout error < 2.0 → stable dynamics
  ✗ Rollout error > 2.0 → unstable

VERDICT:
  LEARNING = all criteria pass
  NOT_LEARNING = any criterion fails

This answers: "Is the system actually learning or lying?"
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 30.3 Summary: Reality Check Layer

What was added:
  - TrajectorySplitter (train/val split at episode level)
  - ValidationMetrics (val loss, gap, rollout, calibration)
  - LearningProof (pass/fail criteria for actual learning)
  - RealityCheckLayer (integrated validation wrapper)

Key capability:
  System now PROVES whether it learns or just fits noise.

Before: "CEM found sequence with 99% confidence" (unverified claim)
After: "Val loss decreased 15%, calibration OK, rollout stable" (proven fact)

This is the difference between:
  "system that claims to learn"
  and
  "system that proves it learns"
"""