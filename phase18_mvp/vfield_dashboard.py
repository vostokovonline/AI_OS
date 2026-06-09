"""
V-Field Dashboard - Text/ASCII Visualization

Minimal dashboard for MVP v3 - no external dependencies.
Can run in terminal.
"""
import numpy as np
import sys
from typing import List, Dict, Any


class VFieldDashboard:
    """
    Simple terminal dashboard for V-field visualization.
    """
    
    def __init__(self, width: int = 80):
        self.width = width
        self.history = []
        
    def clear(self):
        """Clear screen."""
        print("\033[2J\033[H")
    
    def draw(self, viz_data: dict):
        """Draw dashboard from viz data."""
        self.clear()
        
        current = viz_data.get('current', {})
        series = viz_data.get('series', {})
        policy = viz_data.get('policy', {})
        detection = viz_data.get('detection', {})
        alerts = viz_data.get('alerts', [])
        trajectories = viz_data.get('trajectories', {})
        
        # Header
        print("=" * self.width)
        print("PHASE 18 MVP v3 - V-FIELD DASHBOARD".center(self.width))
        print("=" * self.width)
        
        # Alerts
        if alerts:
            print("\n⚠️  ALERTS:")
            for alert in alerts:
                icon = "🔴" if alert['type'] == 'critical' else "🟡"
                print(f"  {icon} {alert['message']}")
        
        # V-Field Metrics
        print("\n┌─────────────────────────────────────────────┐")
        print("│ V-FIELD METRICS                             │")
        print("├─────────────────────────────────────────────┤")
        
        V = current.get('V', 0)
        div = current.get('diversity', 0)
        col = current.get('collapse', 0)
        inst = current.get('instability', 0)
        trend = current.get('trend', 0)
        status = current.get('status', 'UNKNOWN')
        
        # V bar
        v_bar = self._make_bar(V, 30, colors={'low': '🔴', 'mid': '🟡', 'high': '🟢'})
        print(f"│ V:      {v_bar} {V:.3f}                         │")
        
        # Diversity bar
        d_bar = self._make_bar(div, 30, colors={'low': '🔴', 'mid': '🟡', 'high': '🟢'})
        print(f"│ Div:    {d_bar} {div:.3f}                         │")
        
        # Collapse bar
        c_bar = self._make_bar(col, 30, colors={'low': '🔴', 'mid': '🟡', 'high': '🟢'})
        print(f"│ Collapse: {c_bar} {col:.3f}                       │")
        
        # Instability bar
        i_bar = self._make_bar(inst, 30, colors={'low': '🟢', 'mid': '🟡', 'high': '🔴'})
        print(f"│ Inst:   {i_bar} {inst:.3f}                         │")
        
        print(f"├─────────────────────────────────────────────┤")
        print(f"│ Trend: {trend:+.4f}  Status: {status:<15} │")
        print("└─────────────────────────────────────────────┘")
        
        # V Series
        print("\nV-HISTORY (last 40 steps):")
        if series.get('V'):
            v_hist = series['V'][-40:]
            self._draw_series(v_hist, "V", width=self.width-4)
        
        # Detection signals
        print("\n┌─────────────────────────────────────────────┐")
        print("│ DETECTION SIGNALS                           │")
        print("├─────────────────────────────────────────────┤")
        
        silent = "✓" if not detection.get('silent_collapse') else "✗"
        trap = detection.get('attractor_trap', 0)
        
        print(f"│ Silent Collapse: {silent}    Attractor Trap: {trap:.3f}     │")
        print("└─────────────────────────────────────────────┘")
        
        # Policy
        print("\n┌─────────────────────────────────────────────┐")
        print("│ POLICY DECISIONS                            │")
        print("├─────────────────────────────────────────────┤")
        
        candidates = policy.get('action_candidates', [])
        selected = policy.get('selected_action', -1)
        
        if candidates:
            for c in candidates:
                idx = c['idx']
                v = c['V']
                marker = "→" if idx == selected else " "
                bar = self._make_bar(v, 20)
                print(f"│ {marker} Action {idx}: {bar} {v:.3f}  │")
        
        print("└─────────────────────────────────────────────┘")
        
        # Trajectory Ensemble
        print("\nTRAJECTORY ENDPOINTS (latent space projection):")
        endpoints = trajectories.get('endpoints', [])
        if len(endpoints) > 0:
            endpoints_arr = np.array(endpoints)
            if endpoints_arr.shape[1] >= 2:
                # Simple 2D scatter (first 2 dims)
                self._draw_scatter_2d(endpoints_arr[:, :2])
        
        # Rewards
        rewards = viz_data.get('rewards', {})
        print(f"\nRewards: current={rewards.get('current', 0):.2f}, total={rewards.get('total', 0):.2f}")
        
        print("\n" + "=" * self.width)
    
    def _make_bar(self, value: float, length: int, 
                  colors: dict = None) -> str:
        """Create ASCII bar."""
        if colors is None:
            colors = {'low': '░', 'mid': '▒', 'high': '█'}
        
        filled = int(value * length)
        empty = length - filled
        
        # Color based on value
        if value < 0.3:
            color = colors['low']
        elif value < 0.6:
            color = colors['mid']
        else:
            color = colors['high']
        
        return color * filled + '░' * empty
    
    def _draw_series(self, values: List[float], label: str, width: int = 60):
        """Draw ASCII line chart."""
        if not values:
            return
        
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1
        
        # Normalize to 0-10 range
        height = 10
        normalized = [int((v - min_val) / range_val * height) for v in values]
        
        # Create chart
        chars = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
        
        for h in range(height, -1, -1):
            line = ""
            for n in normalized:
                if n >= h:
                    line += chars[min(h, len(chars)-1)]
                else:
                    line += ' '
            print(f"  {line}")
        
        # X axis
        print("  " + "─" * len(values))
        print(f"  {label}: min={min_val:.3f}, max={max_val:.3f}")
    
    def _draw_scatter_2d(self, points: np.ndarray, width: int = 40, height: int = 15):
        """Draw 2D scatter plot."""
        if len(points) == 0 or points.shape[1] < 2:
            return
        
        # Normalize to grid
        x = points[:, 0]
        y = points[:, 1]
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Add padding
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1
        
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        for pt in points:
            px = int((pt[0] - x_min) / x_range * (width - 1))
            py = int((pt[1] - y_min) / y_range * (height - 1))
            py = height - 1 - py  # Flip Y
            
            px = max(0, min(width - 1, px))
            py = max(0, min(height - 1, py))
            
            if grid[py][px] == ' ':
                grid[py][px] = '•'
            else:
                grid[py][px] = '◉'  # Overlapping point
        
        # Draw
        for row in grid:
            print("  " + "".join(row))
        
        print(f"  x: [{x_min:.2f}, {x_max:.2f}]")
        print(f"  y: [{y_min:.2f}, {y_max:.2f}]")


def run_dashboard():
    """Run live dashboard with MVP."""
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    from dashboard import VFieldDashboardAdapter
    from vfield import VField
    import main
    MVPRunner = main.MVPRunner
    
    print("Initializing V-Field Dashboard...")
    
    # Initialize MVP
    runner = MVPRunner(env_type='simple')
    adapter = VFieldDashboardAdapter(runner.vfield, runner.policy, runner.dynamics, runner.encoder)
    dashboard = VFieldDashboard()
    
    # Reset
    runner.reset()
    obs = runner.env.reset()
    
    print("Running episodes with dashboard visualization...")
    
    for episode in range(3):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}")
        print(f"{'='*60}")
        
        runner.reset()
        runner.vfield.reset_history()
        adapter.history = []
        adapter.total_reward = 0
        
        for step in range(50):  # Short episode for demo
            # Get action
            action_idx, action, V = runner.policy.select_action(
                runner.encoder.encode(runner.env._get_obs()),
                runner.dynamics,
                runner.vfield
            )
            
            # Execute
            obs, reward, done = runner.env.step(action)
            
            # Update adapter
            state = adapter.update(runner.env._get_obs(), action_idx, reward)
            
            # Draw dashboard every 10 steps
            if step % 10 == 0:
                viz_data = adapter.get_viz_data()
                dashboard.draw(viz_data)
                print(f"\nStep {step}, V={state.V:.3f}, Action={action_idx}")
        
        # Final stats
        print(f"\nEpisode {episode + 1} complete:")
        print(f"  Total reward: {adapter.total_reward:.2f}")
        print(f"  V mean: {np.mean([h.V for h in adapter.history]):.3f}")


if __name__ == '__main__':
    run_dashboard()