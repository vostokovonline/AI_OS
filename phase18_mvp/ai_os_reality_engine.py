"""
AI-OS Reality Engine
====================

ARCHITECTURAL SHIFT:
  From: Strategic heuristic layer
  To: Adaptive Strategic Cognitive Runtime with reality feedback
  
The key insight: Goals are not the top of a tree.
Goals are part of a dynamic system:
  
  Goals
    ↕
  Trajectory
    ↕
  Momentum
    ↕
  Environment
    ↕
  Execution Reality

CORE PRINCIPLE: 
  AI-OS must connect to real execution data.
  Without reality feedback, goal hierarchy is mostly symbolic.
  
PRIORITY: Not new cognitive modules.
PRIORITY: Reality feedback loop.

Components to build:
  1. Event Runtime - telemetry from real systems
  2. Temporal Memory - causal behavior history
  3. Execution Sessions - deep work, recovery, etc.
  4. Real Interventions - behavioral guidance
  5. Goal Vitality - alive/dead/toxic detection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict


# ============================================================================
# EVENT RUNTIME
# ============================================================================
"""
Event Runtime: Reality Telemetry

Connects AI-OS to real execution data:

- Git activity (commits, PRs, branches)
- IDE events (open files, debugging, building)
- Terminal commands
- Task lifecycle events
- Browser tabs, idle/activity
- Sleep/wake cycles
- Interruptions, context switches

Raw Events → Behavioral Streams → Execution Patterns
"""

class EventSource(Enum):
    """Sources of execution telemetry."""
    GIT = "git"
    IDE = "ide"
    TERMINAL = "terminal"
    TASK = "task"
    BROWSER = "browser"
    CALENDAR = "calendar"
    SLEEP = "sleep"
    USER_INPUT = "user_input"
    MESSAGE = "message"
    CHAT = "chat"


@dataclass
class RawEvent:
    """Raw telemetry event from environment."""
    event_id: str
    timestamp: datetime
    source: EventSource
    event_type: str  # e.g., "commit", "file_open", "task_complete"
    
    # Event data
    data: Dict[str, Any]
    
    # Derived
    session_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'event_type': self.event_type,
            'data': self.data
        }


class EventRuntime:
    """
    Event Runtime: Captures real execution telemetry.
    
    Raw Events → Behavioral Streams
    """
    
    def __init__(self):
        self.events: List[RawEvent] = []
        self.event_counter = 0
        
        # Event streams by source
        self.streams: Dict[EventSource, List[RawEvent]] = defaultdict(list)
        
        # Active sessions
        self.active_sessions: Dict[str, 'ExecutionSession'] = {}
        
        # Event handlers
        self.handlers: Dict[str, List[callable]] = {}
        
    def ingest(self, source: EventSource, event_type: str, 
               data: Dict, timestamp: datetime = None) -> RawEvent:
        """Ingest raw event from environment."""
        self.event_counter += 1
        
        event = RawEvent(
            event_id=f"evt_{self.event_counter}",
            timestamp=timestamp or datetime.now(),
            source=source,
            event_type=event_type,
            data=data
        )
        
        self.events.append(event)
        self.streams[source].append(event)
        
        # Call handlers
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    handler(event)
                except Exception:
                    pass
        
        # Keep memory bounded
        if len(self.events) > 10000:
            self.events = self.events[-5000:]
        
        return event
    
    def ingest_git(self, action: str, repo: str, details: Dict):
        """Ingest git event."""
        return self.ingest(EventSource.GIT, action, {
            'repo': repo,
            **details
        })
    
    def ingest_ide(self, action: str, file: str, details: Dict):
        """Ingest IDE event."""
        return self.ingest(EventSource.IDE, action, {
            'file': file,
            **details
        })
    
    def ingest_terminal(self, command: str, cwd: str, exit_code: int = 0):
        """Ingest terminal event."""
        return self.ingest(EventSource.TERMINAL, 'command', {
            'command': command,
            'cwd': cwd,
            'exit_code': exit_code
        })
    
    def ingest_task(self, task_id: str, action: str, details: Dict = None):
        """Ingest task event."""
        return self.ingest(EventSource.TASK, action, {
            'task_id': task_id,
            **(details or {})
        })
    
    def ingest_browser(self, action: str, url: str = None, title: str = None):
        """Ingest browser event."""
        return self.ingest(EventSource.BROWSER, action, {
            'url': url,
            'title': title
        })
    
    def ingest_sleep(self, event_type: str, duration_minutes: float = None):
        """Ingest sleep event."""
        return self.ingest(EventSource.SLEEP, event_type, {
            'duration_minutes': duration_minutes
        })
    
    def ingest_user_input(self, input_type: str, content: str):
        """Ingest user input event."""
        return self.ingest(EventSource.USER_INPUT, input_type, {
            'content': content
        })
    
    def get_stream(self, source: EventSource, 
                   since: datetime = None) -> List[RawEvent]:
        """Get event stream from source."""
        events = self.streams.get(source, [])
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events
    
    def get_recent_events(self, minutes: int = 60, 
                          sources: List[EventSource] = None) -> List[RawEvent]:
        """Get recent events within time window."""
        since = datetime.now() - timedelta(minutes=minutes)
        
        if sources:
            events = []
            for source in sources:
                events.extend(self.get_stream(source, since))
        else:
            events = [e for e in self.events if e.timestamp >= since]
        
        events.sort(key=lambda x: x.timestamp, reverse=True)
        return events
    
    def get_activity_summary(self, minutes: int = 60) -> Dict:
        """Get activity summary for time period."""
        recent = self.get_recent_events(minutes)
        
        by_source = defaultdict(int)
        by_type = defaultdict(int)
        
        for event in recent:
            by_source[event.source.value] += 1
            by_type[event.event_type] += 1
        
        return {
            'total_events': len(recent),
            'by_source': dict(by_source),
            'by_type': dict(by_type),
            'time_window_minutes': minutes
        }


# ============================================================================
# TEMPORAL MEMORY
# ============================================================================
"""
Temporal Memory: Causal Behavior History

Not just embeddings.
Real causal chains:

  action -> delayed consequences -> trajectory update

Key structures:
  - Timelines (what happened when)
  - Sessions (coherent work periods)
  - Execution chains (how things got done)
  - Interruption chains (why momentum broke)
  - Recovery chains (how user recovered)
  - Historical causality (what led to what)
"""

@dataclass
class TimelineEntry:
    """Entry in behavioral timeline."""
    timestamp: datetime
    entry_type: str  # "action", "consequence", "state_change", "milestone"
    
    content: str
    causality: str  # "caused_by", "enabled", "blocked", "preceded"
    related_entries: List[str] = field(default_factory=list)
    
    # Impact metrics
    energy_impact: float = 0.0  # How this affected energy
    momentum_impact: float = 0.0  # How this affected momentum
    trajectory_impact: float = 0.0  # How this affected trajectory direction
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalMemory:
    """
    Temporal Memory: Causal behavior history.
    
    Tracks:
      - Execution chains (how things got done)
      - Interruption chains (why momentum broke)
      - Recovery chains (how user recovered)
      - Historical causality (what led to what)
    """
    
    def __init__(self):
        self.timeline: List[TimelineEntry] = []
        self.timeline_counter = 0
        
        # Chains by type
        self.execution_chains: List[List[str]] = []  # [timeline_entry_ids]
        self.interruption_chains: List[List[str]] = []
        self.recovery_chains: List[List[str]] = []
        
        # Causal graph: entry_id -> [caused_entry_ids]
        self.causal_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Pattern recognition
        self.success_patterns: List[Dict] = []
        self.failure_patterns: List[Dict] = []
        
    def record(self, entry_type: str, content: str,
              causality: str = "action",
              related: List[str] = None,
              energy_impact: float = 0.0,
              momentum_impact: float = 0.0,
              trajectory_impact: float = 0.0,
              metadata: Dict = None) -> str:
        """Record timeline entry."""
        self.timeline_counter += 1
        entry_id = f"tl_{self.timeline_counter}"
        
        entry = TimelineEntry(
            timestamp=datetime.now(),
            entry_type=entry_type,
            content=content,
            causality=causality,
            related_entries=related or [],
            energy_impact=energy_impact,
            momentum_impact=momentum_impact,
            trajectory_impact=trajectory_impact,
            metadata=metadata or {}
        )
        
        self.timeline.append(entry)
        
        # Update causal graph
        if related:
            for rel_id in related:
                self.causal_graph[rel_id].append(entry_id)
        
        # Detect patterns
        self._detect_patterns(entry)
        
        # Keep bounded
        if len(self.timeline) > 5000:
            self.timeline = self.timeline[-2000:]
        
        return entry_id
    
    def record_action(self, action: str, context: Dict = None):
        """Record executed action."""
        return self.record(
            entry_type="action",
            content=action,
            causality="action",
            metadata=context
        )
    
    def record_consequence(self, consequence: str, caused_by: str,
                          energy_impact: float = 0.0,
                          momentum_impact: float = 0.0):
        """Record consequence of previous action."""
        return self.record(
            entry_type="consequence",
            content=consequence,
            causality="caused_by",
            related=[caused_by],
            energy_impact=energy_impact,
            momentum_impact=momentum_impact
        )
    
    def record_interruption(self, reason: str, context: Dict = None):
        """Record execution interruption."""
        chain_id = len(self.interruption_chains)
        
        entry_id = self.record(
            entry_type="interruption",
            content=reason,
            causality="action",
            momentum_impact=-0.3,
            metadata={'chain_id': chain_id, **(context or {})}
        )
        
        # Add to interruption chain
        self.interruption_chains.append([entry_id])
        
        return entry_id
    
    def record_recovery(self, recovery_type: str, from_interruption: str):
        """Record recovery from interruption."""
        chain_id = len(self.recovery_chains)
        
        entry_id = self.record(
            entry_type="recovery",
            content=recovery_type,
            causality="enabled_by",
            related=[from_interruption],
            energy_impact=0.1,
            momentum_impact=0.2,
            metadata={'chain_id': chain_id}
        )
        
        # Add to recovery chain
        self.recovery_chains.append([entry_id])
        
        return entry_id
    
    def record_milestone(self, milestone: str, goal_id: str = None):
        """Record significant milestone."""
        return self.record(
            entry_type="milestone",
            content=milestone,
            causality="achieved",
            trajectory_impact=0.3,
            metadata={'goal_id': goal_id} if goal_id else {}
        )
    
    def _detect_patterns(self, entry: TimelineEntry):
        """Detect success/failure patterns."""
        # Recent entries
        recent = self.timeline[-20:]
        
        # Check for failure pattern: many interruptions close together
        interruptions = [e for e in recent if e.entry_type == "interruption"]
        if len(interruptions) >= 3:
            time_diffs = np.diff([e.timestamp for e in interruptions])
            avg_gap = np.mean([d.total_seconds() / 60 for d in time_diffs])
            
            if avg_gap < 30:  # Interrupts within 30 minutes
                pattern = {
                    'type': 'fragmentation',
                    'severity': min(1.0, len(interruptions) / 5),
                    'count': len(interruptions),
                    'entries': [str(e.timestamp) for e in interruptions]
                }
                if pattern not in self.failure_patterns:
                    self.failure_patterns.append(pattern)
        
        # Check for success pattern: milestone followed by progress
        if entry.entry_type == "milestone":
            following = self.timeline[-5:]
            actions = [e for e in following if e.entry_type == "action"]
            if len(actions) >= 3:
                pattern = {
                    'type': 'milestone_momentum',
                    'milestone_id': entry.event_id,
                    'subsequent_actions': len(actions)
                }
                if pattern not in self.success_patterns:
                    self.success_patterns.append(pattern)
    
    def get_causal_chain(self, entry_id: str, depth: int = 3) -> List[TimelineEntry]:
        """Get causal chain from entry."""
        chain = []
        current_ids = [entry_id]
        visited = set()
        
        for _ in range(depth):
            next_ids = []
            for eid in current_ids:
                if eid in visited:
                    continue
                visited.add(eid)
                
                # Find entry
                for entry in self.timeline:
                    pass  # Simplified - would need proper lookup
            
            current_ids = next_ids
        
        return chain
    
    def analyze_success_patterns(self) -> Dict:
        """Analyze what patterns lead to success."""
        if not self.success_patterns:
            return {'patterns': [], 'insights': []}
        
        # Aggregate patterns
        pattern_counts = defaultdict(int)
        for pattern in self.success_patterns:
            pattern_counts[pattern['type']] += 1
        
        insights = []
        if pattern_counts.get('milestone_momentum', 0) > 2:
            insights.append("Milestones followed by sustained action indicate high-value periods")
        
        return {
            'patterns': list(pattern_counts.items()),
            'insights': insights,
            'total_milestones': len([e for e in self.timeline if e.entry_type == 'milestone'])
        }
    
    def analyze_failure_patterns(self) -> Dict:
        """Analyze what patterns lead to failure."""
        if not self.failure_patterns:
            return {'patterns': [], 'warnings': []}
        
        warnings = []
        for pattern in self.failure_patterns[-5:]:
            if pattern['type'] == 'fragmentation':
                warnings.append({
                    'type': 'fragmentation_risk',
                    'severity': pattern['severity'],
                    'message': f"Fragmentation detected: {pattern['count']} interruptions in short period"
                })
        
        return {
            'patterns': self.failure_patterns[-10:],
            'warnings': warnings
        }
    
    def get_timeline_summary(self) -> Dict:
        """Get timeline summary."""
        entries_by_type = defaultdict(int)
        for entry in self.timeline:
            entries_by_type[entry.entry_type] += 1
        
        return {
            'total_entries': len(self.timeline),
            'by_type': dict(entries_by_type),
            'execution_chains': len(self.execution_chains),
            'interruption_chains': len(self.interruption_chains),
            'recovery_chains': len(self.recovery_chains),
            'success_patterns': len(self.success_patterns),
            'failure_patterns': len(self.failure_patterns)
        }


# ============================================================================
# EXECUTION SESSION MODEL
# ============================================================================
"""
Execution Session Model: Killer Feature

Not "tasks".
Sessions:

- Deep work session
- Recovery session
- Maintenance session
- Exploration session
- Deployment session

AI-OS must understand:
  - When session is breaking
  - When momentum is growing
  - When context switch is dangerous
  - When to restart
"""

class SessionType(Enum):
    """Types of execution sessions."""
    DEEP_WORK = "deep_work"      # Focused, high-complexity
    RECOVERY = "recovery"        # Low-intensity, maintenance
    EXPLORATION = "exploration"  # Learning, research
    DEPLOYMENT = "deployment"    # Shipping, release
    MAINTENANCE = "maintenance"  # Fixing, refactoring
    PLANNING = "planning"        # Strategic thinking
    COLLABORATION = "collaboration"  # Meetings, sync


class SessionState(Enum):
    """Session execution states."""
    EMERGING = "emerging"        # Just started
    BUILDING = "building"        # Momentum growing
    SUSTAINED = "sustained"      # In flow
    DEGRADING = "degrading"      # Losing focus
    BROKEN = "broken"            # Session ended
    COMPLETED = "completed"      # Natural finish


@dataclass
class ExecutionSession:
    """Model of execution session."""
    session_id: str
    session_type: SessionType
    
    start_time: datetime
    expected_duration: float  # minutes
    
    # State tracking
    state: SessionState = SessionState.EMERGING
    
    # Metrics
    continuity: float = 0.0     # How unbroken is session
    intensity: float = 0.0     # How focused
    productivity: float = 0.0  # Output quality
    
    # Context
    goal_id: Optional[str] = None
    tasks_in_scope: List[str] = field(default_factory=list)
    
    # Events in session
    events: List[str] = field(default_factory=list)
    interruptions: int = 0
    
    # Warning flags
    fragmentation_risk: float = 0.0
    collapse_risk: float = 0.0
    
    # Completion
    end_time: Optional[datetime] = None
    natural_completion: bool = False
    
    def compute_state(self, current_time: datetime):
        """Compute current session state based on metrics."""
        elapsed = (current_time - self.start_time).total_seconds() / 60
        
        # State transitions
        if self.state == SessionState.EMERGING:
            if self.continuity > 0.5:
                self.state = SessionState.BUILDING
        
        if self.state == SessionState.BUILDING:
            if self.continuity > 0.7 and self.intensity > 0.6:
                self.state = SessionState.SUSTAINED
        
        if self.state in [SessionState.BUILDING, SessionState.SUSTAINED]:
            if self.fragmentation_risk > 0.7:
                self.state = SessionState.DEGRADING
        
        if self.state == SessionState.DEGRADING:
            if self.collapse_risk > 0.5:
                self.state = SessionState.BROKEN
        
        return self.state
    
    def compute_collapse_risk(self) -> float:
        """Compute session collapse risk."""
        # Interruption rate
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed > 0:
            interruption_rate = self.interruptions / elapsed
        else:
            interruption_rate = 0
        
        # Continuity decay
        continuity_trend = 0.0
        if len(self.events) > 3:
            # Recent continuity vs early
            recent = self.continuity
            self.collapse_risk = min(1.0, interruption_rate * 0.5 + (1 - self.continuity) * 0.3)
        
        return self.collapse_risk


class SessionModel:
    """
    Execution Session Model.
    
    Tracks session lifecycle, detects breaks, predicts outcomes.
    """
    
    def __init__(self, temporal_memory: TemporalMemory):
        self.temporal_memory = temporal_memory
        
        self.sessions: List[ExecutionSession] = []
        self.session_counter = 0
        
        self.active_session: Optional[ExecutionSession] = None
        
        # Session templates
        self.session_templates = {
            SessionType.DEEP_WORK: {
                'expected_duration': 90,  # minutes
                'ideal_continuity': 0.8,
                'interrupt_threshold': 3,
                'collapse_risk_threshold': 0.4
            },
            SessionType.RECOVERY: {
                'expected_duration': 30,
                'ideal_continuity': 0.5,
                'interrupt_threshold': 5,
                'collapse_risk_threshold': 0.6
            },
            SessionType.EXPLORATION: {
                'expected_duration': 45,
                'ideal_continuity': 0.6,
                'interrupt_threshold': 4,
                'collapse_risk_threshold': 0.5
            },
            SessionType.DEPLOYMENT: {
                'expected_duration': 60,
                'ideal_continuity': 0.7,
                'interrupt_threshold': 2,
                'collapse_risk_threshold': 0.3
            }
        }
    
    def start_session(self, session_type: SessionType, 
                     goal_id: str = None, tasks: List[str] = None) -> ExecutionSession:
        """Start new execution session."""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        
        template = self.session_templates.get(session_type, {})
        
        session = ExecutionSession(
            session_id=session_id,
            session_type=session_type,
            start_time=datetime.now(),
            expected_duration=template.get('expected_duration', 60),
            goal_id=goal_id,
            tasks_in_scope=tasks or []
        )
        
        self.active_session = session
        self.sessions.append(session)
        
        # Record in timeline
        self.temporal_memory.record(
            entry_type="session_start",
            content=f"Started {session_type.value} session",
            causality="action",
            metadata={'session_id': session_id}
        )
        
        return session
    
    def record_event(self, event_type: str, context: Dict = None):
        """Record event in active session."""
        if not self.active_session:
            return
        
        self.active_session.events.append(event_type)
        
        # Update metrics
        if event_type == "focused_work":
            self.active_session.continuity = min(1.0, self.active_session.continuity + 0.1)
            self.active_session.intensity = min(1.0, self.active_session.intensity + 0.05)
        
        elif event_type == "interruption":
            self.active_session.interruptions += 1
            self.active_session.continuity = max(0, self.active_session.continuity - 0.2)
            self.active_session.fragmentation_risk = min(1.0, 
                self.active_session.fragmentation_risk + 0.2)
            
            # Record in temporal memory
            self.temporal_memory.record_interruption(
                reason=context.get('reason', 'unknown') if context else 'unknown',
                context={'session_id': self.active_session.session_id}
            )
        
        # Update state
        self.active_session.compute_state(datetime.now())
        self.active_session.compute_collapse_risk()
    
    def end_session(self, natural_completion: bool = False):
        """End active session."""
        if not self.active_session:
            return
        
        self.active_session.end_time = datetime.now()
        self.active_session.natural_completion = natural_completion
        self.active_session.state = (SessionState.COMPLETED if natural_completion 
                                    else SessionState.BROKEN)
        
        # Record completion
        self.temporal_memory.record(
            entry_type="session_end",
            content=f"Completed {self.active_session.session_type.value} session",
            causality="achieved" if natural_completion else "abandoned",
            momentum_impact=0.2 if natural_completion else -0.2,
            metadata={
                'session_id': self.active_session.session_id,
                'duration': (datetime.now() - self.active_session.start_time).total_seconds() / 60,
                'interruptions': self.active_session.interruptions
            }
        )
        
        self.active_session = None
    
    def get_active_session_summary(self) -> Optional[Dict]:
        """Get summary of active session."""
        if not self.active_session:
            return None
        
        session = self.active_session
        elapsed = (datetime.now() - session.start_time).total_seconds() / 60
        
        return {
            'session_id': session.session_id,
            'type': session.session_type.value,
            'state': session.state.value,
            'elapsed_minutes': elapsed,
            'expected_minutes': session.expected_duration,
            'continuity': session.continuity,
            'intensity': session.intensity,
            'interruptions': session.interruptions,
            'collapse_risk': session.collapse_risk,
            'fragmentation_risk': session.fragmentation_risk,
            'progress_pct': min(100, (elapsed / session.expected_duration) * 100)
        }
    
    def predict_session_outcome(self) -> Dict:
        """Predict likely session outcome."""
        if not self.active_session:
            return {'status': 'no_active_session'}
        
        session = self.active_session
        elapsed = (datetime.now() - session.start_time).total_seconds() / 60
        
        # Compute prediction
        template = self.session_templates.get(session.session_type, {})
        
        # Factors
        interrupt_threshold = template.get('interrupt_threshold', 3)
        collapse_threshold = template.get('collapse_risk_threshold', 0.4)
        
        if session.interruptions > interrupt_threshold:
            outcome = "likely_broken"
            confidence = 0.8
        elif session.collapse_risk > collapse_threshold:
            outcome = "at_risk"
            confidence = 0.7
        elif elapsed > session.expected_duration * 1.5:
            outcome = "overrunning"
            confidence = 0.6
        else:
            outcome = "on_track"
            confidence = 0.8
        
        return {
            'outcome': outcome,
            'confidence': confidence,
            'recommendation': self._get_session_recommendation(outcome, session)
        }
    
    def _get_session_recommendation(self, outcome: str, session: ExecutionSession) -> str:
        """Get recommendation based on outcome prediction."""
        if outcome == "likely_broken":
            return "Session is breaking. Consider natural end and restart later."
        elif outcome == "at_risk":
            return "Session at risk. Reduce complexity and focus on completion."
        elif outcome == "overrunning":
            return "Session exceeding expected duration. Assess if continuation is valuable."
        else:
            return "Session is healthy. Continue current trajectory."
    
    def get_session_history(self, days: int = 7) -> Dict:
        """Get session history summary."""
        since = datetime.now() - timedelta(days=days)
        recent = [s for s in self.sessions if s.start_time >= since]
        
        by_type = defaultdict(int)
        by_state = defaultdict(int)
        completion_rate = 0
        
        for session in recent:
            by_type[session.session_type.value] += 1
            by_state[session.state.value] += 1
            if session.state == SessionState.COMPLETED:
                completion_rate += 1
        
        return {
            'total_sessions': len(recent),
            'by_type': dict(by_type),
            'by_state': dict(by_state),
            'completion_rate': completion_rate / max(1, len(recent)),
            'avg_duration': np.mean([(s.end_time - s.start_time).total_seconds() / 60 
                                     for s in recent if s.end_time]) if recent else 0
        }


# ============================================================================
# GOAL VITALITY SYSTEM
# ============================================================================
"""
Goal Vitality System

Not just "goal exists".
Goal has vitality:
  - alive
  - stagnant
  - decaying
  - abandoned
  - toxic
  - overfit

Also:
  - Goal pressure propagation
  - Goal conflict detection
  - Goal reality coupling
"""

class GoalVitality(Enum):
    """Goal vitality states."""
    ALIVE = "alive"           # Active progress
    STAGNANT = "stagnant"     # No progress
    DECAYING = "decaying"     # Progress decreasing
    ABANDONED = "abandoned"   # Ignored
    TOXIC = "toxic"           # Harmful to trajectory
    OVERFIT = "overfit"       # Obsessive focus


@dataclass
class GoalVitalityMetrics:
    """Vitality metrics for goal."""
    goal_id: str
    
    # Core metrics
    progress_rate: float = 0.0       # How fast goal is advancing
    momentum: float = 0.0           # Is progress accelerating?
    alignment: float = 0.0          # Still aligned with strategy?
    
    # Health indicators
    age_days: float = 0.0
    last_activity_hours: float = 0.0
    progress_stagnation: float = 0.0  # Days since meaningful progress
    
    # Conflicts
    conflicts_with: List[str] = field(default_factory=list)
    resource_competition: float = 0.0
    
    # Derived
    vitality: GoalVitality = GoalVitality.ALIVE
    vitality_score: float = 0.5  # 0-1
    
    def compute_vitality(self):
        """Compute vitality from metrics."""
        # Progress rate factor
        if self.progress_rate < 0.1:
            if self.last_activity_hours > 48:
                self.vitality = GoalVitality.ABANDONED
            elif self.progress_stagnation > 7:
                self.vitality = GoalVitality.STAGNANT
            else:
                self.vitality = GoalVitality.DECAYING
        elif self.momentum < 0:
            self.vitality = GoalVitality.DECAYING
        
        # Conflict check
        if len(self.conflicts_with) > 0:
            if self.resource_competition > 0.7:
                self.vitality = GoalVitality.TOXIC
        
        # Overfit check (progress too high relative to quality)
        if self.progress_rate > 0.8 and self.alignment < 0.3:
            self.vitality = GoalVitality.OVERFIT
        
        # Compute vitality score
        self.vitality_score = (
            self.progress_rate * 0.3 +
            (1 - self.progress_stagnation / 14) * 0.2 +
            self.alignment * 0.3 +
            (1 - self.resource_competition) * 0.2
        )
        self.vitality_score = max(0, min(1, self.vitality_score))


class GoalVitalitySystem:
    """
    Goal Vitality System.
    
    Tracks goal health, detects problems, propagates pressure.
    """
    
    def __init__(self, event_runtime: EventRuntime, temporal_memory: TemporalMemory):
        self.event_runtime = event_runtime
        self.temporal_memory = temporal_memory
        
        self.goal_vitalities: Dict[str, GoalVitalityMetrics] = {}
        self.goal_pressures: Dict[str, float] = {}  # Meta goal pressure
        
        # Conflict detection
        self.conflict_pairs: List[Tuple[str, str]] = []
    
    def update_goal_vitality(self, goal_id: str, goal_data: Dict,
                            recent_events: List[RawEvent] = None):
        """Update vitality metrics for goal."""
        if goal_id not in self.goal_vitalities:
            self.goal_vitalities[goal_id] = GoalVitalityMetrics(goal_id=goal_id)
        
        metrics = self.goal_vitalities[goal_id]
        
        # Update from goal data
        if 'progress_rate' in goal_data:
            metrics.progress_rate = goal_data['progress_rate']
        if 'momentum' in goal_data:
            metrics.momentum = goal_data['momentum']
        if 'alignment' in goal_data:
            metrics.alignment = goal_data['alignment']
        if 'age_days' in goal_data:
            metrics.age_days = goal_data['age_days']
        
        # Update from events
        if recent_events:
            goal_events = [e for e in recent_events 
                          if goal_data.get('task_ids', []) and 
                          e.data.get('task_id') in goal_data.get('task_ids', [])]
            
            if goal_events:
                metrics.last_activity_hours = 0
            else:
                # No recent activity
                if metrics.last_activity_hours < 1000:
                    metrics.last_activity_hours += 1/60  # Assuming update every minute
        
        # Compute stagnation
        if metrics.progress_rate < 0.05:
            metrics.progress_stagnation += 1/60
        else:
            metrics.progress_stagnation = 0
        
        # Compute vitality
        metrics.compute_vitality()
    
    def propagate_meta_pressure(self, meta_goal_id: str, urgency: float):
        """Propagate meta goal pressure to child goals."""
        self.goal_pressures[meta_goal_id] = urgency
        
        # Pressure affects vitality scoring
        for goal_id, metrics in self.goal_vitalities.items():
            if metrics.alignment > 0.7:
                # High alignment = high pressure effect
                pressure_effect = urgency * metrics.alignment
                metrics.vitality_score = min(1.0, metrics.vitality_score + pressure_effect * 0.1)
    
    def detect_conflicts(self, goals: List[Dict]) -> List[Dict]:
        """Detect conflicts between goals."""
        conflicts = []
        
        # Check for resource conflicts
        for i, goal1 in enumerate(goals):
            for goal2 in goals[i+1:]:
                # Check for opposite directions
                if goal1.get('type') == 'build' and goal2.get('type') == 'destroy':
                    conflicts.append({
                        'type': 'directional',
                        'goals': [goal1['id'], goal2['id']],
                        'severity': 0.8,
                        'description': 'Goals have opposing directions'
                    })
                
                # Check for time competition
                if (goal1.get('deadline') and goal2.get('deadline') and
                    abs((goal1['deadline'] - goal2['deadline']).days) < 7):
                    conflicts.append({
                        'type': 'temporal',
                        'goals': [goal1['id'], goal2['id']],
                        'severity': 0.6,
                        'description': 'Goals compete for same time window'
                    })
                
                # Check for resource competition
                shared_resources = set(goal1.get('resources', [])) & set(goal2.get('resources', []))
                if shared_resources:
                    conflicts.append({
                        'type': 'resource',
                        'goals': [goal1['id'], goal2['id']],
                        'severity': 0.5 * len(shared_resources) / max(1, len(shared_resources)),
                        'shared_resources': list(shared_resources),
                        'description': f"Goals compete for {len(shared_resources)} shared resources"
                    })
        
        self.conflict_pairs = [(c['goals'][0], c['goals'][1]) for c in conflicts]
        return conflicts
    
    def get_dead_goals(self) -> List[str]:
        """Get goals that are effectively dead."""
        dead = []
        
        for goal_id, metrics in self.goal_vitalities.items():
            if metrics.vitality in [GoalVitality.ABANDONED, GoalVitality.STAGNANT]:
                if metrics.progress_stagnation > 14:  # 2 weeks
                    dead.append(goal_id)
        
        return dead
    
    def get_toxic_goals(self) -> List[str]:
        """Get goals that are toxic to trajectory."""
        toxic = []
        
        for goal_id, metrics in self.goal_vitalities.items():
            if metrics.vitality == GoalVitality.TOXIC:
                toxic.append(goal_id)
        
        return toxic
    
    def get_revival_candidates(self) -> List[Tuple[str, float]]:
        """Get goals that might benefit from revival."""
        candidates = []
        
        for goal_id, metrics in self.goal_vitalities.items():
            if metrics.vitality == GoalVitality.STAGNANT:
                if metrics.alignment > 0.5:
                    candidates.append((goal_id, metrics.alignment))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates


# ============================================================================
# REAL INTERVENTION LAYER
# ============================================================================
"""
Real Intervention Layer

Not analytics dashboard.
Executive interventions:

- "Don't open new project now"
- "Don't go into refactoring"
- "Your continuity is dropping"
- "This is ideal restart window"

The system must know when and how to intervene.
"""

class InterventionType(Enum):
    """Types of executive interventions."""
    PREVENT = "prevent"          # Block harmful action
    PROMOTE = "promote"          # Encourage beneficial action
    CORRECT = "correct"          # Steer back on track
    RECOVER = "recover"          # Help recover from break
    ADAPT = "adapt"              # Adjust to new reality


@dataclass
class Intervention:
    """Executive intervention."""
    intervention_id: str
    intervention_type: InterventionType
    
    trigger: str  # Why this intervention
    message: str   # What to tell user
    
    # Timing
    timestamp: datetime
    urgency: float  # 0-1
    
    # Context
    related_session: Optional[str] = None
    related_goal: Optional[str] = None
    
    # Result
    applied: bool = False
    effective: Optional[bool] = None  # Did it help?
    
    def to_dict(self) -> Dict:
        return {
            'id': self.intervention_id,
            'type': self.intervention_type.value,
            'trigger': self.trigger,
            'message': self.message,
            'urgency': self.urgency,
            'applied': self.applied,
            'effective': self.effective
        }


class InterventionLayer:
    """
    Real Intervention Layer.
    
    Decides when and how to intervene in user execution.
    """
    
    def __init__(self, event_runtime: EventRuntime,
                 session_model: SessionModel,
                 goal_vitality: GoalVitalitySystem,
                 momentum_engine):
        self.event_runtime = event_runtime
        self.session_model = session_model
        self.goal_vitality = goal_vitality
        self.momentum_engine = momentum_engine
        
        self.interventions: List[Intervention] = []
        self.intervention_counter = 0
        
        # Intervention rules
        self.rules: List[Dict] = []
        
        # Cooldown
        self.last_intervention = datetime.now()
        self.intervention_cooldown_minutes = 30
    
    def evaluate_interventions(self) -> List[Intervention]:
        """Evaluate if interventions are needed."""
        interventions = []
        now = datetime.now()
        
        # Check cooldown
        if (now - self.last_intervention).total_seconds() / 60 < self.intervention_cooldown_minutes:
            return interventions
        
        # 1. Session-based interventions
        session_summary = self.session_model.get_active_session_summary()
        if session_summary:
            outcome = self.session_model.predict_session_outcome()
            
            if outcome['outcome'] == 'likely_broken':
                interventions.append(self._create_intervention(
                    InterventionType.CORRECT,
                    trigger="Session breaking",
                    message="Your session is breaking. Consider wrapping up and taking a break.",
                    urgency=0.8
                ))
            
            elif outcome['outcome'] == 'at_risk':
                interventions.append(self._create_intervention(
                    InterventionType.ADAPT,
                    trigger="Session at risk",
                    message="Session at risk. Simplify what you're doing.",
                    urgency=0.6
                ))
        
        # 2. Momentum-based interventions
        momentum = self.momentum_engine.current_state
        
        if momentum.execution_capacity < 0.3:
            interventions.append(self._create_intervention(
                InterventionType.PREVENT,
                trigger="Low execution capacity",
                message="Your execution capacity is very low. Don't start complex tasks now.",
                urgency=0.7
            ))
        
        elif momentum.reactivation_cost > 0.7:
            interventions.append(self._create_intervention(
                InterventionType.RECOVER,
                trigger="High restart cost",
                message="High reactivation cost. Try to maintain current context rather than switching.",
                urgency=0.6
            ))
        
        # 3. Activity-based interventions
        activity = self.event_runtime.get_activity_summary(minutes=30)
        
        if activity['total_events'] == 0:
            interventions.append(self._create_intervention(
                InterventionType.PROMOTE,
                trigger="No recent activity",
                message="You haven't been active for a while. What would you like to work on?",
                urgency=0.4
            ))
        
        # 4. Goal-based interventions
        dead_goals = self.goal_vitality.get_dead_goals()
        if dead_goals:
            interventions.append(self._create_intervention(
                InterventionType.ADAPT,
                trigger="Dead goals detected",
                message=f"You have {len(dead_goals)} goals with no progress for 2+ weeks. Consider closing them.",
                urgency=0.5
            ))
        
        # Apply highest urgency
        if interventions:
            interventions.sort(key=lambda x: x.urgency, reverse=True)
            selected = interventions[0]
            self._apply_intervention(selected)
        
        return interventions
    
    def _create_intervention(self, intervention_type: InterventionType,
                            trigger: str, message: str, 
                            urgency: float) -> Intervention:
        """Create new intervention."""
        self.intervention_counter += 1
        
        return Intervention(
            intervention_id=f"int_{self.intervention_counter}",
            intervention_type=intervention_type,
            trigger=trigger,
            message=message,
            timestamp=datetime.now(),
            urgency=urgency
        )
    
    def _apply_intervention(self, intervention: Intervention):
        """Apply intervention."""
        self.interventions.append(intervention)
        self.last_intervention = datetime.now()
        
        # Record in timeline
        # (Would integrate with temporal memory)
    
    def record_outcome(self, intervention_id: str, effective: bool):
        """Record intervention effectiveness."""
        for intervention in self.interventions:
            if intervention.intervention_id == intervention_id:
                intervention.effective = effective
                break
    
    def get_intervention_history(self, days: int = 7) -> Dict:
        """Get intervention history."""
        since = datetime.now() - timedelta(days=days)
        recent = [i for i in self.interventions if i.timestamp >= since]
        
        applied = [i for i in recent if i.applied]
        effective = [i for i in applied if i.effective]
        
        return {
            'total_interventions': len(recent),
            'applied': len(applied),
            'effective': len(effective),
            'effectiveness_rate': len(effective) / max(1, len(applied)),
            'by_type': defaultdict(int, {i.intervention_type.value: 1 for i in applied})
        }


# ============================================================================
# INTEGRATED REALITY ENGINE
# ============================================================================

class RealityEngine:
    """
    Integrated Reality Engine.
    
    Bridges strategic layer with execution reality:
    
      Goals
        ↕
      Trajectory
        ↕
      Momentum
        ↕
      Environment
        ↕
      Execution Reality
    """
    
    def __init__(self):
        # Core systems
        self.event_runtime = EventRuntime()
        self.temporal_memory = TemporalMemory()
        
        # Models
        self.session_model = SessionModel(self.temporal_memory)
        self.goal_vitality = GoalVitalitySystem(self.event_runtime, self.temporal_memory)
        
        # Momentum (simplified reference)
        self.momentum_engine = None  # Would integrate with Strategic Execution Engine
        
        # Interventions
        self.intervention_layer = InterventionLayer(
            self.event_runtime, self.session_model, 
            self.goal_vitality, self.momentum_engine
        )
        
    def connect_momentum_engine(self, momentum_engine):
        """Connect momentum engine from Strategic Execution Engine."""
        self.momentum_engine = momentum_engine
        self.intervention_layer.momentum_engine = momentum_engine
    
    def run_cycle(self) -> Dict:
        """Run reality engine cycle."""
        # 1. Gather telemetry
        activity = self.event_runtime.get_activity_summary(minutes=60)
        
        # 2. Evaluate interventions
        interventions = self.intervention_layer.evaluate_interventions()
        
        # 3. Get session status
        session = self.session_model.get_active_session_summary()
        
        # 4. Get goal vitality
        dead_goals = self.goal_vitality.get_dead_goals()
        toxic_goals = self.goal_vitality.get_toxic_goals()
        revival_candidates = self.goal_vitality.get_revival_candidates()
        
        # 5. Get temporal patterns
        success = self.temporal_memory.analyze_success_patterns()
        failures = self.temporal_memory.analyze_failure_patterns()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'activity_summary': activity,
            'active_session': session,
            'interventions': [i.to_dict() for i in interventions],
            'goal_vitality': {
                'dead_goals': dead_goals,
                'toxic_goals': toxic_goals,
                'revival_candidates': [g for g, _ in revival_candidates[:3]]
            },
            'temporal_patterns': {
                'success_insights': success.get('insights', []),
                'failure_warnings': [w['message'] for w in failures.get('warnings', [])]
            },
            'recommendations': self._generate_recommendations(
                activity, session, interventions, dead_goals, failures
            )
        }
    
    def _generate_recommendations(self, activity, session, interventions,
                                  dead_goals, failures) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Based on interventions
        if interventions:
            recommendations.append({
                'type': 'intervention',
                'priority': 'high',
                'message': interventions[0].message
            })
        
        # Based on session
        if session and session.get('collapse_risk', 0) > 0.5:
            recommendations.append({
                'type': 'session',
                'priority': 'medium',
                'message': "Session at risk. Consider wrapping up."
            })
        
        # Based on failures
        for warning in failures.get('warnings', [])[:2]:
            recommendations.append({
                'type': 'pattern_warning',
                'priority': 'medium',
                'message': warning['message']
            })
        
        # Based on dead goals
        if dead_goals:
            recommendations.append({
                'type': 'goal_maintenance',
                'priority': 'low',
                'message': f"Consider closing {len(dead_goals)} stagnant goals."
            })
        
        return recommendations
    
    # Telemetry ingestion methods
    def ingest_git_commit(self, repo: str, message: str, files: List[str]):
        """Ingest git commit."""
        self.event_runtime.ingest_git('commit', repo, {
            'message': message,
            'files': files
        })
        self.temporal_memory.record_action(
            f"Committed to {repo}: {message[:50]}",
            {'repo': repo, 'files': files}
        )
    
    def ingest_ide_activity(self, action: str, file: str, details: Dict = None):
        """Ingest IDE activity."""
        self.event_runtime.ingest_ide(action, file, details or {})
        
        if action == 'focused_work':
            self.session_model.record_event('focused_work', {'file': file})
    
    def ingest_terminal_command(self, command: str, exit_code: int):
        """Ingest terminal command."""
        self.event_runtime.ingest_terminal(command, '', exit_code)
        
        if exit_code == 0:
            self.temporal_memory.record_action(f"Ran: {command[:50]}")
    
    def start_deep_work_session(self, goal_id: str = None, tasks: List[str] = None):
        """Start deep work session."""
        session = self.session_model.start_session(
            SessionType.DEEP_WORK, goal_id, tasks
        )
        return session.session_id
    
    def record_interruption(self, reason: str):
        """Record execution interruption."""
        self.session_model.record_event('interruption', {'reason': reason})
        self.temporal_memory.record_interruption(reason)


# ============================================================================
# TESTS
# ============================================================================

def test_event_runtime():
    """Test Event Runtime."""
    print("\n" + "=" * 60)
    print("EVENT RUNTIME TEST")
    print("=" * 60)
    
    runtime = EventRuntime()
    
    # Simulate events
    for i in range(20):
        runtime.ingest_git('commit', 'ai-os', {
            'message': f'Commit {i}',
            'files': [f'file{j}.py' for j in range(3)]
        })
        
        if i % 5 == 0:
            runtime.ingest_ide('focus', f'module{i}.py', {})
    
    summary = runtime.get_activity_summary(minutes=60)
    print(f"\n  Activity summary: {summary}")
    
    print(f"\n  Events by source: {list(runtime.streams.keys())}")


def test_temporal_memory():
    """Test Temporal Memory."""
    print("\n" + "=" * 60)
    print("TEMPORAL MEMORY TEST")
    print("=" * 60)
    
    memory = TemporalMemory()
    
    # Simulate execution
    action_id = memory.record_action("Implemented feature X")
    memory.record_consequence("Feature working", caused_by=action_id, momentum_impact=0.2)
    
    memory.record_interruption("Meeting")
    memory.record_recovery("Resumed work", from_interruption=memory.timeline[-1].timestamp.isoformat())
    
    memory.record_milestone("Feature shipped", goal_id="goal_1")
    
    # Simulate fragmentation
    for _ in range(5):
        memory.record_interruption("Slack notification")
    
    summary = memory.get_timeline_summary()
    print(f"\n  Timeline summary: {summary}")
    
    failures = memory.analyze_failure_patterns()
    print(f"\n  Failure patterns: {failures}")


def test_session_model():
    """Test Session Model."""
    print("\n" + "=" * 60)
    print("SESSION MODEL TEST")
    print("=" * 60)
    
    memory = TemporalMemory()
    model = SessionModel(memory)
    
    # Start session
    session = model.start_session(SessionType.DEEP_WORK, goal_id="goal_1")
    print(f"\n  Started session: {session.session_id}")
    
    # Simulate work
    for i in range(15):
        if i % 4 == 0 and i > 0:
            model.record_event('interruption', {'reason': 'context_switch'})
        else:
            model.record_event('focused_work', {'task': f'task_{i}'})
    
    summary = model.get_active_session_summary()
    print(f"\n  Session summary: {summary}")
    
    outcome = model.predict_session_outcome()
    print(f"\n  Predicted outcome: {outcome}")


def test_goal_vitality():
    """Test Goal Vitality System."""
    print("\n" + "=" * 60)
    print("GOAL VITALITY TEST")
    print("=" * 60)
    
    runtime = EventRuntime()
    memory = TemporalMemory()
    vitality = GoalVitalitySystem(runtime, memory)
    
    # Update goals
    vitality.update_goal_vitality('goal_1', {
        'progress_rate': 0.3,
        'momentum': 0.1,
        'alignment': 0.8,
        'age_days': 5
    })
    
    vitality.update_goal_vitality('goal_2', {
        'progress_rate': 0.05,
        'momentum': -0.1,
        'alignment': 0.3,
        'age_days': 14
    })
    
    # Detect conflicts
    goals = [
        {'id': 'goal_1', 'type': 'build', 'deadline': datetime.now() + timedelta(days=7)},
        {'id': 'goal_2', 'type': 'refactor', 'deadline': datetime.now() + timedelta(days=3)}
    ]
    conflicts = vitality.detect_conflicts(goals)
    
    print(f"\n  Vitalities: {len(vitality.goal_vitalities)}")
    print(f"  Conflicts: {len(conflicts)}")
    
    dead = vitality.get_dead_goals()
    print(f"\n  Dead goals: {dead}")


def test_intervention_layer():
    """Test Intervention Layer."""
    print("\n" + "=" * 60)
    print("INTERVENTION LAYER TEST")
    print("=" * 60)
    
    runtime = EventRuntime()
    memory = TemporalMemory()
    session_model = SessionModel(memory)
    goal_vitality = GoalVitalitySystem(runtime, memory)
    
    # Mock momentum engine
    class MockMomentum:
        def __init__(self):
            self.current_state = MockMomentumState()
    
    class MockMomentumState:
        execution_capacity = 0.4
        reactivation_cost = 0.5
    
    mock_momentum = MockMomentum()
    
    layer = InterventionLayer(runtime, session_model, goal_vitality, mock_momentum)
    
    # Evaluate interventions
    interventions = layer.evaluate_interventions()
    print(f"\n  Interventions triggered: {len(interventions)}")


def test_reality_engine():
    """Test Reality Engine."""
    print("\n" + "=" * 60)
    print("REALITY ENGINE TEST")
    print("=" * 60)
    
    engine = RealityEngine()
    
    # Simulate work
    engine.ingest_git_commit('ai-os', 'Add feature', ['file1.py', 'file2.py'])
    engine.ingest_ide_activity('focused_work', 'file1.py')
    
    # Start session
    session_id = engine.start_deep_work_session(goal_id='goal_1')
    print(f"\n  Session started: {session_id}")
    
    # Simulate interruptions
    engine.record_interruption("Meeting")
    
    # Run cycle
    result = engine.run_cycle()
    
    print(f"\n  Activity: {result['activity_summary']['total_events']} events")
    print(f"  Session: {result['active_session'] is not None}")
    print(f"  Interventions: {len(result['interventions'])}")
    print(f"  Recommendations: {len(result['recommendations'])}")


if __name__ == "__main__":
    test_event_runtime()
    test_temporal_memory()
    test_session_model()
    test_goal_vitality()
    test_intervention_layer()
    test_reality_engine()
    
    print("\n" + "=" * 60)
    print("AI-OS REALITY ENGINE")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Strategic heuristic layer
  To: Adaptive Strategic Cognitive Runtime with reality feedback
  
CORE PRINCIPLE: 
  AI-OS must connect to real execution data.
  Without reality feedback, goal hierarchy is mostly symbolic.
  
PRIORITY: Not new cognitive modules.
PRIORITY: Reality feedback loop.

COMPONENTS:

1. EVENT RUNTIME
   - Captures real execution telemetry
   - Git, IDE, terminal, browser, sleep events
   - Behavioral streams from real systems
   
2. TEMPORAL MEMORY
   - Causal behavior history
   - Execution chains, interruption chains
   - Success/failure pattern detection
   
3. EXECUTION SESSION MODEL
   - Deep work, recovery, exploration, deployment sessions
   - Session state tracking (emerging, building, sustained, degrading, broken)
   - Collapse risk prediction
   
4. GOAL VITALITY SYSTEM
   - Alive, stagnant, decaying, abandoned, toxic, overfit
   - Goal pressure propagation
   - Conflict detection
   - Dead goal detection
   
5. REAL INTERVENTION LAYER
   - Executive interventions
   - "Don't open new project now"
   - "Your continuity is dropping"
   - "This is ideal restart window"

GOAL: Bridge strategic layer with execution reality

Goals
  ↕
Trajectory
  ↕
Momentum
  ↕
Environment
  ↕
Execution Reality

This makes AI-OS a living system, not a simulation.
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
AI-OS Reality Engine Summary

Key insight: Goals are not the top of a tree.
Goals are part of a dynamic system.

What this adds:

1. EVENT RUNTIME
   - Real telemetry from execution environment
   - Git, IDE, terminal, browser events
   - Activity streams

2. TEMPORAL MEMORY
   - Causal chains: action -> consequence -> trajectory
   - Success/failure pattern recognition
   - Interruption and recovery tracking

3. EXECUTION SESSION MODEL
   - Session types: deep work, recovery, exploration, deployment
   - Session state tracking
   - Collapse risk prediction

4. GOAL VITALITY SYSTEM
   - Goal health tracking
   - Dead, toxic, stagnant detection
   - Conflict detection
   - Pressure propagation

5. REAL INTERVENTION LAYER
   - Executive interventions
   - Behavioral guidance
   - Outcome tracking

This transforms AI-OS from:
  strategic heuristic layer → adaptive strategic cognitive runtime
  
The system now:
  - Connects to real execution data
  - Tracks causal behavior chains
  - Detects session problems
  - Identifies dead/toxic goals
  - Provides real interventions
  
Not a simulation. A living system.
"""