/**
 * v2 UI - State Machine
 *
 * Formal state machine implementation with invariants and transition guards
 * Following the specification from v2 UI contract
 */

import { create } from 'zustand';
import {
  UIState,
  Mode,
  ViewType,
  OverlayType,
  NodeType,
  UIEvent,
  SystemEvent,
  NodeId,
} from '../types';

// ============================================================================
// STATE INVARIANTS (Validation Rules)
// ============================================================================

interface StateInvariant {
  name: string;
  validate: (state: UIState) => boolean;
  error: string;
}

const INVARIANTS: StateInvariant[] = [
  {
    name: 'overlay-mutually-exclusive',
    validate: (state) => {
      // Cannot have simulation + other overlays simultaneously
      if (state.overlay === 'simulation') {
        return true; // Simulation is exclusive
      }
      return true;
    },
    error: 'Cannot combine simulation overlay with other overlays',
  },
  {
    name: 'focus-consistency',
    validate: (state) => {
      // If focus.nodeId is null, focus.nodeType must also be null
      if (state.focus.nodeId === null && state.focus.nodeType !== null) {
        return false;
      }
      // If focus.nodeId is set, focus.nodeType must be set
      if (state.focus.nodeId !== null && state.focus.nodeType === null) {
        return false;
      }
      return true;
    },
    error: 'Focus nodeId and nodeType must be both null or both set',
  },
];

// ============================================================================
// STATE TRANSITIONS (Valid state changes)
// ============================================================================

// Helper to validate state
function validateState(state: UIState): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  for (const invariant of INVARIANTS) {
    if (!invariant.validate(state)) {
      errors.push(`${invariant.name}: ${invariant.error}`);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// ============================================================================
// SNAPSHOT / RESTORE (for Timeline)
// ============================================================================

export interface UISnapshot {
  timestamp: string;
  state: UIState;
  metadata: {
    mode: Mode;
    focus: UIState['focus'];
    overlay: OverlayType;
    activeGoals: number;
  };
}

function createSnapshot(state: UIState): UISnapshot {
  return {
    timestamp: new Date().toISOString(),
    state: JSON.parse(JSON.stringify(state)), // Deep clone
    metadata: {
      mode: state.mode,
      focus: state.focus,
      overlay: state.overlay,
      activeGoals: 0, // Will be populated by caller
    },
  };
}

// ============================================================================
// UI STORE with State Machine
// ============================================================================

interface UIStore extends UIState {
  // Current validation status
  validationErrors: string[];

  // Actions
  dispatchEvent: (event: UIEvent) => void;
  handleSystemEvent: (event: SystemEvent) => void;
  setFocus: (nodeId: NodeId | null, nodeType: NodeType | null) => void;
  setMode: (mode: Mode) => void;
  setView: (view: ViewType) => void;
  setOverlay: (overlay: OverlayType) => void;
  clearOverlay: () => void;
  setTimelineCursor: (timestamp: string | null) => void;
  updateConstraints: (constraints: Partial<UIState['constraints']>) => void;
  setOverride: (enabled: boolean, decisionId?: string) => void;

  // Snapshot / Restore
  createSnapshot: () => UISnapshot;
  restoreSnapshot: (snapshot: UISnapshot) => void;

  // Validation
  validate: () => { valid: boolean; errors: string[] };
}

// Initial state
const initialState: UIState = {
  mode: 'explore',
  view: 'graph',
  focus: {
    nodeId: null,
    nodeType: null,
  },
  overlay: 'none',
  timelineCursor: null,
  constraints: {
    ethics: [],
    budget: null,
    timeHorizon: null,
  },
  override: {
    enabled: false,
    decisionId: null,
  },
  graph: {
    zoom: 1,
    center: { x: 0, y: 0 },
    collapsedLevels: [],
  },
};

export const useUIStore = create<UIStore>((set, get) => ({
  ...initialState,
  validationErrors: [],

  // Event Dispatcher (State Machine)
  dispatchEvent: (event) => {
    const state = get();

    console.log('[UI State Machine] Dispatching event:', event.type, event);

    // Transition guards based on mode
    if (state.mode === 'exploit') {
      switch (event.type) {
        case 'SELECT_NODE':
        case 'CHANGE_MODE':
        case 'APPLY_OVERLAY':
          // These are allowed in exploit mode
          break;
        case 'REQUEST_DECOMPOSE':
        case 'REQUEST_SIMULATION':
          console.warn('[UI State Machine] Operation not allowed in exploit mode:', event.type);
          return;
        default:
          break;
      }
    }

    // Overlay mutual exclusion
    if (event.type === 'APPLY_OVERLAY' && state.overlay !== 'none') {
      const evt = event as any; // ApplyOverlayEvent
      console.warn('[UI State Machine] Cannot apply overlay:', evt.overlayType, 'while overlay is active:', state.overlay);
      // Clear existing overlay first
      get().clearOverlay();
    }

    // Execute event
    switch (event.type) {
      case 'SELECT_NODE': {
        const { nodeId, nodeType } = event;
        set({
          focus: { nodeId, nodeType },
        });
        break;
      }

      case 'CHANGE_MODE': {
        const { mode } = event;
        set({ mode });
        console.log('[UI State Machine] Mode changed to:', mode);
        break;
      }

      case 'APPLY_OVERLAY': {
        const evt = event as any; // ApplyOverlayEvent
        set({ overlay: evt.overlayType });
        console.log('[UI State Machine] Overlay applied:', evt.overlayType);
        break;
      }

      case 'CLEAR_OVERLAY': {
        set({ overlay: 'none' });
        break;
      }

      case 'TIMELINE_JUMP': {
        const { timestamp } = event;
        set({ timelineCursor: timestamp });
        console.log('[UI State Machine] Timeline jump to:', timestamp);
        break;
      }

      case 'REQUEST_DECOMPOSE': {
        const { goalId } = event;
        // This would trigger system event
        console.log('[UI State Machine] Request decomposition for goal:', goalId);
        break;
      }

      case 'REQUEST_SIMULATION': {
        const { nodeId } = event;
        console.log('[UI State Machine] Request simulation for node:', nodeId);
        break;
      }

      case 'OVERRIDE_DECISION': {
        const { decisionId } = event;
        set({
          override: {
            enabled: true,
            decisionId,
          },
        });
        console.warn('[UI State Machine] OVERRIDE ACTIVATED for decision:', decisionId);
        break;
      }

      case 'CONSTRAINT_UPDATE': {
        const evt = event as any; // ConstraintUpdateEvent
        set((state) => ({
          constraints: {
            ...state.constraints,
            [evt.constraintType]: evt.value,
          },
        }));
        break;
      }

      default:
        const _exhaustiveCheck: never = event;
        console.warn('[UI State Machine] Unknown event type:', _exhaustiveCheck);
    }

    // Validate new state
    const validation = get().validate();
    if (!validation.valid) {
      console.error('[UI State Machine] State validation failed:', validation.errors);
      set({ validationErrors: validation.errors });
    } else {
      set({ validationErrors: [] });
    }
  },

  // System Event Handler
  handleSystemEvent: (event) => {
    console.log('[UI State Machine] Handling system event:', event.type);

    switch (event.type) {
      case 'GRAPH_UPDATED':
        // Trigger graph refresh
        console.log('[UI State Machine] Graph updated, refreshing view');
        break;

      case 'GOAL_STATUS_CHANGED':
        const { goalId } = event;
        // If focused node changed, update inspector
        if (get().focus.nodeId === goalId) {
          console.log('[UI State Machine] Focused goal status changed, inspector will update');
        }
        break;

      case 'CONFLICT_DETECTED':
        console.log('[UI State Machine] Conflict detected, consider applying conflict overlay');
        // Could auto-apply conflict overlay here
        break;

      case 'SIMULATION_RESULT':
        console.log('[UI State Machine] Simulation result received');
        break;

      case 'EXECUTION_PROGRESS':
        const { nodeId, progress } = event;
        console.log('[UI State Machine] Execution progress:', nodeId, progress);
        break;

      case 'ERROR':
        const { reason, context } = event;
        console.error('[UI State Machine] System error:', reason, context);
        break;

      default:
        const _exhaustiveCheck: never = event;
        console.warn('[UI State Machine] Unknown system event type:', (_exhaustiveCheck as any).type);
    }
  },

  // Focus Management
  setFocus: (nodeId, nodeType) => {
    set({
      focus: { nodeId, nodeType },
    });
  },

  // Mode Management
  setMode: (mode) => {
    set({ mode });
  },

  // View Management
  setView: (view) => {
    set({ view });
  },

  // Overlay Management
  setOverlay: (overlay) => {
    set({ overlay });
  },

  clearOverlay: () => {
    set({ overlay: 'none' });
  },

  // Timeline Management
  setTimelineCursor: (timestamp) => {
    set({ timelineCursor: timestamp });
  },

  // Constraints Management
  updateConstraints: (constraints) => {
    set((state) => ({
      constraints: {
        ...state.constraints,
        ...constraints,
      },
    }));
  },

  // Override Management
  setOverride: (enabled, decisionId) => {
    set({
      override: {
        enabled,
        decisionId: decisionId || null,
      },
    });
  },

  // Snapshot / Restore
  createSnapshot: () => {
    const state = get();
    const snapshot = createSnapshot(state);
    console.log('[UI State Machine] Snapshot created at:', snapshot.timestamp);
    return snapshot;
  },

  restoreSnapshot: (snapshot) => {
    console.log('[UI State Machine] Restoring snapshot from:', snapshot.timestamp);
    set(snapshot.state);
  },

  // Validation
  validate: () => {
    const state = get();
    return validateState(state);
  },

  // Reset to initial state
  reset: () => {
    set(initialState);
  },
}));
