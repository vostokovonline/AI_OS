/**
 * v2 UI - Zustand Store
 *
 * State machine implementation for UI state management
 * Follows the formal specification from the v2 UI contract
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

interface UIStore extends UIState {
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
  setGraphView: (view: Partial<UIState['graph']>) => void;
  reset: () => void;
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

  dispatchEvent: (event: UIEvent) => {
    const state = get();

    switch (event.type) {
      case 'SELECT_NODE':
        // Cannot select nodes when override is enabled
        if (state.override.enabled) {
          console.warn('[UI] Cannot select node while override is active');
          return;
        }
        set({
          focus: {
            nodeId: event.nodeId,
            nodeType: null, // Will be fetched from graph
          },
        });
        break;

      case 'CHANGE_MODE':
        // Validate mode transitions
        if (state.mode === 'exploit' && event.mode === 'explore') {
          console.warn('[UI] Cannot switch from exploit to explore directly');
          return;
        }
        set({ mode: event.mode });
        break;

      case 'APPLY_OVERLAY':
        // Cannot have overlay + simulation simultaneously
        if (state.overlay === 'simulation') {
          console.warn('[UI] Cannot apply overlay during simulation');
          return;
        }
        set({ overlay: (event as any).overlayType });
        break;

      case 'CLEAR_OVERLAY':
        set({ overlay: 'none' });
        break;

      case 'TIMELINE_JUMP':
        // Timeline jump restores UI state from that moment
        set({
          timelineCursor: event.timestamp,
        });
        break;

      case 'REQUEST_DECOMPOSE':
      case 'REQUEST_SIMULATION':
      case 'OVERRIDE_DECISION':
      case 'CONSTRAINT_UPDATE':
        // These events are forwarded to the API layer
        break;

      default:
        console.warn('[UI] Unknown event type:', event);
    }
  },

  handleSystemEvent: (event: SystemEvent) => {
    switch (event.type) {
      case 'GRAPH_UPDATED':
        // Handled by graph store
        break;

      case 'GOAL_STATUS_CHANGED':
        // Update focus if it's the selected goal
        const state = get();
        if (state.focus.nodeId === event.goalId) {
          // Refetch node data
        }
        break;

      case 'CONFLICT_DETECTED':
        // Auto-switch to conflict overlay if severity is high
        if (event.severity > 0.7) {
          set({ overlay: 'conflicts' });
        }
        break;

      case 'SIMULATION_RESULT':
        set({ overlay: 'simulation' });
        break;

      case 'EXECUTION_PROGRESS':
        // Update progress bar
        break;

      case 'ERROR':
        console.error('[UI] System error:', event.reason, event.context);
        break;

      default:
        console.warn('[UI] Unknown system event:', event);
    }
  },

  setFocus: (nodeId, nodeType) => {
    set({
      focus: { nodeId, nodeType },
    });
  },

  setMode: (mode) => {
    set({ mode });
  },

  setView: (view) => {
    set({ view });
  },

  setOverlay: (overlay) => {
    set({ overlay });
  },

  clearOverlay: () => {
    set({ overlay: 'none' });
  },

  setTimelineCursor: (timestamp) => {
    set({ timelineCursor: timestamp });
  },

  updateConstraints: (constraints) => {
    set((state) => ({
      constraints: {
        ...state.constraints,
        ...constraints,
      },
    }));
  },

  setOverride: (enabled, decisionId) => {
    set({
      override: {
        enabled,
        decisionId: decisionId || null,
      },
    });
  },

  setGraphView: (view) => {
    set((state) => ({
      graph: {
        ...state.graph,
        ...view,
      },
    }));
  },

  reset: () => {
    set(initialState);
  },
}));
