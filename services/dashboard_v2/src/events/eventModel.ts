/**
 * v2 UI - Event Model
 *
 * Formal event contract between UI and Core System
 * Following the specification from v2 UI contract
 */

import { NodeId, GoalId, NodeType } from '../types';

// ============================================================================
// UI → SYSTEM EVENTS (User Actions)
// ============================================================================

/**
 * User clicks on a node to inspect it
 */
export interface SelectNodeEvent {
  type: 'SELECT_NODE';
  nodeId: NodeId;
  nodeType: NodeType;
}

/**
 * User changes thinking mode
 */
export interface ChangeModeEvent {
  type: 'CHANGE_MODE';
  mode: 'explore' | 'exploit' | 'reflect';
}

/**
 * User applies visualization overlay
 */
export interface ApplyOverlayEvent {
  type: 'APPLY_OVERLAY';
  overlayType: 'none' | 'heatmap' | 'conflicts' | 'memory_traces' | 'simulation';
}

/**
 * User clears current overlay
 */
export interface ClearOverlayEvent {
  type: 'CLEAR_OVERLAY';
}

/**
 * User jumps to specific point in timeline
 */
export interface TimelineJumpEvent {
  type: 'TIMELINE_JUMP';
  timestamp: string; // ISO timestamp
}

/**
 * User requests goal decomposition
 */
export interface RequestDecomposeEvent {
  type: 'REQUEST_DECOMPOSE';
  goalId: GoalId;
  options?: {
    maxDepth?: number;
    preserveConstraints?: boolean;
  };
}

/**
 * User requests simulation of execution path
 */
export interface RequestSimulationEvent {
  type: 'REQUEST_SIMULATION';
  nodeId: NodeId;
  options?: {
    duration?: number;
    alternatives?: boolean;
  };
}

/**
 * User manually overrides a decision (DANGEROUS)
 */
export interface OverrideDecisionEvent {
  type: 'OVERRIDE_DECISION';
  decisionId: string;
  reason?: string;
}

/**
 * User updates global constraints
 */
export interface ConstraintUpdateEvent {
  type: 'CONSTRAINT_UPDATE';
  constraintType: 'ethics' | 'budget' | 'time';
  value: any;
}

/**
 * Union type of all UI events
 */
export type UIEvent =
  | SelectNodeEvent
  | ChangeModeEvent
  | ApplyOverlayEvent
  | ClearOverlayEvent
  | TimelineJumpEvent
  | RequestDecomposeEvent
  | RequestSimulationEvent
  | OverrideDecisionEvent
  | ConstraintUpdateEvent;

// ============================================================================
// SYSTEM → UI EVENTS (System Responses)
// ============================================================================

/**
 * Graph structure changed (nodes/edges added/updated/removed)
 */
export interface GraphUpdatedEvent {
  type: 'GRAPH_UPDATED';
  diff: {
    addedNodes: NodeId[];
    updatedNodes: Partial<any>[];
    removedNodes: NodeId[];
    addedEdges: any[];
    removedEdges: string[];
  };
}

/**
 * Goal status changed (pending → active → done)
 */
export interface GoalStatusChangedEvent {
  type: 'GOAL_STATUS_CHANGED';
  goalId: GoalId;
  oldStatus: 'pending' | 'active' | 'blocked';
  newStatus: 'active' | 'done' | 'blocked';
  timestamp: string;
}

/**
 * System detected a conflict
 */
export interface ConflictDetectedEvent {
  type: 'CONFLICT_DETECTED';
  conflictId: string;
  goalA: GoalId;
  goalB: GoalId;
  severity: number; // 0..1
  description: string;
}

/**
 * Simulation completed with results
 */
export interface SimulationResultEvent {
  type: 'SIMULATION_RESULT';
  simulationId: string;
  path: NodeId[];
  score: number;
  duration: number;
  success: boolean;
  result?: any;
}

/**
 * Execution progress update
 */
export interface ExecutionProgressEvent {
  type: 'EXECUTION_PROGRESS';
  nodeId: NodeId;
  progress: number; // 0..1
  phase?: 'planning' | 'execution' | 'verification';
}

/**
 * System error occurred
 */
export interface SystemErrorEvent {
  type: 'ERROR';
  reason: string;
  context: {
    component?: string;
    nodeId?: NodeId;
    stack?: string;
    metadata?: Record<string, any>;
  };
}

/**
 * Union type of all system events
 */
export type SystemEvent =
  | GraphUpdatedEvent
  | GoalStatusChangedEvent
  | ConflictDetectedEvent
  | SimulationResultEvent
  | ExecutionProgressEvent
  | SystemErrorEvent;

// ============================================================================
// EVENT HELPERS
// ============================================================================

/**
 * Type guard to check if event is a UI event
 */
export function isUIEvent(event: any): event is UIEvent {
  return event && typeof event.type === 'string' &&
    ['SELECT_NODE', 'CHANGE_MODE', 'APPLY_OVERLAY', 'CLEAR_OVERLAY',
     'TIMELINE_JUMP', 'REQUEST_DECOMPOSE', 'REQUEST_SIMULATION',
     'OVERRIDE_DECISION', 'CONSTRAINT_UPDATE'].includes(event.type);
}

/**
 * Type guard to check if event is a System event
 */
export function isSystemEvent(event: any): event is SystemEvent {
  return event && typeof event.type === 'string' &&
    ['GRAPH_UPDATED', 'GOAL_STATUS_CHANGED', 'CONFLICT_DETECTED',
     'SIMULATION_RESULT', 'EXECUTION_PROGRESS', 'ERROR'].includes(event.type);
}

/**
 * Helper to create UI events
 */
export const UIEvents = {
  selectNode: (nodeId: NodeId, nodeType: NodeType): SelectNodeEvent => ({
    type: 'SELECT_NODE',
    nodeId,
    nodeType,
  }),

  changeMode: (mode: 'explore' | 'exploit' | 'reflect'): ChangeModeEvent => ({
    type: 'CHANGE_MODE',
    mode,
  }),

  applyOverlay: (overlayType: ApplyOverlayEvent['overlayType']): ApplyOverlayEvent => ({
    type: 'APPLY_OVERLAY',
    overlayType,
  }),

  clearOverlay: (): ClearOverlayEvent => ({
    type: 'CLEAR_OVERLAY',
  }),

  timelineJump: (timestamp: string): TimelineJumpEvent => ({
    type: 'TIMELINE_JUMP',
    timestamp,
  }),

  requestDecompose: (goalId: GoalId, options?: RequestDecomposeEvent['options']): RequestDecomposeEvent => ({
    type: 'REQUEST_DECOMPOSE',
    goalId,
    options,
  }),

  requestSimulation: (nodeId: NodeId, options?: RequestSimulationEvent['options']): RequestSimulationEvent => ({
    type: 'REQUEST_SIMULATION',
    nodeId,
    options,
  }),

  overrideDecision: (decisionId: string, reason?: string): OverrideDecisionEvent => ({
    type: 'OVERRIDE_DECISION',
    decisionId,
    reason,
  }),

  updateConstraint: (constraintType: ConstraintUpdateEvent['constraintType'], value: any): ConstraintUpdateEvent => ({
    type: 'CONSTRAINT_UPDATE',
    constraintType,
    value,
  }),
};

/**
 * Helper to create System events
 */
export const SystemEvents = {
  graphUpdated: (diff: GraphUpdatedEvent['diff']): GraphUpdatedEvent => ({
    type: 'GRAPH_UPDATED',
    diff,
  }),

  goalStatusChanged: (goalId: GoalId, oldStatus: GoalStatusChangedEvent['oldStatus'], newStatus: GoalStatusChangedEvent['newStatus']): GoalStatusChangedEvent => ({
    type: 'GOAL_STATUS_CHANGED',
    goalId,
    oldStatus,
    newStatus,
    timestamp: new Date().toISOString(),
  }),

  conflictDetected: (conflictId: string, goalA: GoalId, goalB: GoalId, severity: number, description: string): ConflictDetectedEvent => ({
    type: 'CONFLICT_DETECTED',
    conflictId,
    goalA,
    goalB,
    severity,
    description,
  }),

  simulationResult: (simulationId: string, path: NodeId[], score: number, duration: number, success: boolean, result?: any): SimulationResultEvent => ({
    type: 'SIMULATION_RESULT',
    simulationId,
    path,
    score,
    duration,
    success,
    result,
  }),

  executionProgress: (nodeId: NodeId, progress: number, phase?: ExecutionProgressEvent['phase']): ExecutionProgressEvent => ({
    type: 'EXECUTION_PROGRESS',
    nodeId,
    progress,
    phase,
  }),

  error: (reason: string, context: SystemErrorEvent['context']): SystemErrorEvent => ({
    type: 'ERROR',
    reason,
    context,
  }),
};
