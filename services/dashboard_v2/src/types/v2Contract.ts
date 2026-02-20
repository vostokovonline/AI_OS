/**
 * v2 UI - Complete Contract Export
 *
 * Central export point for all v2 UI types and contracts
 * This is the formal specification for implementation
 */

// State Machine
export { useUIStore } from '../store/uiStateMachine';
export type { UISnapshot } from '../store/uiStateMachine';

// Event Model
export {
  isUIEvent,
  isSystemEvent,
  UIEvents,
  SystemEvents,
} from '../events/eventModel';
export type {
  UIEvent,
  SystemEvent,
  SelectNodeEvent,
  ChangeModeEvent,
  ApplyOverlayEvent,
  ClearOverlayEvent,
  TimelineJumpEvent,
  RequestDecomposeEvent,
  RequestSimulationEvent,
  OverrideDecisionEvent,
  ConstraintUpdateEvent,
  GraphUpdatedEvent,
  GoalStatusChangedEvent,
  ConflictDetectedEvent,
  SimulationResultEvent,
  ExecutionProgressEvent,
  SystemErrorEvent,
} from '../events/eventModel';

// Core Binding
export {
  coreGoalToUINode,
  uiNodeToCoreGoal,
  createGraphDiff,
} from '../types/coreBinding';
export type {
  GoalNode,
  AgentNode,
  SkillNode,
  MemoryNode,
  TestNode,
  Node,
} from '../types/coreBinding';
export type {
  GraphEdge,
  GraphDiff,
  InspectorContext,
  ExecutionStep,
  ExecutionRecord,
  ErrorPattern,
  SkillInput,
  SkillOutput,
  Conflict,
  HistoryEntry,
  Suggestion,
} from '../types/coreBinding';

// Re-export base types
export type {
  NodeId,
  GoalId,
  AgentId,
  SkillId,
  NodeType,
  NodeStatus,
  Mode,
  OverlayType,
  ViewType,
} from '../types';

// Re-export state types
export type { UIState } from '../types';
