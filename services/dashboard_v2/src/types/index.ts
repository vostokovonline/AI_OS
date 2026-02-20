/**
 * AI-OS v2 UI - Type Definitions
 *
 * Formal contract between UI and Core System
 */

// ============================================================================
// CORE DOMAIN TYPES
// ============================================================================

export type NodeId = string;
export type GoalId = string;
export type AgentId = string;
export type SkillId = string;

export type NodeType = 'goal' | 'agent' | 'skill' | 'memory' | 'test' | 'simulation';
export type NodeStatus = 'pending' | 'active' | 'done' | 'blocked' | 'failed';

export type Mode = 'explore' | 'exploit' | 'reflect';
export type OverlayType = 'none' | 'heatmap' | 'conflicts' | 'memory_traces' | 'simulation';
export type ViewType = 'graph' | 'gantt' | 'tree' | 'observability' | 'questions' | 'decomposition' | 'skills' | 'deployments' | 'occp-observability' | 'federation' | 'artifacts' | 'autonomy' | 'admin';

// ============================================================================
// GOAL NODE
// ============================================================================

export interface GoalNode {
  id: GoalId;
  type: 'goal';
  intent: string;
  goalType: 'achievable' | 'unachievable' | 'philosophical';

  // Progress
  status: NodeStatus;
  progress: number; // 0..1

  // Analysis
  feasibility: number; // 0..1
  conflictScore: number; // 0..1
  uncertainty: number; // 0..1

  // Hierarchy
  parentId?: GoalId;
  childIds: GoalId[];
  depthLevel?: number; // 0=Root/L1, 1=Strategic/L2, 2=Operational/L3, 3=Tactical/L4, 4+=Atomic/L5

  // Timing
  createdAt: string;
  startedAt?: string;
  completedAt?: string;

  // Constraints
  constraints: GoalConstraints;

  // Completion criteria
  completionCriteria?: CompletionCriteria;
}

export interface GoalConstraints {
  ethics?: string[];
  budget?: number;
  timeHorizon?: string; // ISO duration
  maxDepth?: number;
}

export interface CompletionCriteria {
  metrics?: {
    monthlyRevenueTarget?: number;
    sustainabilityPeriodMonths?: number;
  };
  condition?: string;
  artifactsRequired?: ArtifactSpec[];
}

export interface ArtifactSpec {
  type: 'DATASET' | 'FILE' | 'KNOWLEDGE' | 'CODE';
  description: string;
}

// ============================================================================
// ARTIFACTS
// ============================================================================

export type ArtifactType = 'FILE' | 'KNOWLEDGE' | 'DATASET' | 'REPORT' | 'LINK' | 'EXECUTION_LOG';
export type ContentKind = 'file' | 'db' | 'vector' | 'external';
export type VerificationStatus = 'pending' | 'passed' | 'failed' | 'partial';

export interface Artifact {
  id: string;
  type: ArtifactType;
  goal_id: string;
  skill_name: string;
  agent_role?: string;

  content_kind: ContentKind;
  content_location: string;

  domains: string[];
  tags: string[];
  language?: string;

  verification_status: VerificationStatus;
  verification_results: VerificationResult[];

  reusable: boolean;

  created_at: string;
  updated_at: string;
}

export interface VerificationResult {
  name: string;
  passed: boolean;
  details: string;
}

// ============================================================================
// AGENT NODE
// ============================================================================

export interface AgentNode {
  id: AgentId;
  type: 'agent';
  role: string;
  description?: string;

  // Performance
  confidence: number; // 0..1
  successRate: number; // 0..1
  costPerExecution: number; // USD

  // Capabilities
  skills: SkillId[];

  // State
  status: NodeStatus;
  currentTask?: string;

  // History
  lastExecutionAt?: string;
  executionCount: number;
}

// ============================================================================
// SKILL NODE
// ============================================================================

export interface SkillNode {
  id: SkillId;
  type: 'skill';
  name: string;
  description?: string;

  // I/O
  inputs: string[];
  outputs: string[];

  // Performance
  successRate: number; // 0..1
  avgLatencyMs: number;
  costPerUse: number; // USD

  // State
  status: NodeStatus;

  // History
  lastUsedAt?: string;
  usageCount: number;

  // Capabilities
  capabilities: SkillCapabilities;
}

export interface SkillCapabilities {
  deterministic: boolean;
  semantic: boolean;
  uiRobust: boolean;
  speed: 'fast' | 'medium' | 'slow';
}

// ============================================================================
// MEMORY NODE
// ============================================================================

export interface MemoryNode {
  id: NodeId;
  type: 'memory';
  memoryType: 'recent_failure' | 'resource_exhaustion' | 'false_success' | 'overfitting';

  // Signal
  target: string; // skill | goal | llm_profile
  intensity: number; // 0..1
  ttl: number; // cycles remaining

  // State
  createdAt: string;
  expiresAt?: string;
}

// ============================================================================
// GRAPH EDGES
// ============================================================================

export type EdgeType = 'causal' | 'dependency' | 'conflict' | 'reinforcement';

export interface GraphEdge {
  id: string;
  source: NodeId;
  target: NodeId;
  type: EdgeType;
  label?: string;
  strength?: number; // 0..1
}

// ============================================================================
// UI STATE (State Machine)
// ============================================================================

export interface UIState {
  // Mode
  mode: Mode;

  // View
  view: ViewType;

  // Focus
  focus: {
    nodeId: NodeId | null;
    nodeType: NodeType | null;
  };

  // Overlay
  overlay: OverlayType;

  // Timeline
  timelineCursor: string | null; // ISO timestamp

  // Constraints
  constraints: {
    ethics: string[];
    budget: number | null;
    timeHorizon: string | null;
  };

  // Override
  override: {
    enabled: boolean;
    decisionId: string | null;
  };

  // Graph view
  graph: {
    zoom: number;
    center: { x: number; y: number };
    collapsedLevels: number[]; // depth levels to collapse
  };
}

// ============================================================================
// UI EVENTS (UI → System)
// ============================================================================

// Re-export from v2 event model
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

// Legacy types for backward compatibility
export interface ScenarioChange {
  type: 'remove_node' | 'add_edge' | 'change_constraint';
  target: NodeId;
  details: any;
}

export interface DecisionOverride {
  action: 'block' | 'force' | 'modify';
  reason: string;
}

export interface ConstraintUpdate {
  type: 'ethics' | 'budget' | 'time_horizon';
  value: any;
}

// ============================================================================
// SYSTEM EVENTS (System → UI)
// ============================================================================

// Re-exported from eventModel.ts above

export interface GraphDiff {
  addedNodes: Node[];
  updatedNodes: Partial<Node>[];
  removedNodes: NodeId[];
  addedEdges: GraphEdge[];
  removedEdges: string[];
}

export type Node = GoalNode | AgentNode | SkillNode | MemoryNode;

// ============================================================================
// TIMELINE / TEMPORAL
// ============================================================================

export interface TimelineSnapshot {
  timestamp: string;
  knowledgeState: KnowledgeState;
  activeGoals: GoalId[];
  constraints: GoalConstraints;
  decisions: Decision[];
}

export interface KnowledgeState {
  facts: Fact[];
  assumptions: Assumption[];
  uncertainties: Uncertainty[];
}

export interface Fact {
  id: string;
  content: string;
  confidence: number;
  source: string;
}

export interface Assumption {
  id: string;
  content: string;
  probability: number;
}

export interface Uncertainty {
  id: string;
  topic: string;
  entropy: number; // 0..1
}

export interface Decision {
  id: string;
  timestamp: string;
  nodeId: NodeId;
  action: string;
  rationale: string;
  alternatives: Alternative[];
  rejected: boolean;
}

export interface Alternative {
  action: string;
  expectedOutcome: string;
  score: number;
}

// ============================================================================
// INSPECTOR CONTEXT
// ============================================================================

export interface InspectorContext {
  node: Node;
  systemState: SystemState;
  conflicts: Conflict[];
  history: NodeHistory[];
  suggestions: Suggestion[];
}

export interface SystemState {
  totalActiveGoals: number;
  resourceUsage: number; // 0..1
  errorRate: number; // 0..1
  recentFailures: number;
}

export interface Conflict {
  id: string;
  type: 'goal_goal' | 'goal_value' | 'goal_reality';
  nodeA: NodeId;
  nodeB: NodeId;
  description: string;
  severity: number; // 0..1
  resolution?: string;
}

export interface NodeHistory {
  timestamp: string;
  event: string;
  details: any;
}

export interface Suggestion {
  action: string;
  reason: string;
  priority: 'low' | 'medium' | 'high';
}

// ============================================================================
// API REQUEST/RESPONSE
// ============================================================================

export interface GraphQuery {
  rootGoalId?: GoalId;
  depth?: number;
  includeMemory?: boolean;
}

export interface GraphResponse {
  nodes: Node[];
  edges: GraphEdge[];
  timestamp: string;
}

export interface ExecuteActionRequest {
  nodeId: NodeId;
  action: string;
  context?: any;
}

export interface ExecuteActionResponse {
  success: boolean;
  result?: any;
  error?: string;
  executionTime: number;
}

// ============================================================================
// EMOTIONAL LAYER
// ============================================================================

export interface EmotionalState {
  arousal: number;      // 0..1, baseline 0.5
  valence: number;      // -1..1, baseline 0.0
  focus: number;        // 0..1, baseline 0.5
  confidence: number;   // 0..1, baseline 0.5
  timestamp?: string;
  source?: string;
}

export interface EmotionalInfluence {
  complexity_penalty: number;    // 0..1
  exploration_bias: number;      // -1..1
  explanation_depth: number;     // 0..1
  pace_modifier: number;         // -1..1
}

export interface EmotionalContext {
  complexity_limit: number;      // 0..1
  max_depth: number;             // 1..3
  exploration: 'conservative' | 'balanced' | 'aggressive';
  explanation: 'minimal' | 'normal' | 'detailed';
  pace: 'slow' | 'normal' | 'fast';
  confidence: 'low' | 'medium' | 'high';
}

export interface EmotionalSignals {
  user_text?: string;
  goal_stats?: {
    created?: number;
    completed?: number;
    aborted?: number;
    active?: number;
    success_ratio?: number;
  };
  system_metrics?: {
    avg_goal_complexity?: number;
    success_ratio?: number;
  };
}
