/**
 * v2 UI - Core Schema Binding
 *
 * Formal contract between UI nodes and Core system entities
 * Following the specification from v2 UI contract
 */

import { NodeId, GoalId, AgentId, SkillId } from '../types';

// ============================================================================
// GOAL NODE (Core Schema Projection)
// ============================================================================

/**
 * UI projection of a Goal from Core system
 * UI does NOT store this - it reflects the snapshot
 */
export interface GoalNode {
  id: GoalId;
  type: 'goal';

  // Core properties
  intent: string;
  goalType: 'achievable' | 'unachievable' | 'philosophical';

  // Progress & Status
  status: 'pending' | 'active' | 'done' | 'blocked' | 'failed';
  progress: number; // 0..1

  // Analysis
  feasibility: number; // 0..1
  conflictScore: number; // 0..1
  uncertainty: number; // 0..1

  // Hierarchy
  parentId?: GoalId;
  childIds: GoalId[];
  depthLevel?: number;

  // Timing
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  estimatedCompletion?: string;

  // Constraints
  constraints?: {
    ethics?: string[];
    budget?: number;
    timeHorizon?: string;
  };

  // Execution tracking
  executionPlan?: ExecutionStep[];

  // Relationships (computed)
  dependencies?: NodeId[];
  dependents?: NodeId[];
  conflicts?: NodeId[];
  reinforcements?: NodeId[];
}

export interface ExecutionStep {
  stepId: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  result?: string;
  startedAt?: string;
  completedAt?: string;
}

// ============================================================================
// AGENT NODE (Core Schema Projection)
// ============================================================================

/**
 * UI projection of an Agent from Core system
 */
export interface AgentNode {
  id: AgentId;
  type: 'agent';

  // Identity
  role: string;
  description?: string;

  // Performance
  confidence: number; // 0..1
  successRate: number; // 0..1
  totalExecutions: number;

  // Cost
  costPerExecution: number;
  totalCost: number;

  // Capabilities
  skills: SkillId[];
  capabilities?: string[];

  // Current State
  status: 'idle' | 'executing' | 'error';
  currentTask?: NodeId;

  // Performance History
  recentExecutions?: ExecutionRecord[];
  errorPatterns?: ErrorPattern[];
}

export interface ExecutionRecord {
  executionId: string;
  taskId: NodeId;
  startedAt: string;
  completedAt?: string;
  success: boolean;
  duration: number;
  output?: any;
}

export interface ErrorPattern {
  errorType: string;
  frequency: number;
  lastOccurred: string;
}

// ============================================================================
// SKILL NODE (Core Schema Projection)
// ============================================================================

/**
 * UI projection of a Skill from Core system
 */
export interface SkillNode {
  id: SkillId;
  type: 'skill';

  // Identity
  name: string;
  description?: string;

  // Interface (I/O)
  inputs: SkillInput[];
  outputs: SkillOutput[];

  // Performance
  successRate: number; // 0..1
  avgDuration: number; // milliseconds
  totalUses: number;

  // Metadata
  lastUsed?: string;
  tags?: string[];

  // Dependencies
  requires?: SkillId[];
  requiredBy?: AgentId[];
}

export interface SkillInput {
  name: string;
  type: string;
  required: boolean;
  description?: string;
}

export interface SkillOutput {
  name: string;
  type: string;
  description?: string;
}

// ============================================================================
// MEMORY NODE (Core Schema Projection)
// ============================================================================

/**
 * UI projection of a Memory trace from Core system
 */
export interface MemoryNode {
  id: string;
  type: 'memory';

  // Content
  content: string;
  memoryType: 'episodic' | 'semantic' | 'procedural' | 'working';

  // Temporal
  timestamp: string;
  expiresAt?: string;

  // Importance
  importance: number; // 0..1
  accessCount: number;

  // Associations
  relatedGoals?: GoalId[];
  relatedAgents?: AgentId[];
  relatedMemories?: string[];
}

// ============================================================================
// TEST / SIMULATION NODE
// ============================================================================

/**
 * UI projection of a Test or Simulation
 */
export interface TestNode {
  id: string;
  type: 'test' | 'simulation';

  // Identity
  name: string;
  description?: string;

  // Test definition
  testType: 'unit' | 'integration' | 'simulation' | 'exploration';

  // Target
  targetId: NodeId;

  // Results
  status: 'pending' | 'running' | 'passed' | 'failed';
  score?: number; // 0..1

  // Timing
  startedAt?: string;
  completedAt?: string;
  duration?: number;

  // Details
  result?: any;
  error?: string;
}

// ============================================================================
// GRAPH EDGE (Relationship)
// ============================================================================

/**
 * Relationship between nodes in the graph
 */
export interface GraphEdge {
  id: string;
  source: NodeId;
  target: NodeId;
  type: 'causal' | 'dependency' | 'conflict' | 'reinforcement';
  label: string;
  strength: number; // 0..1
  metadata?: Record<string, any>;
}

// ============================================================================
// NODE UNION TYPE
// ============================================================================

export type Node = GoalNode | AgentNode | SkillNode | MemoryNode | TestNode;

// ============================================================================
// INSPECTOR BINDING
// ============================================================================

/**
 * System Context for Inspector Panel
 * Provides additional context beyond the node itself
 */
export interface InspectorContext {
  node: Node;
  systemState: {
    totalActiveGoals: number;
    resourceUsage: number; // 0..1
    errorRate: number; // 0..1
    recentFailures: number;
  };
  conflicts: Conflict[];
  history: HistoryEntry[];
  suggestions: Suggestion[];
}

export interface Conflict {
  id: string;
  severity: number; // 0..1
  description: string;
  relatedNodes: NodeId[];
}

export interface HistoryEntry {
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
// SERIALIZATION / SYNC
// ============================================================================

/**
 * Convert Core Goal to UI GoalNode
 */
export function coreGoalToUINode(coreGoal: any): GoalNode {
  return {
    id: coreGoal.id,
    type: 'goal',
    intent: coreGoal.title || coreGoal.intent || 'Untitled Goal',
    goalType: coreGoal.goal_type || 'achievable',
    status: coreGoal.status || 'pending',
    progress: coreGoal.progress || 0,
    feasibility: coreGoal.feasibility || 0.5,
    conflictScore: coreGoal.conflict_score || 0,
    uncertainty: coreGoal.uncertainty || 0,
    parentId: coreGoal.parent_id || undefined,
    childIds: coreGoal.child_ids || [],
    depthLevel: coreGoal.depth_level,
    createdAt: coreGoal.created_at || new Date().toISOString(),
    startedAt: coreGoal.started_at,
    completedAt: coreGoal.completed_at,
    constraints: coreGoal.constraints,
  };
}

/**
 * Convert UI GoalNode to Core Goal format
 * Used for updates back to Core system
 */
export function uiNodeToCoreGoal(uiNode: GoalNode): any {
  return {
    id: uiNode.id,
    title: uiNode.intent,
    goal_type: uiNode.goalType,
    status: uiNode.status,
    progress: uiNode.progress,
    feasibility: uiNode.feasibility,
    conflict_score: uiNode.conflictScore,
    uncertainty: uiNode.uncertainty,
    parent_id: uiNode.parentId,
    child_ids: uiNode.childIds,
    depth_level: uiNode.depthLevel,
    created_at: uiNode.createdAt,
    started_at: uiNode.startedAt,
    completed_at: uiNode.completedAt,
    constraints: uiNode.constraints,
  };
}

/**
 * Graph Diff for incremental updates
 */
export interface GraphDiff {
  addedNodes: Node[];
  updatedNodes: Partial<Node>[];
  removedNodes: NodeId[];
  addedEdges: GraphEdge[];
  removedEdges: string[];
}

/**
 * Create graph diff from two node sets
 */
export function createGraphDiff(
  oldNodes: Map<NodeId, Node>,
  newNodes: Map<NodeId, Node>,
  oldEdges: GraphEdge[],
  newEdges: GraphEdge[]
): GraphDiff {
  const addedNodes: Node[] = [];
  const updatedNodes: Partial<Node>[] = [];
  const removedNodes: NodeId[] = [];

  // Find added and updated nodes
  newNodes.forEach((node, id) => {
    if (!oldNodes.has(id)) {
      addedNodes.push(node);
    } else {
      const oldNode = oldNodes.get(id)!;
      // Simple comparison - could be deep equality check
      if (JSON.stringify(oldNode) !== JSON.stringify(node)) {
        updatedNodes.push(node);
      }
    }
  });

  // Find removed nodes
  oldNodes.forEach((_, id) => {
    if (!newNodes.has(id)) {
      removedNodes.push(id);
    }
  });

  // Edge diffs
  const oldEdgeIds = new Set(oldEdges.map(e => e.id));
  const newEdgeIds = new Set(newEdges.map(e => e.id));

  const addedEdges = newEdges.filter(e => !oldEdgeIds.has(e.id));
  const removedEdges = oldEdges.filter(e => !newEdgeIds.has(e.id)).map(e => e.id);

  return {
    addedNodes,
    updatedNodes,
    removedNodes,
    addedEdges,
    removedEdges,
  };
}
