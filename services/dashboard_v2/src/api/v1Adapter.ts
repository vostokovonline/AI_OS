/**
 * v1 API Adapter
 *
 * Integrates v1 goal data into v2 dashboard format
 */

import { GoalNode, Node, GraphEdge } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// V1 Goal interface from backend
interface V1Goal {
  id: string;
  parent_id: string | null;
  title: string;
  description: string | null;
  status: string;
  progress: number;
  goal_type: string;
  depth_level: number;
  is_atomic: boolean;
  created_at: string | null;
  updated_at: string | null;
  completed_at: string | null;
}

interface V1GoalsResponse {
  status: string;
  goals: V1Goal[];
  total: number;
}

/**
 * Convert V1 goal to V2 GoalNode
 */
export function v1GoalToV2GoalNode(v1Goal: V1Goal): GoalNode {
  // Map goal types
  const goalTypeMap: Record<string, 'achievable' | 'unachievable' | 'philosophical'> = {
    'achievable': 'achievable',
    'continuous': 'philosophical',
    'directional': 'philosophical',
    'exploratory': 'achievable',
    'meta': 'philosophical',
  };

  const goalType = goalTypeMap[v1Goal.goal_type] || 'achievable';

  // Map status
  const statusMap: Record<string, 'pending' | 'active' | 'done' | 'blocked' | 'failed'> = {
    'active': 'active',
    'pending': 'pending',
    'done': 'done',
    'blocked': 'blocked',
    'improving': 'active',
    'failed': 'failed',
  };

  const status = statusMap[v1Goal.status] || 'pending';

  return {
    id: v1Goal.id,
    type: 'goal',
    intent: v1Goal.title,
    goalType: goalType,

    // Progress
    status: status,
    progress: v1Goal.progress || 0,

    // Analysis (simulated for now)
    feasibility: 0.7, // Will be calculated based on goal properties
    conflictScore: 0.2, // Will be calculated based on conflicts
    uncertainty: v1Goal.goal_type === 'exploratory' ? 0.6 : 0.3,

    // Hierarchy
    parentId: v1Goal.parent_id || undefined,
    childIds: [], // Will be populated after loading all goals
    depthLevel: v1Goal.depth_level,

    // Timing - use nullish coalescing for better NULL handling
    createdAt: v1Goal.created_at ?? new Date().toISOString(),
    startedAt: v1Goal.updated_at ?? undefined,
    completedAt: v1Goal.completed_at ?? (status === 'done' ? (v1Goal.updated_at ?? new Date().toISOString()) : undefined),

    // Constraints (default empty)
    constraints: {},

    // Completion criteria
    completionCriteria: undefined,
  };
}

/**
 * Convert V1 goals list to V2 graph structure
 */
export function v1GoalsToV2Graph(v1Goals: V1Goal[]): {
  nodes: Node[];
  edges: GraphEdge[];
} {
  // Convert all goals to V2 format
  const v2Nodes = v1Goals.map(v1GoalToV2GoalNode);

  // Populate childIds for each node
  const childMap = new Map<string, string[]>();
  v1Goals.forEach(goal => {
    if (goal.parent_id) {
      if (!childMap.has(goal.parent_id)) {
        childMap.set(goal.parent_id, []);
      }
      childMap.get(goal.parent_id)!.push(goal.id);
    }
  });

  v2Nodes.forEach(node => {
    if (node.type === 'goal') {
      node.childIds = childMap.get(node.id) || [];
    }
  });

  // Create edges from parent-child relationships
  const edges: GraphEdge[] = [];
  let edgeCounter = 0;

  // Add dependency edges from parent_id
  v1Goals.forEach(goal => {
    if (goal.parent_id) {
      edges.push({
        id: `edge-dep-${edgeCounter++}`,
        source: goal.parent_id,
        target: goal.id,
        type: 'dependency' as const,
        label: 'subgoal',
        strength: 1.0,
      });
    }
  });

  // TODO: Load additional relations (causal, conflict, reinforcement) from API
  // These will be fetched from /relations/{goal_id} endpoint

  return {
    nodes: v2Nodes,
    edges,
  };
}

/**
 * Fetch goals from V1 backend
 */
export async function fetchV1Goals(apiUrl: string = API_BASE_URL): Promise<V1GoalsResponse> {
  const response = await fetch(`${apiUrl}/goals/list`);

  if (!response.ok) {
    throw new Error(`Failed to fetch V1 goals: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Load V1 goals and convert to V2 format
 */
export async function loadV1GoalsAsV2Graph(): Promise<{
  nodes: Node[];
  edges: GraphEdge[];
}> {
  const v1Response = await fetchV1Goals();

  if (v1Response.status !== 'ok') {
    throw new Error('V1 API returned error status');
  }

  const graph = v1GoalsToV2Graph(v1Response.goals);

  // Skip loading relations due to CORS issues with 500 errors
  // Relations are not critical for observability interface
  const relations: any[] = [];

  // Convert relations to edges
  let edgeCounter = 1000; // Start from 1000 to avoid conflicts with dependency edges
  relations.forEach((rel: any) => {
    // Skip if this is a dependency relation (already handled by parent_id)
    if (rel.relation_type === 'dependency') {
      return;
    }

    graph.edges.push({
      id: `edge-rel-${edgeCounter++}`,
      source: rel.from_goal_id,
      target: rel.to_goal_id,
      type: rel.relation_type as any,
      label: rel.relation_type,
      strength: rel.strength || 1.0,
    });
  });

  return graph;
}
