/**
 * v2 UI - Graph Store
 *
 * Manages the goal/agent/skill graph state
 */

import { create } from 'zustand';
import {
  Node,
  GraphEdge,
  GraphDiff,
  NodeId,
} from '../types';

export interface FilterState {
  showPending: boolean;
  showActive: boolean;
  showDone: boolean;
  showBlocked: boolean;
  searchQuery: string;
  minDepth?: number;
  maxDepth?: number;
  showOnlyRoots: boolean;  // Show only goals without parents
  collapseChildren: boolean;  // Hide children of selected goals
  showOnlyAtomic: boolean;  // Show only atomic goals (is_atomic = true)
}

interface GraphStore {
  nodes: Map<NodeId, Node>;
  edges: GraphEdge[];
  loading: boolean;
  error: string | null;
  filters: FilterState;
  collapsedNodes: Set<NodeId>; // Track which nodes are collapsed

  // Actions
  setNodes: (nodes: Node[]) => void;
  updateNode: (nodeId: NodeId, updates: Partial<Node>) => void;
  addNodes: (nodes: Node[]) => void;
  removeNodes: (nodeIds: NodeId[]) => void;
  setEdges: (edges: GraphEdge[]) => void;
  addEdges: (edges: GraphEdge[]) => void;
  removeEdges: (edgeIds: string[]) => void;
  applyDiff: (diff: GraphDiff) => void;
  getNode: (nodeId: NodeId) => Node | undefined;
  clear: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setFilters: (filters: Partial<FilterState>) => void;
  getFilteredNodes: () => Node[];
  toggleCollapse: (nodeId: NodeId) => void;
  isNodeCollapsed: (nodeId: NodeId) => boolean;
}

const initialFilters: FilterState = {
  showPending: true,
  showActive: true,
  showDone: true,
  showBlocked: true,
  searchQuery: '',
  minDepth: undefined,
  maxDepth: undefined,
  showOnlyRoots: false,
  collapseChildren: false,
  showOnlyAtomic: false,  // Default: show all goals
};

export const useGraphStore = create<GraphStore>((set, get) => ({
  nodes: new Map(),
  edges: [],
  loading: false,
  error: null,
  filters: initialFilters,
  collapsedNodes: new Set<NodeId>(),

  setNodes: (nodes: Node[]) => {
    const nodeMap = new Map<NodeId, Node>();
    nodes.forEach((node) => {
      nodeMap.set(node.id, node);
    });
    set({ nodes: nodeMap });
  },

  updateNode: (nodeId: NodeId, updates: Partial<Node>) => {
    set((state) => {
      const node = state.nodes.get(nodeId);
      if (!node) return state;

      const updatedNodes = new Map(state.nodes);
      updatedNodes.set(nodeId, { ...node, ...updates } as Node);
      return { nodes: updatedNodes };
    });
  },

  addNodes: (nodes: Node[]) => {
    set((state) => {
      const updatedNodes = new Map(state.nodes);
      nodes.forEach((node) => {
        updatedNodes.set(node.id, node);
      });
      return { nodes: updatedNodes };
    });
  },

  removeNodes: (nodeIds: NodeId[]) => {
    set((state) => {
      const updatedNodes = new Map(state.nodes);
      nodeIds.forEach((id) => {
        updatedNodes.delete(id);
      });
      return { nodes: updatedNodes };
    });
  },

  setEdges: (edges: GraphEdge[]) => {
    set({ edges });
  },

  addEdges: (edges: GraphEdge[]) => {
    set((state) => ({
      edges: [...state.edges, ...edges],
    }));
  },

  removeEdges: (edgeIds: string[]) => {
    set((state) => ({
      edges: state.edges.filter((e) => !edgeIds.includes(e.id)),
    }));
  },

  applyDiff: (diff: GraphDiff) => {
    set((state) => {
      const updatedNodes = new Map(state.nodes);

      // Add new nodes
      diff.addedNodes.forEach((node) => {
        updatedNodes.set(node.id, node);
      });

      // Update existing nodes
      diff.updatedNodes.forEach((updates) => {
        if ('id' in updates && updates.id) {
          const existing = updatedNodes.get(updates.id);
          if (existing) {
            updatedNodes.set(updates.id, { ...existing, ...updates } as Node);
          }
        }
      });

      // Remove nodes
      diff.removedNodes.forEach((id) => {
        updatedNodes.delete(id);
      });

      // Update edges
      const updatedEdges = [...state.edges];
      diff.addedEdges.forEach((edge) => {
        updatedEdges.push(edge);
      });
      const filteredEdges = updatedEdges.filter(
        (e) => !diff.removedEdges.includes(e.id)
      );

      return {
        nodes: updatedNodes,
        edges: filteredEdges,
      };
    });
  },

  getNode: (nodeId: NodeId) => {
    return get().nodes.get(nodeId);
  },

  clear: () => {
    set({
      nodes: new Map(),
      edges: [],
      error: null,
    });
  },

  setLoading: (loading: boolean) => {
    set({ loading });
  },

  setError: (error: string | null) => {
    set({ error });
  },

  setFilters: (newFilters) => {
    set((state) => ({
      filters: { ...state.filters, ...newFilters },
    }));
  },

  getFilteredNodes: () => {
    const state = get();
    const { nodes, filters, collapsedNodes } = state;

    // Helper function to check if a node should be hidden due to parent collapse
    const isHiddenByCollapsedParent = (nodeId: string, visited = new Set<string>()): boolean => {
      if (visited.has(nodeId)) return false;
      visited.add(nodeId);

      const node = nodes.get(nodeId);
      if (!node || node.type !== 'goal' || !(node as any).parentId) return false;

      const parentId = (node as any).parentId;

      // Check if parent is collapsed
      if (collapsedNodes.has(parentId)) return true;

      // Recursively check up the tree
      return isHiddenByCollapsedParent(parentId, visited);
    };

    return Array.from(nodes.values()).filter((node: any) => {
      if (node.type === 'goal') {
        // Hide if any ancestor is collapsed
        if (isHiddenByCollapsedParent(node.id)) return false;

        // Show only roots filter - hide all nodes that have a parent
        if (filters.showOnlyRoots && node.parentId) return false;

        // Show only atomic goals filter
        if (filters.showOnlyAtomic && !(node as any).isAtomic) return false;

        // Status filter
        if (!filters.showPending && node.status === 'pending') return false;
        if (!filters.showActive && node.status === 'active') return false;
        if (!filters.showDone && node.status === 'done') return false;
        if (!filters.showBlocked && node.status === 'blocked') return false;

        // Search filter
        if (filters.searchQuery) {
          const query = filters.searchQuery.toLowerCase();
          return (
            node.intent?.toLowerCase().includes(query) ||
            node.goalType?.toLowerCase().includes(query)
          );
        }

        // Depth filter
        if (filters.minDepth !== undefined && node.depthLevel !== undefined) {
          if (node.depthLevel < filters.minDepth) return false;
        }
        if (filters.maxDepth !== undefined && node.depthLevel !== undefined) {
          if (node.depthLevel > filters.maxDepth) return false;
        }
      }
      return true;
    });
  },

  toggleCollapse: (nodeId: NodeId) => {
    set((state) => {
      const newCollapsed = new Set(state.collapsedNodes);
      if (newCollapsed.has(nodeId)) {
        newCollapsed.delete(nodeId);
      } else {
        newCollapsed.add(nodeId);
      }
      return { collapsedNodes: newCollapsed };
    });
  },

  isNodeCollapsed: (nodeId: NodeId) => {
    return get().collapsedNodes.has(nodeId);
  },
}));
