/**
 * v2 UI - Graph Canvas
 *
 * Main graph visualization using ReactFlow
 * Displays goals, agents, skills, and their relationships
 *
 * FEATURES:
 * - Position persistence (saved to localStorage)
 * - Auto-grouping by parent_id (related goals stay close)
 */

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  NodeTypes,
  useNodesState,
  useEdgesState,
  Connection,
  addEdge,
  ReactFlowProvider,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useUIStore } from '../../store/uiStore';
import { useGraphStore } from '../../store/graphStore';
import { loadV1GoalsAsV2Graph } from '../../api/v1Adapter';
import { simpleMockData } from '../../api/mockData';
import { GoalNode, AgentNode, SkillNode, MemoryNode } from '../../types';

// Custom node components
import GoalNodeComponent from './nodes/GoalNode';
import AgentNodeComponent from './nodes/AgentNode';
import SkillNodeComponent from './nodes/SkillNode';
import MemoryNodeComponent from './nodes/MemoryNode';
import { FilterToolbar } from './FilterToolbar';

const nodeTypes: NodeTypes = {
  goal: GoalNodeComponent,
  agent: AgentNodeComponent,
  skill: SkillNodeComponent,
  memory: MemoryNodeComponent,
};

// ============================================================================
// POSITION PERSISTENCE & LAYOUT HELPERS
// ============================================================================

const STORAGE_KEY = 'ai-os-graph-layout';

interface SavedLayout {
  nodes: { id: string; position: { x: number; y: number } }[];
}

/**
 * Load saved positions from localStorage
 */
function loadSavedPositions(): Map<string, { x: number; y: number }> {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return new Map();

    const layout: SavedLayout = JSON.parse(saved);
    return new Map(layout.nodes.map(n => [n.id, n.position]));
  } catch (error) {
    console.error('[GraphCanvas] Failed to load saved positions:', error);
    return new Map();
  }
}

/**
 * Save current positions to localStorage
 */
function savePositions(nodes: Node[]) {
  try {
    const layout: SavedLayout = {
      nodes: nodes.map(n => ({ id: n.id, position: n.position })),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
  } catch (error) {
    console.error('[GraphCanvas] Failed to save positions:', error);
  }
}

/**
 * Layout 1: Level-based (horizontal bands)
 * Groups goals by depth_level, places in horizontal rows
 */
function calculateLevelsLayout(
  nodes: any[],
  savedPositions: Map<string, { x: number; y: number }>
): Node[] {
  const levelGroups = new Map<number, any[]>();
  for (let i = 0; i <= 5; i++) {
    levelGroups.set(i, []);
  }

  // Debug: log depth levels
  const depthLevels = nodes.map(n => (n as any).depthLevel ?? 0);
  console.log('[LevelsLayout] Total nodes:', nodes.length, 'Depth levels:', depthLevels);

  nodes.forEach(node => {
    const level = (node as any).depthLevel ?? 0;
    const group = levelGroups.get(level) || levelGroups.get(0)!;
    group.push(node);
  });

  // Debug: log groups
  console.log('[LevelsLayout] Level groups:', Array.from(levelGroups.entries()).map(([lvl, nodes]) => [lvl, nodes.length]));

  const placed = new Set<string>();
  const result: Node[] = [];

  const LEVEL_HEIGHT = 300; // Reduced from 400
  const NODE_WIDTH = 320;   // Reduced from 350
  const NODES_PER_ROW = 4;  // Reduced from 5

  for (let level = 0; level <= 5; level++) {
    const nodesInLevel = levelGroups.get(level) || [];
    if (nodesInLevel.length === 0) continue;

    console.log(`[LevelsLayout] Level ${level}: ${nodesInLevel.length} nodes`);

    nodesInLevel.forEach((node, index) => {
      if (placed.has(node.id)) return;

      const savedPos = savedPositions.get(node.id);
      const row = Math.floor(index / NODES_PER_ROW);
      const col = index % NODES_PER_ROW;

      const defaultPos = {
        x: col * NODE_WIDTH,
        y: level * LEVEL_HEIGHT + row * 220,
      };

      const position = savedPos || defaultPos;

      result.push({
        id: node.id,
        type: node.type,
        position,
        data: node,
      });

      placed.add(node.id);
    });
  }

  console.log('[LevelsLayout] Total placed:', result.length);
  return result;
}

/**
 * Layout 2: Hierarchy (tree structure)
 * Children placed below parents with indentation
 */
function calculateHierarchyLayout(
  nodes: any[],
  savedPositions: Map<string, { x: number; y: number }>
): Node[] {
  // Build parent-child map
  const childMap = new Map<string | null, any[]>();
  childMap.set(null, []);

  // Debug: log parent-child relationships
  console.log('[HierarchyLayout] Building parent-child map...');

  nodes.forEach(node => {
    const parentId = (node as any).parentId || null;
    if (!childMap.has(parentId)) {
      childMap.set(parentId, []);
    }
    childMap.get(parentId)!.push(node);
  });

  // Debug: log roots
  const roots = childMap.get(null) || [];
  console.log('[HierarchyLayout] Root nodes:', roots.length, 'Total nodes:', nodes.length);
  console.log('[HierarchyLayout] Child map:', Array.from(childMap.entries()).map(([pid, children]) => [pid?.slice(0, 8) + '...' || 'null', children.length]));

  const placed = new Set<string>();
  const result: Node[] = [];

  const NODE_WIDTH = 280;
  const NODE_HEIGHT = 180;

  // Recursive layout function
  const layoutNode = (node: any, x: number, y: number): void => {
    if (placed.has(node.id)) return;

    const savedPos = savedPositions.get(node.id);
    const position = savedPos || { x, y };

    result.push({
      id: node.id,
      type: node.type,
      position,
      data: node,
    });

    placed.add(node.id);

    // Layout children
    const children = childMap.get(node.id) || [];
    if (children.length > 0) {
      const totalWidth = children.length * NODE_WIDTH;
      const startX = x - totalWidth / 2;

      children.forEach((child, index) => {
        layoutNode(child, startX + index * NODE_WIDTH, y + NODE_HEIGHT);
      });
    }
  };

  // Start with root nodes - spread them horizontally
  roots.forEach((root, index) => {
    layoutNode(root, index * (NODE_WIDTH * 2), 0);
  });

  console.log('[HierarchyLayout] Total placed:', result.length);
  return result;
}

/**
 * Layout 3: Force-directed (circular clusters)
 * Nodes distributed in a spiral pattern
 */
function calculateForceLayout(
  nodes: any[],
  _savedPositions: Map<string, { x: number; y: number }>
): Node[] {
  const result: Node[] = [];
  const centerX = 1000;
  const centerY = 1000;
  const radiusPerNode = 40;

  nodes.forEach((node, index) => {
    const angle = (index / nodes.length) * Math.PI * 2 * 3; // 3 spirals
    const radius = Math.sqrt(index) * radiusPerNode + 100;

    const position = {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };

    result.push({
      id: node.id,
      type: node.type,
      position,
      data: node,
    });
  });

  return result;
}

/**
 * Layout 4: Circular
 * Nodes arranged in a circle
 */
function calculateCircularLayout(
  nodes: any[],
  _savedPositions: Map<string, { x: number; y: number }>
): Node[] {
  const result: Node[] = [];
  const centerX = 1000;
  const centerY = 1000;
  const radius = Math.max(500, nodes.length * 30);

  nodes.forEach((node, index) => {
    const angle = (index / nodes.length) * Math.PI * 2;

    const position = {
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
    };

    result.push({
      id: node.id,
      type: node.type,
      position,
      data: node,
    });
  });

  return result;
}

/**
 * Calculate layout with auto-grouping by depth level
 * - Groups goals by depth_level (L1=0, L2=1, L3=2, L4=3, L5=4+)
 * - Places nodes in horizontal bands based on level
 * - Uses saved positions when available
 */
function calculateLayout(
  nodes: any[],
  savedPositions: Map<string, { x: number; y: number }>,
  layoutMode: LayoutMode
): Node[] {
  switch (layoutMode) {
    case 'levels':
      return calculateLevelsLayout(nodes, savedPositions);
    case 'hierarchy':
      return calculateHierarchyLayout(nodes, savedPositions);
    case 'force':
      return calculateForceLayout(nodes, savedPositions);
    case 'circular':
      return calculateCircularLayout(nodes, savedPositions);
    default:
      return calculateLevelsLayout(nodes, savedPositions);
  }
}

// Level band colors for visual separation
const LEVEL_COLORS = [
  'rgba(59, 130, 246, 0.03)',  // L1 - Blue (Mission)
  'rgba(16, 185, 129, 0.03)',  // L2 - Green (Strategic)
  'rgba(245, 158, 11, 0.03)',  // L3 - Orange (Operational)
  'rgba(139, 92, 246, 0.03)',  // L4 - Purple (Tactical)
  'rgba(239, 68, 68, 0.03)',   // L5 - Red (Atomic)
];

const LEVEL_LABELS = [
  'L1: Mission (Root)',
  'L2: Strategic',
  'L3: Operational',
  'L4: Tactical',
  'L5: Atomic',
];

type LayoutMode = 'levels' | 'hierarchy' | 'force' | 'circular';

function GraphCanvasInner() {
  const { mode, overlay, dispatchEvent } = useUIStore();
  const { nodes: graphNodes, edges, loading, setLoading, setError, getFilteredNodes, filters } = useGraphStore();

  // Load saved positions once on mount
  const [savedPositions] = useState(() => loadSavedPositions());
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('levels');

  // Convert graph nodes to ReactFlow nodes with smart layout
  const flowNodes: Node[] = useMemo(() => {
    // Get filtered nodes
    let filteredNodes = getFilteredNodes();

    // Calculate depth level dynamically based on parent_id hierarchy
    const calculateDepthLevel = (node: any, visited = new Set<string>()): number => {
      if (visited.has(node.id)) return 0; // Prevent infinite loops
      visited.add(node.id);

      if (!node.parentId) return 0; // Root = L1

      const parent = graphNodes.get(node.parentId);
      if (!parent) return 0;

      return calculateDepthLevel(parent, visited) + 1;
    };

    // Enrich nodes with calculated depthLevel
    const enrichedNodes = filteredNodes.map(node => ({
      ...node,
      depthLevel: calculateDepthLevel(node),
    }));

    // Debug: log depth levels
    const depthCounts = new Map<number, number>();
    enrichedNodes.forEach(n => {
      const level = (n as any).depthLevel ?? 0;
      depthCounts.set(level, (depthCounts.get(level) ?? 0) + 1);
    });
    console.log('[GraphCanvas] Depth distribution:', Object.fromEntries(depthCounts));

    // For non-levels modes, don't use saved positions to force fresh layout
    const positionsToUse = layoutMode === 'levels' ? savedPositions : new Map();

    // Calculate layout with selected layout mode
    const layoutNodes = calculateLayout(enrichedNodes, positionsToUse, layoutMode);

    console.log(`[GraphCanvas] Layout mode: ${layoutMode}, nodes: ${layoutNodes.length}`);

    return layoutNodes.map((node) => {
      const baseNode: Node = {
        ...node,
      };

      // Apply depth level styling
      if (node.data.type === 'goal') {
        const depthLevel = (node.data as any).depthLevel ?? 0;
        baseNode.style = getDepthLevelStyle(depthLevel);
      }

      // Style based on overlay
      if (overlay === 'heatmap') {
        baseNode.style = { ...baseNode.style, ...getHeatmapStyle(node.data) };
      } else if (overlay === 'conflicts') {
        baseNode.style = { ...baseNode.style, ...getConflictStyle(node.data) };
      }

      return baseNode;
    });
  }, [graphNodes, overlay, filters, getFilteredNodes, savedPositions, layoutMode]);

  const flowEdges: Edge[] = useMemo(() => {
    return edges.map((edge) => {
      // Check if source or target node is active/executing
      const sourceNode = graphNodes.get(edge.source);
      const targetNode = graphNodes.get(edge.target);
      const isActiveChain =
        (sourceNode?.type === 'goal' && sourceNode.status === 'active') ||
        (targetNode?.type === 'goal' && targetNode.status === 'active');

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: getEdgeType(edge.type),
        label: edge.label,
        style: getEdgeStyle(edge.type),
        animated: isActiveChain || edge.type === 'causal' || edge.type === 'reinforcement',
        className: isActiveChain ? 'edge-flowing' : '',
      };
    });
  }, [edges, overlay, graphNodes]);

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes);
  const [edgesState, setEdges, onEdgesChange] = useEdgesState(flowEdges);

  // Update nodes when graph changes
  useEffect(() => {
    setNodes(flowNodes);
  }, [flowNodes, setNodes]);

  // Update edges when they change
  useEffect(() => {
    setEdges(flowEdges);
  }, [flowEdges, setEdges]);

  // Save positions when nodes change (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      savePositions(nodes);
    }, 500); // Save after 500ms of inactivity

    return () => clearTimeout(timer);
  }, [nodes]);

  // Load initial graph from V1 backend
  useEffect(() => {
    const loadGraph = async () => {
      setLoading(true);
      try {
        // Load V1 goals and convert to V2 format
        console.log('[GraphCanvas] Loading from V1 API...');
        const v2Graph = await loadV1GoalsAsV2Graph();

        // Update graph store with converted data
        const nodeMap = new Map();
        v2Graph.nodes.forEach((node) => {
          nodeMap.set(node.id, node);
        });

        // Store in graph store (we'll access directly since setNodes isn't exposed)
        useGraphStore.setState({
          nodes: nodeMap,
          edges: v2Graph.edges,
          loading: false,
          error: null,
        });

        console.log(`[GraphCanvas] Loaded ${v2Graph.nodes.length} nodes from V1 API`);
      } catch (error) {
        console.error('[GraphCanvas] Failed to load graph from V1, falling back to mock data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load graph');

        // Fallback to mock data
        const v2Graph = simpleMockData;
        const nodeMap = new Map();
        v2Graph.nodes.forEach((node) => {
          nodeMap.set(node.id, node);
        });

        useGraphStore.setState({
          nodes: nodeMap,
          edges: v2Graph.edges,
          loading: false,
          error: null,
        });

        console.log(`[GraphCanvas] Loaded ${v2Graph.nodes.length} mock nodes (FALLBACK)`);
      } finally {
        setLoading(false);
      }
    };

    loadGraph();
  }, [setLoading, setError]);

  // Handle node selection
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      if (!node.type) return; // Skip if node type is undefined

      dispatchEvent({
        type: 'SELECT_NODE',
        nodeId: node.id,
        nodeType: node.type as any, // Type assertion needed
      });
    },
    [dispatchEvent]
  );

  // Handle new connections (simulation mode)
  const onConnect = useCallback(
    (connection: Connection) => {
      if (mode === 'explore') {
        // Allow adding new connections in explore mode
        setEdges((eds) => addEdge(connection, eds));
      }
    },
    [mode, setEdges]
  );

  // Reset layout to default
  const handleResetLayout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    window.location.reload();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900">
        <div className="text-white text-lg">Loading graph...</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gray-900 relative">
      {/* Level Bands Background - only show in levels mode */}
      {layoutMode === 'levels' && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          {LEVEL_COLORS.map((color, index) => (
            <div
              key={index}
              className="absolute left-0 right-0 border-b border-gray-800/50"
              style={{
                top: `${index * 20}%`,
                height: '20%',
                backgroundColor: color,
              }}
            >
              {/* Level Label */}
              <div className="absolute top-2 left-4 text-xs font-bold text-gray-600 uppercase tracking-wider">
                {LEVEL_LABELS[index]}
              </div>
            </div>
          ))}
        </div>
      )}

      <ReactFlow
        nodes={nodes}
        edges={edgesState}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        className="bg-gray-900"
      >
        <Background color="#374151" gap={16} />
        <Controls className="bg-gray-800 border-gray-700" />
        <MiniMap
          nodeColor={(node) => getNodeColor(node.data)}
          maskColor="rgba(0, 0, 0, 0.6)"
          position="bottom-right"
          style={{
            backgroundColor: '#1f2937',
            border: '1px solid #4b5563',
            borderRadius: '8px',
            width: 220,
            height: 160,
            zIndex: 1000,
          }}
          zoomable
          pannable
          ariaLabel="Mini map"
        />

        {/* Mode indicator */}
        <Panel position="top-left" className="bg-gray-800 text-white px-3 py-2 rounded" style={{ top: '10px', left: '10px' }}>
          <span className="text-sm font-medium">Mode: {mode.toUpperCase()}</span>
        </Panel>

        {/* Filters */}
        <Panel position="top-right" className="!w-64">
          <FilterToolbar />
        </Panel>

        {/* Overlay indicator */}
        {overlay !== 'none' && (
          <Panel position="top-right" className="bg-gray-800 text-white px-3 py-2 rounded" style={{ top: '350px' }}>
            <span className="text-sm font-medium">Overlay: {overlay}</span>
          </Panel>
        )}

        {/* Layout Controls */}
        <Panel position="bottom-right" className="bg-gray-800 border border-gray-700 text-white px-4 py-3 rounded" style={{ bottom: '180px' }}>
          <div className="text-xs font-bold mb-2 text-gray-400 uppercase">Layout Controls</div>
          <div className="space-y-2 text-xs">
            {/* Layout Mode Buttons */}
            <div className="mb-3">
              <div className="text-gray-400 mb-1.5">Layout Mode:</div>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  onClick={() => setLayoutMode('levels')}
                  className={`px-2 py-1.5 rounded text-xs transition-colors ${
                    layoutMode === 'levels'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  📊 Levels
                </button>
                <button
                  onClick={() => setLayoutMode('hierarchy')}
                  className={`px-2 py-1.5 rounded text-xs transition-colors ${
                    layoutMode === 'hierarchy'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  🌳 Tree
                </button>
                <button
                  onClick={() => setLayoutMode('force')}
                  className={`px-2 py-1.5 rounded text-xs transition-colors ${
                    layoutMode === 'force'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  🌀 Spiral
                </button>
                <button
                  onClick={() => setLayoutMode('circular')}
                  className={`px-2 py-1.5 rounded text-xs transition-colors ${
                    layoutMode === 'circular'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  ⭕ Circle
                </button>
              </div>
            </div>

            {/* Status indicators */}
            <div className="flex items-center justify-between gap-3">
              <span className="text-gray-300">💾 Positions saved</span>
              <span className="text-green-400">✓ Auto</span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-gray-300">🎯 L1-L5 Bands</span>
              <span className={`text-xs ${layoutMode === 'levels' ? 'text-green-400' : 'text-gray-500'}`}>
                {layoutMode === 'levels' ? '✓ Visible' : '○ Hidden'}
              </span>
            </div>

            <button
              onClick={handleResetLayout}
              className="w-full bg-red-900/30 hover:bg-red-900/50 text-red-300 border border-red-800 px-3 py-1.5 rounded transition-colors"
            >
              🔄 Reset Layout
            </button>
            <div className="text-xs text-gray-500 mt-1 italic">
              Drag nodes to reposition. Positions are saved automatically.
            </div>
          </div>
        </Panel>

        {/* Level Structure Legend - only show in levels mode */}
        {layoutMode === 'levels' && (
          <Panel position="top-left" className="bg-gray-800 border border-gray-700 text-white px-4 py-3 rounded" style={{ top: '60px', left: '10px' }}>
            <div className="text-xs font-bold mb-2 text-gray-400 uppercase">Goal Levels</div>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-blue-500/30 border border-blue-500"></div>
                <span className="text-gray-300">L1: Mission (Root)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-green-500/30 border border-green-500"></div>
                <span className="text-gray-300">L2: Strategic</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-orange-500/30 border border-orange-500"></div>
                <span className="text-gray-300">L3: Operational</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-purple-500/30 border border-purple-500"></div>
                <span className="text-gray-300">L4: Tactical</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-red-500/30 border border-red-500"></div>
                <span className="text-gray-300">L5: Atomic</span>
              </div>
            </div>
          </Panel>
        )}
      </ReactFlow>
    </div>
  );
}

// Helper functions

function getDepthLevelStyle(depthLevel: number): React.CSSProperties {
  const levelStyles: Record<number, React.CSSProperties> = {
    0: { // L1: Mission (Root)
      border: '3px solid #3b82f6',
      boxShadow: '0 0 10px rgba(59, 130, 246, 0.3)',
    },
    1: { // L2: Strategic
      border: '2px solid #10b981',
      boxShadow: '0 0 8px rgba(16, 185, 129, 0.2)',
    },
    2: { // L3: Operational
      border: '2px solid #f59e0b',
      boxShadow: '0 0 6px rgba(245, 158, 11, 0.2)',
    },
    3: { // L4: Tactical
      border: '2px solid #8b5cf6',
      boxShadow: '0 0 6px rgba(139, 92, 246, 0.2)',
    },
    4: { // L5: Atomic
      border: '2px solid #ef4444',
      boxShadow: '0 0 4px rgba(239, 68, 68, 0.2)',
    },
  };

  return levelStyles[depthLevel] || levelStyles[0];
}

function getEdgeType(edgeType: string): string {
  switch (edgeType) {
    case 'causal':
      return 'smoothstep';
    case 'dependency':
      return 'straight';
    case 'conflict':
      return 'default';
    case 'reinforcement':
      return 'smoothstep';
    default:
      return 'default';
  }
}

function getEdgeStyle(edgeType: string): React.CSSProperties {
  const baseStyle: React.CSSProperties = {
    strokeWidth: 2.5,
  };

  switch (edgeType) {
    case 'causal':
      return { ...baseStyle, stroke: '#3b82f6' }; // Blue - solid line
    case 'dependency':
      return { ...baseStyle, stroke: '#f59e0b', strokeDasharray: '5,5' }; // Orange - dashed
    case 'conflict':
      return { ...baseStyle, stroke: '#ef4444', strokeWidth: 3 }; // Red - thicker
    case 'reinforcement':
      return { ...baseStyle, stroke: '#10b981' }; // Green - solid
    default:
      return { ...baseStyle, stroke: '#8b5cf6' }; // Purple - default
  }
}

function getHeatmapStyle(node: GoalNode | AgentNode | SkillNode | MemoryNode): React.CSSProperties {
  if (node.type !== 'goal') return {};

  const goalNode = node as GoalNode;
  const intensity = goalNode.conflictScore * 255;

  return {
    background: `rgb(${intensity}, ${255 - intensity}, 100)`,
    border: '2px solid #fbbf24',
  };
}

function getConflictStyle(node: GoalNode | AgentNode | SkillNode | MemoryNode): React.CSSProperties {
  if (node.type !== 'goal') return {};

  const goalNode = node as GoalNode;
  if (goalNode.conflictScore > 0.5) {
    return {
      border: '3px solid #ef4444',
      boxShadow: '0 0 10px rgba(239, 68, 68, 0.5)',
    };
  }
  return {};
}

function getNodeColor(node: GoalNode | AgentNode | SkillNode | MemoryNode): string {
  switch (node.type) {
    case 'goal':
      return '#3b82f6';
    case 'agent':
      return '#10b981';
    case 'skill':
      return '#8b5cf6';
    case 'memory':
      return '#f59e0b';
    default:
      return '#6b7280';
  }
}

// Wrapper with provider
export default function GraphCanvas() {
  return (
    <ReactFlowProvider>
      <GraphCanvasInner />
    </ReactFlowProvider>
  );
}
