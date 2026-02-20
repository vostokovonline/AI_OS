/**
 * Dependency Tree View
 *
 * Hierarchical tree visualization showing goal dependencies (inspired by v1 dashboard)
 */

import React, { useMemo, useState } from 'react';
import { useGraphStore } from '../../store/graphStore';
import { useUIStore } from '../../store/uiStore';
import {
  Clock,
  CheckCircle,
  Zap,
  XCircle,
  ChevronDown,
  ChevronRight,
  Network,
} from 'lucide-react';

interface TreeNodeProps {
  goal: any;
  level: number;
  isExpanded: boolean;
  onToggleExpand: (nodeId: string) => void;
  onGoalClick?: (goal: any) => void;
}

const TreeNode: React.FC<TreeNodeProps> = ({ goal, level, isExpanded, onToggleExpand, onGoalClick }) => {
  const hasChildren = goal.childIds && goal.childIds.length > 0;

  const getStatusIcon = () => {
    switch (goal.status) {
      case 'pending':
        return <Clock size={14} className="text-gray-400" />;
      case 'active':
        return <Zap size={14} className="text-blue-400" />;
      case 'done':
        return <CheckCircle size={14} className="text-green-400" />;
      case 'blocked':
        return <XCircle size={14} className="text-red-400" />;
      default:
        return null;
    }
  };

  const getStatusColor = () => {
    switch (goal.status) {
      case 'pending':
        return 'border-gray-600 hover:border-gray-500';
      case 'active':
        return 'border-blue-600 hover:border-blue-500 bg-blue-900/10';
      case 'done':
        return 'border-green-600 hover:border-green-500 bg-green-900/10';
      case 'blocked':
        return 'border-red-600 hover:border-red-500 bg-red-900/10';
      default:
        return 'border-gray-600';
    }
  };

  const getGoalTypeColor = () => {
    switch (goal.goalType) {
      case 'achievable':
        return 'text-blue-400';
      case 'philosophical':
        return 'text-purple-400';
      case 'unachievable':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getProgressColor = () => {
    if (goal.status === 'done') return 'bg-green-500';
    if (goal.status === 'active') return 'bg-blue-500';
    if (goal.status === 'blocked') return 'bg-red-500';
    return 'bg-gray-600';
  };

  return (
    <div className="select-none">
      {/* Tree Node */}
      <div
        className={`
          flex items-center gap-2 py-2 px-3 rounded-lg border-l-4
          ${getStatusColor()}
          transition-all duration-200 cursor-pointer hover:bg-gray-800/50
        `}
        style={{ marginLeft: `${level * 24}px` }}
        onClick={() => {
          onGoalClick?.(goal);
          if (hasChildren) {
            onToggleExpand(goal.id);
          }
        }}
      >
        {/* Expand/Collapse */}
        <div className="w-5 flex-shrink-0 flex justify-center">
          {hasChildren ? (
            <button className="text-gray-500 hover:text-white transition-colors">
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </button>
          ) : (
            <div className="w-3.5 h-3.5 rounded-full bg-gray-700" />
          )}
        </div>

        {/* Status Icon */}
        <div className="w-6 flex-shrink-0 flex justify-center">{getStatusIcon()}</div>

        {/* Goal Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-white truncate">{goal.intent}</span>
            {hasChildren && (
              <span className="text-xs text-gray-500 flex-shrink-0">
                ({goal.childIds.length} subgoal{goal.childIds.length > 1 ? 's' : ''})
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 text-xs">
            {/* Goal Type */}
            <span className={`px-1.5 py-0.5 rounded ${getGoalTypeColor()} bg-gray-800`}>
              {goal.goalType}
            </span>

            {/* Progress Bar */}
            <div className="flex items-center gap-1.5 flex-1 min-w-0">
              <div className="w-24 bg-gray-700 rounded-full h-1.5 overflow-hidden">
                <div
                  className={`h-1.5 rounded-full ${getProgressColor()}`}
                  style={{ width: `${goal.progress * 100}%` }}
                />
              </div>
              <span className="text-gray-400 flex-shrink-0">{Math.round(goal.progress * 100)}%</span>
            </div>

            {/* Metrics */}
            <div className="flex items-center gap-2 text-gray-500 flex-shrink-0">
              <span title="Feasibility">⚡ {Math.round(goal.feasibility * 100)}%</span>
              <span title="Conflict">⚠️ {Math.round(goal.conflictScore * 100)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const DependencyTree: React.FC = () => {
  const { nodes, getFilteredNodes } = useGraphStore();
  const { dispatchEvent } = useUIStore();
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  const toggleExpand = (nodeId: string) => {
    setExpandedNodes((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  // Handle goal click to show in Inspector
  const handleGoalClick = (goal: any) => {
    dispatchEvent({
      type: 'SELECT_NODE',
      nodeId: goal.id,
      nodeType: 'goal',
    });
  };

  // Build tree structure
  const { treeData, goalsMap, stats } = useMemo(() => {
    const filteredNodes = getFilteredNodes().filter((n: any) => n.type === 'goal');
    const goalNodes = filteredNodes as any[];

    if (goalNodes.length === 0) {
      return { treeData: [], goalsMap: new Map(), stats: { total: 0, done: 0, active: 0, pending: 0, blocked: 0 } };
    }

    // Build goals map
    const map = new Map(goalNodes.map(g => [g.id, g]));

    // Calculate statistics
    const stats = {
      total: goalNodes.length,
      done: goalNodes.filter(g => g.status === 'done').length,
      active: goalNodes.filter(g => g.status === 'active').length,
      pending: goalNodes.filter(g => g.status === 'pending').length,
      blocked: goalNodes.filter(g => g.status === 'blocked').length,
    };

    // Get root nodes (no parents)
    const rootNodes = goalNodes.filter(g => !g.parentId);

    return {
      treeData: rootNodes,
      goalsMap: map,
      stats,
    };
  }, [nodes, getFilteredNodes]);

  // Recursively render tree
  const renderTree = (goal: any, level: number): React.ReactNode => {
    const isExpanded = expandedNodes.has(goal.id);
    const hasChildren = goal.childIds && goal.childIds.length > 0;

    return (
      <div key={goal.id}>
        <TreeNode
          goal={goal}
          level={level}
          isExpanded={isExpanded}
          onToggleExpand={toggleExpand}
          onGoalClick={handleGoalClick}
        />
        {isExpanded && hasChildren && (
          <div>
            {goal.childIds.map((childId: string) => {
              const childGoal = goalsMap.get(childId);
              if (childGoal) {
                return renderTree(childGoal, level + 1);
              }
              return null;
            })}
          </div>
        )}
      </div>
    );
  };

  if (treeData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900">
        <div className="text-gray-400">No goals to display</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Network size={18} className="text-green-400" />
            <h2 className="text-lg font-semibold text-white">Dependency Tree</h2>
          </div>

          {/* Statistics */}
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-2">
              <span className="text-gray-400">Total:</span>
              <span className="text-white font-medium">{stats.total}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1">
                <CheckCircle size={12} className="text-green-400" />
                <span className="text-gray-400">Done:</span>
                <span className="text-green-400 font-medium">{stats.done}</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1">
                <Zap size={12} className="text-blue-400" />
                <span className="text-gray-400">Active:</span>
                <span className="text-blue-400 font-medium">{stats.active}</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1">
                <Clock size={12} className="text-gray-400" />
                <span className="text-gray-400">Pending:</span>
                <span className="text-gray-300 font-medium">{stats.pending}</span>
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1">
                <XCircle size={12} className="text-red-400" />
                <span className="text-gray-400">Blocked:</span>
                <span className="text-red-400 font-medium">{stats.blocked}</span>
              </span>
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-2 text-xs">
          <span className="text-gray-400">Goal Types:</span>
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-blue-400" />
            <span className="text-gray-300">Achievable</span>
          </span>
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-purple-400" />
            <span className="text-gray-300">Philosophical</span>
          </span>
          <span className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-red-400" />
            <span className="text-gray-300">Unachievable</span>
          </span>
          <span className="ml-4 text-gray-400">Click on goals to expand/collapse</span>
        </div>
      </div>

      {/* Tree Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-1">
          {treeData.map((goal) => renderTree(goal, 0))}
        </div>
      </div>
    </div>
  );
};

export default DependencyTree;
