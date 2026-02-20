/**
 * Goal Node Component
 *
 * Custom ReactFlow node for displaying goals with execution animations and floating labels
 */

import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { GoalNode as GoalNodeType } from '../../../types';
import { useGraphStore } from '../../../store/graphStore';
import FloatingLabel from './FloatingLabel';

const GoalNode: React.FC<NodeProps<GoalNodeType>> = ({ data, selected }) => {
  const { toggleCollapse, isNodeCollapsed } = useGraphStore();
  const isCollapsed = isNodeCollapsed(data.id);
  const hasChildren = data.childIds && data.childIds.length > 0;
  const [isExecuting, setIsExecuting] = useState(false);
  const [showCompletion, setShowCompletion] = useState(false);
  const [executionDuration, setExecutionDuration] = useState<number | undefined>();

  // Simulate execution animation for demo
  useEffect(() => {
    if (data.status === 'active' && !isExecuting) {
      setIsExecuting(true);
      setShowCompletion(false);
      setExecutionDuration(undefined);
    }
    if (data.status === 'done' && !showCompletion) {
      setShowCompletion(true);
      setIsExecuting(false);
      // Simulate random duration for demo
      setExecutionDuration(Math.floor(Math.random() * 2000) + 500);
    }
    if (data.status === 'pending') {
      setIsExecuting(false);
      setShowCompletion(false);
    }
  }, [data.status]);

  const getNodeColor = () => {
    switch (data.goalType) {
      case 'achievable':
        return 'border-blue-500 bg-blue-900/30';
      case 'unachievable':
        return 'border-red-500 bg-red-900/30';
      case 'philosophical':
        return 'border-purple-500 bg-purple-900/30';
      default:
        return 'border-gray-500 bg-gray-800';
    }
  };

  const getStatusColor = () => {
    switch (data.status) {
      case 'pending':
        return 'bg-gray-500';
      case 'active':
        return 'bg-blue-500';
      case 'done':
        return 'bg-green-500';
      case 'blocked':
        return 'bg-red-500';
      case 'failed':
        return 'bg-gray-700';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusAnimation = () => {
    if (data.status === 'active') {
      return 'node-executing node-executing-border';
    }
    if (data.status === 'done' && showCompletion) {
      return 'node-complete';
    }
    return '';
  };

  const getFloatingLabelText = () => {
    if (data.status === 'active') {
      return `Executing: ${data.intent.substring(0, 20)}...`;
    }
    if (data.status === 'done' && showCompletion) {
      return 'Completed!';
    }
    return undefined;
  };

  return (
    <div className="relative">
      {/* Floating Label above node */}
      <FloatingLabel
        status={data.status}
        title={getFloatingLabelText()}
        progress={data.progress}
        duration={executionDuration}
        visible={data.status === 'active' || (data.status === 'done' && showCompletion)}
      />

      <div
        className={`
          px-4 py-3 rounded-lg border-2 min-w-[200px] max-w-[300px]
          ${getNodeColor()} ${selected ? 'ring-2 ring-white' : ''}
          ${isCollapsed ? 'border-dashed opacity-75' : ''}
          transition-all duration-200 relative
          ${getStatusAnimation()}
        `}
      >
        {/* Active glow effect */}
        {data.status === 'active' && (
          <div className="absolute inset-0 rounded-lg bg-blue-500/20 animate-pulse pointer-events-none" />
        )}

        <Handle type="target" position={Position.Top} className="w-2 h-2" />

        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            {/* Collapse/Expand button */}
            {hasChildren && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleCollapse(data.id);
                }}
                className="flex items-center justify-center w-5 h-5 text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700"
                title={isCollapsed ? `Expand (${data.childIds.length} children)` : `Collapse (${data.childIds.length} children)`}
              >
                {isCollapsed ? (
                  <ChevronRight size={14} />
                ) : (
                  <ChevronDown size={14} />
                )}
              </button>
            )}
            <span className="text-xs font-bold text-white uppercase">Goal</span>
            {hasChildren && !isCollapsed && (
              <span className="text-xs text-gray-400">({data.childIds.length})</span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Execution spinner */}
            {data.status === 'active' && (
              <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full spinner" />
            )}
            {/* Completion checkmark */}
            {data.status === 'done' && showCompletion && (
              <svg className="w-4 h-4 text-green-400 checkmark-animate" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            )}
            <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
          </div>
        </div>

        {/* Intent */}
        <div className="text-white text-sm font-medium mb-2 line-clamp-2">
          {data.intent}
        </div>

        {/* Progress bar with shimmer effect when active */}
        <div className="w-full bg-gray-700 rounded-full h-1.5 mb-2 overflow-hidden">
          <div
            className={`h-1.5 rounded-full transition-all ${
              data.status === 'active' ? 'bg-blue-500 progress-animated' : 'bg-blue-500'
            }`}
            style={{ width: `${data.progress * 100}%` }}
          />
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-3 gap-2 text-xs text-gray-300">
          <div>
            <span className="block text-gray-400">Feasibility</span>
            <span className="font-medium">{(data.feasibility * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span className="block text-gray-400">Conflict</span>
            <span className="font-medium">{(data.conflictScore * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span className="block text-gray-400">Uncertainty</span>
            <span className="font-medium">{(data.uncertainty * 100).toFixed(0)}%</span>
          </div>
        </div>

        <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
      </div>
    </div>
  );
};

export default memo(GoalNode);
