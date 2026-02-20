/**
 * Agent Node Component
 *
 * Custom ReactFlow node for displaying agents
 */

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { AgentNode as AgentNodeType } from '../../../types';

const AgentNode: React.FC<NodeProps<AgentNodeType>> = ({ data, selected }) => {
  return (
    <div
      className={`
        px-4 py-3 rounded-lg border-2 min-w-[180px] max-w-[250px]
        border-green-500 bg-green-900/30
        ${selected ? 'ring-2 ring-white' : ''}
        transition-all duration-200
      `}
    >
      <Handle type="target" position={Position.Top} className="w-2 h-2" />

      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-white uppercase">Agent</span>
        <div
          className={`w-2 h-2 rounded-full ${
            data.status === 'active'
              ? 'bg-green-500 animate-pulse'
              : data.status === 'done'
              ? 'bg-green-500'
              : 'bg-gray-500'
          }`}
        />
      </div>

      {/* Role */}
      <div className="text-white text-sm font-medium mb-1">{data.role}</div>

      {/* Description */}
      {data.description && (
        <div className="text-gray-300 text-xs mb-2 line-clamp-2">{data.description}</div>
      )}

      {/* Current task */}
      {data.currentTask && (
        <div className="text-xs text-blue-300 mb-2">
          <span className="font-medium">Task:</span> {data.currentTask}
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-2 text-xs text-gray-300 mb-2">
        <div>
          <span className="block text-gray-400">Confidence</span>
          <span className="font-medium">{(data.confidence * 100).toFixed(0)}%</span>
        </div>
        <div>
          <span className="block text-gray-400">Success</span>
          <span className="font-medium">{(data.successRate * 100).toFixed(0)}%</span>
        </div>
        <div>
          <span className="block text-gray-400">Cost</span>
          <span className="font-medium">${data.costPerExecution.toFixed(2)}</span>
        </div>
      </div>

      {/* Skills count */}
      <div className="text-xs text-gray-400">
        Skills: {data.skills.length}
      </div>

      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
};

export default memo(AgentNode);
