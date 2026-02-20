/**
 * Skill Node Component
 *
 * Custom ReactFlow node for displaying skills
 */

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { SkillNode as SkillNodeType } from '../../../types';

const SkillNode: React.FC<NodeProps<SkillNodeType>> = ({ data, selected }) => {
  return (
    <div
      className={`
        px-4 py-3 rounded-lg border-2 min-w-[180px] max-w-[250px]
        border-purple-500 bg-purple-900/30
        ${selected ? 'ring-2 ring-white' : ''}
        transition-all duration-200
      `}
    >
      <Handle type="target" position={Position.Top} className="w-2 h-2" />

      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-white uppercase">Skill</span>
        <div
          className={`w-2 h-2 rounded-full ${
            data.status === 'active'
              ? 'bg-purple-500 animate-pulse'
              : 'bg-gray-500'
          }`}
        />
      </div>

      {/* Name */}
      <div className="text-white text-sm font-medium mb-1">{data.name}</div>

      {/* Description */}
      {data.description && (
        <div className="text-gray-300 text-xs mb-2 line-clamp-2">{data.description}</div>
      )}

      {/* I/O */}
      <div className="mb-2">
        <div className="text-xs text-gray-400 mb-1">Inputs/Outputs</div>
        <div className="flex flex-wrap gap-1">
          {data.inputs.slice(0, 2).map((input, i) => (
            <span key={i} className="text-xs bg-purple-800 text-white px-2 py-0.5 rounded">
              {input}
            </span>
          ))}
          {data.inputs.length > 2 && (
            <span className="text-xs text-gray-400">+{data.inputs.length - 2}</span>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-2 text-xs text-gray-300 mb-2">
        <div>
          <span className="block text-gray-400">Success</span>
          <span className="font-medium">{(data.successRate * 100).toFixed(0)}%</span>
        </div>
        <div>
          <span className="block text-gray-400">Latency</span>
          <span className="font-medium">{data.avgLatencyMs}ms</span>
        </div>
        <div>
          <span className="block text-gray-400">Cost</span>
          <span className="font-medium">${data.costPerUse.toFixed(2)}</span>
        </div>
      </div>

      {/* Usage */}
      <div className="text-xs text-gray-400">
        Used {data.usageCount} times
      </div>

      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
};

export default memo(SkillNode);
