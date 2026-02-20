/**
 * Memory Node Component
 *
 * Custom ReactFlow node for displaying memories
 */

import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { MemoryNode as MemoryNodeType } from '../../../types';

const MemoryNode: React.FC<NodeProps<MemoryNodeType>> = ({ data, selected }) => {
  const getMemoryTypeColor = () => {
    switch (data.memoryType) {
      case 'recent_failure':
        return 'border-red-500 bg-red-900/30';
      case 'resource_exhaustion':
        return 'border-orange-500 bg-orange-900/30';
      case 'false_success':
        return 'border-yellow-500 bg-yellow-900/30';
      case 'overfitting':
        return 'border-purple-500 bg-purple-900/30';
      default:
        return 'border-gray-500 bg-gray-800';
    }
  };

  const getIntensityColor = () => {
    const intensity = data.intensity;
    if (intensity > 0.7) return 'bg-red-500';
    if (intensity > 0.4) return 'bg-orange-500';
    return 'bg-yellow-500';
  };

  return (
    <div
      className={`
        px-4 py-3 rounded-lg border-2 min-w-[180px] max-w-[250px]
        ${getMemoryTypeColor()} ${selected ? 'ring-2 ring-white' : ''}
        transition-all duration-200
      `}
    >
      <Handle type="target" position={Position.Top} className="w-2 h-2" />

      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-white uppercase">Memory</span>
        <div className={`w-2 h-2 rounded-full ${getIntensityColor()}`} />
      </div>

      {/* Memory Type */}
      <div className="text-white text-sm font-medium mb-1 capitalize">
        {data.memoryType.replace(/_/g, ' ')}
      </div>

      {/* Target */}
      <div className="text-xs text-gray-300 mb-2">
        <span className="font-medium">Target:</span> {data.target}
      </div>

      {/* Intensity bar */}
      <div className="mb-2">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Intensity</span>
          <span>{(data.intensity * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-1.5">
          <div
            className={`${getIntensityColor()} h-1.5 rounded-full`}
            style={{ width: `${data.intensity * 100}%` }}
          />
        </div>
      </div>

      {/* TTL */}
      <div className="text-xs text-gray-400">
        TTL: {data.ttl} cycles
      </div>

      {/* Created at */}
      <div className="text-xs text-gray-500 mt-1">
        {new Date(data.createdAt).toLocaleString()}
      </div>

      <Handle type="source" position={Position.Bottom} className="w-2 h-2" />
    </div>
  );
};

export default memo(MemoryNode);
