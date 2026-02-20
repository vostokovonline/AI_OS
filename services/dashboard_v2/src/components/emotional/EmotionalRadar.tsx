/**
 * v2 UI - Emotional Radar Chart
 *
 * Radar/spider chart for 4-dimensional emotional state
 */

import { FC } from 'react';
import { EmotionalState } from '../../types';

interface EmotionalRadarProps {
  state: EmotionalState;
  size?: number;
}

const EmotionalRadar: FC<EmotionalRadarProps> = ({ state, size = 200 }) => {
  const centerX = size / 2;
  const centerY = size / 2;
  const radius = (size / 2) * 0.8;

  // Normalize valence from [-1, 1] to [0, 1]
  const normalizedValence = (state.valence + 1) / 2;

  // Dimensions and their angles (4 axes)
  const dimensions = [
    { name: 'Arousal', value: state.arousal, angle: -Math.PI / 2, icon: '⚡' },
    { name: 'Valence', value: normalizedValence, angle: 0, icon: '😊' },
    { name: 'Focus', value: state.focus, angle: Math.PI / 2, icon: '🎯' },
    { name: 'Confidence', value: state.confidence, angle: Math.PI, icon: '💪' },
  ];

  // Generate polygon points
  const getPoints = (scale: number = 1) => {
    return dimensions.map((dim) => {
      const r = radius * dim.value * scale;
      const x = centerX + r * Math.cos(dim.angle);
      const y = centerY + r * Math.sin(dim.angle);
      return `${x},${y}`;
    }).join(' ');
  };

  // Get axis end points
  const getAxisEnd = (angle: number) => {
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    return { x, y };
  };

  // Get label position (slightly outside the radius)
  const getLabelPos = (angle: number) => {
    const labelRadius = radius * 1.15;
    const x = centerX + labelRadius * Math.cos(angle);
    const y = centerY + labelRadius * Math.sin(angle);
    return { x, y };
  };

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="bg-gray-900 rounded-lg">
        {/* Background circles (concentric) */}
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <circle
            key={scale}
            cx={centerX}
            cy={centerY}
            r={radius * scale}
            fill="none"
            stroke="#374151"
            strokeWidth="1"
            strokeDasharray="4 2"
          />
        ))}

        {/* Axes */}
        {dimensions.map((dim) => {
          const end = getAxisEnd(dim.angle);
          return (
            <line
              key={dim.name}
              x1={centerX}
              y1={centerY}
              x2={end.x}
              y2={end.y}
              stroke="#4B5563"
              strokeWidth="1"
            />
          );
        })}

        {/* Data polygon */}
        <polygon
          points={getPoints()}
          fill="rgba(59, 130, 246, 0.3)"
          stroke="#3B82F6"
          strokeWidth="2"
        />

        {/* Data points */}
        {dimensions.map((dim) => {
          const r = radius * dim.value;
          const x = centerX + r * Math.cos(dim.angle);
          const y = centerY + r * Math.sin(dim.angle);
          return (
            <circle
              key={dim.name}
              cx={x}
              cy={y}
              r="4"
              fill="#3B82F6"
              stroke="#1F2937"
              strokeWidth="1"
            />
          );
        })}

        {/* Labels */}
        {dimensions.map((dim) => {
          const pos = getLabelPos(dim.angle);
          return (
            <text
              key={dim.name}
              x={pos.x}
              y={pos.y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-gray-400 text-xs"
              fontSize="10"
            >
              {dim.icon}
            </text>
          );
        })}

        {/* Center point */}
        <circle
          cx={centerX}
          cy={centerY}
          r="3"
          fill="#6B7280"
        />
      </svg>

      {/* Legend */}
      <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
        <div className="flex items-center gap-2">
          <span>⚡</span>
          <span className="text-gray-400">Arousal:</span>
          <span className="text-white font-mono">{(state.arousal * 100).toFixed(0)}%</span>
        </div>
        <div className="flex items-center gap-2">
          <span>😊</span>
          <span className="text-gray-400">Valence:</span>
          <span className="text-white font-mono">{state.valence.toFixed(2)}</span>
        </div>
        <div className="flex items-center gap-2">
          <span>🎯</span>
          <span className="text-gray-400">Focus:</span>
          <span className="text-white font-mono">{(state.focus * 100).toFixed(0)}%</span>
        </div>
        <div className="flex items-center gap-2">
          <span>💪</span>
          <span className="text-gray-400">Confidence:</span>
          <span className="text-white font-mono">{(state.confidence * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
};

export default EmotionalRadar;
