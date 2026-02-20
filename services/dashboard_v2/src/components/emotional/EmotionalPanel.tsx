/**
 * v2 UI - Emotional Layer Panel
 *
 * Displays current emotional state and influence
 */

import { useEffect } from 'react';
import { useEmotionalStore, useCurrentEmotionalState, useEmotionalInfluence } from '../../store/emotionalStore';
import { EmotionalState as EmotionalStateType } from '../../types';

const TEST_USER_ID = '00000000-0000-0000-0000-000000000001';

// Helper to get color based on value
const getValueColor = (value: number, type: 'arousal' | 'valence' | 'focus' | 'confidence'): string => {
  const normalized = type === 'valence' ? (value + 1) / 2 : value; // valence is -1..1
  if (normalized < 0.3) return 'bg-red-500';
  if (normalized < 0.5) return 'bg-yellow-500';
  if (normalized < 0.7) return 'bg-blue-500';
  return 'bg-green-500';
};

const getLabelForValue = (value: number, type: 'arousal' | 'valence' | 'focus' | 'confidence'): string => {
  if (type === 'valence') {
    if (value < -0.4) return 'Negative';
    if (value > 0.4) return 'Positive';
    return 'Neutral';
  }
  if (value < 0.4) return 'Low';
  if (value > 0.7) return 'High';
  return 'Medium';
};

const EmotionalDimension: React.FC<{
  label: string;
  value: number;
  type: 'arousal' | 'valence' | 'focus' | 'confidence';
  icon: string;
}> = ({ label, value, type, icon }) => {
  const displayValue = type === 'valence' ? value.toFixed(2) : (value * 100).toFixed(0) + '%';
  const barWidth = type === 'valence' ? ((value + 1) / 2) * 100 : value * 100;

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <span className="text-gray-400 text-sm">
          <span className="mr-1">{icon}</span>
          {label}
        </span>
        <span className="text-white font-mono text-sm">
          {displayValue} ({getLabelForValue(value, type)})
        </span>
      </div>
      <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${getValueColor(value, type)} transition-all duration-300`}
          style={{ width: `${Math.max(0, Math.min(100, barWidth))}%` }}
        />
      </div>
    </div>
  );
};

const EmotionalInfluenceDisplay: React.FC<{ influence: any }> = ({ influence }) => {
  if (!influence) return null;

  return (
    <div className="mt-4 p-3 bg-gray-800 rounded-lg border border-gray-700">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Current Influence</h4>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-gray-400">Complexity:</span>
          <span className="ml-2 text-white">
            {(influence.complexity_penalty * 100).toFixed(0)}% penalty
          </span>
        </div>
        <div>
          <span className="text-gray-400">Exploration:</span>
          <span className="ml-2 text-white">
            {influence.exploration_bias > 0.2 ? 'Aggressive' :
             influence.exploration_bias < -0.2 ? 'Conservative' : 'Balanced'}
          </span>
        </div>
        <div>
          <span className="text-gray-400">Explanation:</span>
          <span className="ml-2 text-white">
            {(influence.explanation_depth * 100).toFixed(0)}% depth
          </span>
        </div>
        <div>
          <span className="text-gray-400">Pace:</span>
          <span className="ml-2 text-white">
            {influence.pace_modifier > 0.2 ? 'Fast' :
             influence.pace_modifier < -0.2 ? 'Slow' : 'Normal'}
          </span>
        </div>
      </div>
    </div>
  );
};

const EmotionalPanel: React.FC = () => {
  const currentState = useCurrentEmotionalState(TEST_USER_ID);
  const influence = useEmotionalInfluence(TEST_USER_ID);
  const loading = useEmotionalStore((state) => state.loading);
  const error = useEmotionalStore((state) => state.error);
  const fetchCurrentState = useEmotionalStore((state) => state.fetchCurrentState);

  useEffect(() => {
    // Fetch on mount
    fetchCurrentState(TEST_USER_ID);

    // Poll every 5 seconds
    const interval = setInterval(() => {
      fetchCurrentState(TEST_USER_ID);
    }, 5000);

    return () => clearInterval(interval);
  }, [fetchCurrentState]);

  if (loading && !currentState) {
    return (
      <div className="p-4 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg">
        <p className="text-red-400 text-sm">Error loading emotional state: {error}</p>
      </div>
    );
  }

  // Default state if none exists
  const state: EmotionalStateType = currentState || {
    arousal: 0.5,
    valence: 0.0,
    focus: 0.5,
    confidence: 0.5,
  };

  return (
    <div className="p-4 bg-gray-900 rounded-lg border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Emotional State</h3>
        {state.timestamp && (
          <span className="text-xs text-gray-500">
            {new Date(state.timestamp).toLocaleTimeString()}
          </span>
        )}
      </div>

      {/* Emotional Dimensions */}
      <EmotionalDimension
        label="Arousal"
        value={state.arousal}
        type="arousal"
        icon="⚡"
      />
      <EmotionalDimension
        label="Valence"
        value={state.valence}
        type="valence"
        icon="😊"
      />
      <EmotionalDimension
        label="Focus"
        value={state.focus}
        type="focus"
        icon="🎯"
      />
      <EmotionalDimension
        label="Confidence"
        value={state.confidence}
        type="confidence"
        icon="💪"
      />

      {/* Current Influence */}
      <EmotionalInfluenceDisplay influence={influence} />
    </div>
  );
};

export default EmotionalPanel;
