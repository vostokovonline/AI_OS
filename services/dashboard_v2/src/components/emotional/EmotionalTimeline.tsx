/**
 * v2 UI - Emotional Timeline
 *
 * Displays emotional state history over time
 */

import { useEffect } from 'react';
import { useEmotionalStore, useEmotionalHistory } from '../../store/emotionalStore';
import { EmotionalState } from '../../types';

const TEST_USER_ID = '00000000-0000-0000-0000-000000000001';

const EmotionalTimeline: React.FC = () => {
  const history = useEmotionalHistory(TEST_USER_ID);
  const loading = useEmotionalStore((state) => state.loading);
  const error = useEmotionalStore((state) => state.error);
  const fetchHistory = useEmotionalStore((state) => state.fetchHistory);

  useEffect(() => {
    fetchHistory(TEST_USER_ID, 50);
  }, [fetchHistory]);

  if (loading && history.length === 0) {
    return (
      <div className="p-4 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-900/20 border border-red-700 rounded-lg">
        <p className="text-red-400 text-sm">Error loading history: {error}</p>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="p-4 bg-gray-800 rounded-lg border border-gray-700 text-center">
        <p className="text-gray-400 text-sm">No emotional history available</p>
      </div>
    );
  }

  // Calculate min/max for scaling
  const minValence = Math.min(...history.map((s) => s.valence));
  const maxValence = Math.max(...history.map((s) => s.valence));

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Emotional Timeline</h3>
        <span className="text-xs text-gray-500">
          Last {history.length} states
        </span>
      </div>

      {/* Timeline Visualization */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {history.map((state: EmotionalState, index: number) => {
          const timeStr = state.timestamp
            ? new Date(state.timestamp).toLocaleTimeString()
            : `State ${index + 1}`;

          return (
            <div
              key={state.timestamp || index}
              className="flex items-center p-2 bg-gray-800 rounded-lg hover:bg-gray-750 transition-colors"
            >
              {/* Time */}
              <div className="w-24 text-xs text-gray-400 flex-shrink-0">
                {timeStr}
              </div>

              {/* Bars */}
              <div className="flex-1 grid grid-cols-4 gap-2">
                {/* Arousal */}
                <div className="flex flex-col">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-500">⚡</span>
                    <span className="text-gray-300">{(state.arousal * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        state.arousal > 0.7 ? 'bg-red-500' :
                        state.arousal > 0.4 ? 'bg-yellow-500' :
                        'bg-blue-500'
                      }`}
                      style={{ width: `${state.arousal * 100}%` }}
                    />
                  </div>
                </div>

                {/* Valence */}
                <div className="flex flex-col">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-500">😊</span>
                    <span className="text-gray-300">{state.valence.toFixed(2)}</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        state.valence > 0.3 ? 'bg-green-500' :
                        state.valence < -0.3 ? 'bg-red-500' :
                        'bg-gray-500'
                      }`}
                      style={{
                        width: `${((state.valence - minValence) / (maxValence - minValence || 1)) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Focus */}
                <div className="flex flex-col">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-500">🎯</span>
                    <span className="text-gray-300">{(state.focus * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        state.focus > 0.7 ? 'bg-green-500' :
                        state.focus > 0.4 ? 'bg-blue-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${state.focus * 100}%` }}
                    />
                  </div>
                </div>

                {/* Confidence */}
                <div className="flex flex-col">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-500">💪</span>
                    <span className="text-gray-300">{(state.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        state.confidence > 0.7 ? 'bg-green-500' :
                        state.confidence > 0.4 ? 'bg-blue-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${state.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="flex flex-wrap gap-4 text-xs text-gray-400">
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-red-500 rounded"></span>
            <span>High/Low</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-yellow-500 rounded"></span>
            <span>Medium</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-blue-500 rounded"></span>
            <span>Baseline</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-green-500 rounded"></span>
            <span>Optimal</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmotionalTimeline;
