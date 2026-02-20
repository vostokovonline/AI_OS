/**
 * v2 UI - Emotional Layer Dashboard
 *
 * Comprehensive emotional state visualization
 * Combines panel, timeline, and radar chart
 */

import React, { useState } from 'react';
import EmotionalPanel from './EmotionalPanel';
import EmotionalTimeline from './EmotionalTimeline';
import EmotionalRadar from './EmotionalRadar';
import { useCurrentEmotionalState } from '../../store/emotionalStore';

const TEST_USER_ID = '00000000-0000-0000-0000-000000000001';

type TabType = 'overview' | 'timeline' | 'radar';

const EmotionalDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const currentState = useCurrentEmotionalState(TEST_USER_ID);

  const defaultState = {
    arousal: 0.5,
    valence: 0.0,
    focus: 0.5,
    confidence: 0.5,
  };

  const state = currentState || defaultState;

  return (
    <div className="h-full flex flex-col bg-gray-900">
      {/* Header with tabs */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <h2 className="text-lg font-semibold text-white">Emotional Layer</h2>
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              activeTab === 'overview'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('timeline')}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              activeTab === 'timeline'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            Timeline
          </button>
          <button
            onClick={() => setActiveTab('radar')}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              activeTab === 'radar'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:text-white'
            }`}
          >
            Radar
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'overview' && <EmotionalPanel />}
        {activeTab === 'timeline' && <EmotionalTimeline />}
        {activeTab === 'radar' && (
          <div className="flex flex-col items-center justify-center h-full">
            <EmotionalRadar state={state} size={300} />
            <div className="mt-6 p-4 bg-gray-800 rounded-lg border border-gray-700 max-w-md">
              <h4 className="text-sm font-semibold text-gray-300 mb-2">About This View</h4>
              <p className="text-xs text-gray-400">
                This radar chart shows the current emotional state across 4 dimensions.
                The blue area represents the emotional state - larger areas indicate higher
                arousal, focus, and confidence, while the position on the valence axis
                (right side) shows positive vs negative sentiment.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-700">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>User ID: {TEST_USER_ID}</span>
          <span>
            {currentState ? 'Live' : 'Baseline'}
            <span className="inline-block w-2 h-2 bg-green-500 rounded-full ml-1 animate-pulse" />
          </span>
        </div>
      </div>
    </div>
  );
};

export default EmotionalDashboard;
