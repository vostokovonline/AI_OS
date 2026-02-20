/**
 * v2 UI - Timeline Strip
 *
 * Bottom panel showing causality and decision history
 */

import React, { useState, useEffect } from 'react';
import { useUIStore } from '../../store/uiStore';
import { apiClient } from '../../api/client';
import { TimelineSnapshot } from '../../types';
import {
  ChevronLeft,
  ChevronRight,
  GitBranch,
  Clock,
  CheckCircle,
  XCircle,
} from 'lucide-react';

const TimelineStrip: React.FC = () => {
  const { timelineCursor, setTimelineCursor } = useUIStore();
  const [snapshots, setSnapshots] = useState<TimelineSnapshot[]>([]);
  const [selectedSnapshot, setSelectedSnapshot] = useState<TimelineSnapshot | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadTimeline = async () => {
      setLoading(true);
      try {
        const response = await apiClient.getTimelineSnapshot(timelineCursor || undefined) as any;
        // API returns {status, events, total} - we need the events array
        const snapshots = Array.isArray(response) ? response : (response?.events || []);
        setSnapshots(snapshots);
      } catch (error) {
        console.error('Failed to load timeline:', error);
        setSnapshots([]);
      } finally {
        setLoading(false);
      }
    };

    loadTimeline();
  }, [timelineCursor]);

  const handleSnapshotClick = (snapshot: TimelineSnapshot) => {
    setTimelineCursor(snapshot.timestamp);
    setSelectedSnapshot(snapshot);
  };

  return (
    <div className="h-48 bg-gray-800 border-t border-gray-700 flex flex-col">
      {/* Header */}
      <div className="px-4 py-2 border-b border-gray-700 flex items-center justify-between">
        <h3 className="text-white font-bold text-sm flex items-center gap-2">
          <Clock size={16} className="text-blue-400" />
          Timeline & Causality
        </h3>

        {/* Navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              if (!Array.isArray(snapshots)) return;
              const currentIndex = snapshots.findIndex(
                (s) => s.timestamp === timelineCursor
              );
              if (currentIndex > 0) {
                handleSnapshotClick(snapshots[currentIndex - 1]);
              }
            }}
            className="p-1 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
            disabled={!timelineCursor}
          >
            <ChevronLeft size={16} className="text-white" />
          </button>
          <span className="text-gray-400 text-sm">
            {Array.isArray(snapshots) ? `${snapshots.findIndex((s) => s.timestamp === timelineCursor) + 1} / ${snapshots.length}` : '0 / 0'}
          </span>
          <button
            onClick={() => {
              if (!Array.isArray(snapshots)) return;
              const currentIndex = snapshots.findIndex(
                (s) => s.timestamp === timelineCursor
              );
              if (currentIndex < snapshots.length - 1) {
                handleSnapshotClick(snapshots[currentIndex + 1]);
              }
            }}
            className="p-1 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
            disabled={
              !timelineCursor ||
              !Array.isArray(snapshots) ||
              snapshots.findIndex((s) => s.timestamp === timelineCursor) === snapshots.length - 1
            }
          >
            <ChevronRight size={16} className="text-white" />
          </button>
          <button
            onClick={() => {
              setTimelineCursor(null);
              setSelectedSnapshot(null);
            }}
            className="ml-2 px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-500 transition-colors"
          >
            Return to Present
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {loading ? (
          <div className="text-gray-400 text-sm">Loading timeline...</div>
        ) : snapshots.length === 0 ? (
          <div className="text-gray-400 text-sm">No timeline data available</div>
        ) : (
          <div className="space-y-3">
            {/* Snapshots */}
            {snapshots.map((snapshot, index) => {
              const isSelected = snapshot.timestamp === timelineCursor;
              const isLast = index === snapshots.length - 1;

              return (
                <div
                  key={snapshot.timestamp}
                  onClick={() => handleSnapshotClick(snapshot)}
                  className={`
                    p-3 rounded-lg border-2 cursor-pointer transition-all
                    ${
                      isSelected
                        ? 'border-blue-500 bg-blue-900/30'
                        : 'border-gray-700 bg-gray-700/50 hover:border-gray-600'
                    }
                  `}
                >
                  {/* Timestamp */}
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {isLast ? (
                        <CheckCircle size={16} className="text-green-400" />
                      ) : (
                        <GitBranch size={16} className="text-gray-400" />
                      )}
                      <span className="text-white text-sm font-medium">
                        {new Date(snapshot.timestamp).toLocaleString()}
                      </span>
                      {isLast && (
                        <span className="text-xs bg-green-800 text-white px-2 py-0.5 rounded">
                          Current
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Active Goals */}
                  <div className="mb-2">
                    <span className="text-gray-400 text-xs">Active Goals: </span>
                    <span className="text-white text-sm">
                      {snapshot.activeGoals?.length || 0} goals
                    </span>
                  </div>

                  {/* Decisions */}
                  {snapshot.decisions && snapshot.decisions.length > 0 && (
                    <div>
                      <span className="text-gray-400 text-xs">Decisions: </span>
                      <div className="mt-1 space-y-1">
                        {snapshot.decisions.slice(0, 2).map((decision) => (
                          <div
                            key={decision.id}
                            className="flex items-start gap-2 text-xs"
                          >
                            {decision.rejected ? (
                              <XCircle size={12} className="text-red-400 mt-0.5" />
                            ) : (
                              <CheckCircle size={12} className="text-green-400 mt-0.5" />
                            )}
                            <div className="flex-1">
                              <div className="text-gray-300">{decision.action}</div>
                              <div className="text-gray-500">
                                {decision.rationale ? decision.rationale.slice(0, 60) + '...' : 'No rationale'}
                              </div>
                            </div>
                          </div>
                        ))}
                        {(snapshot.decisions?.length || 0) > 2 && (
                          <div className="text-gray-500 text-xs">
                            +{snapshot.decisions.length - 2} more decisions
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Selected snapshot details */}
      {selectedSnapshot && (
        <div className="border-t border-gray-700 p-4 bg-gray-900/50">
          <h4 className="text-white text-sm font-bold mb-2">Snapshot Details</h4>

          <div className="grid grid-cols-3 gap-4 text-xs">
            {/* Knowledge State */}
            <div>
              <div className="text-gray-400 mb-1">Knowledge State</div>
              <div className="space-y-1">
                <div className="text-blue-300">
                  Facts: {selectedSnapshot.knowledgeState?.facts?.length || 0}
                </div>
                <div className="text-yellow-300">
                  Assumptions: {selectedSnapshot.knowledgeState?.assumptions?.length || 0}
                </div>
                <div className="text-red-300">
                  Uncertainties: {selectedSnapshot.knowledgeState?.uncertainties?.length || 0}
                </div>
              </div>
            </div>

            {/* Constraints */}
            <div>
              <div className="text-gray-400 mb-1">Constraints</div>
              <div className="space-y-1">
                {selectedSnapshot.constraints.ethics && selectedSnapshot.constraints.ethics.length > 0 && (
                  <div className="text-green-300">
                    Ethics: {selectedSnapshot.constraints.ethics.join(', ')}
                  </div>
                )}
                {selectedSnapshot.constraints.budget && (
                  <div className="text-yellow-300">
                    Budget: ${selectedSnapshot.constraints.budget}
                  </div>
                )}
                {selectedSnapshot.constraints.timeHorizon && (
                  <div className="text-blue-300">
                    Time: {selectedSnapshot.constraints.timeHorizon}
                  </div>
                )}
              </div>
            </div>

            {/* Decisions Summary */}
            <div>
              <div className="text-gray-400 mb-1">Decisions</div>
              <div className="space-y-1">
                <div className="text-green-300">
                  Accepted:{' '}
                  {selectedSnapshot.decisions?.filter((d) => !d.rejected).length || 0}
                </div>
                <div className="text-red-300">
                  Rejected:{' '}
                  {selectedSnapshot.decisions?.filter((d) => d.rejected).length || 0}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TimelineStrip;
