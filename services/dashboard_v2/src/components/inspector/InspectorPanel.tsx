/**
 * v2 UI - Inspector Panel
 *
 * Context-aware panel showing detailed information about selected nodes
 */

import React, { useMemo, useState, useEffect } from 'react';
import { useUIStore } from '../../store/uiStore';
import { useGraphStore } from '../../store/graphStore';
import { InspectorContext, Artifact } from '../../types';
import { apiClient } from '../../api/client';
import { X, AlertTriangle, CheckCircle, Clock, Zap, Calendar, FileText, Target, Package, Database } from 'lucide-react';
import { ArtifactModal } from '../artifacts/ArtifactModal';
import { ArtifactCard } from '../artifacts/ArtifactCard';

const InspectorPanel: React.FC = () => {
  const { focus, setFocus } = useUIStore();
  const { getNode } = useGraphStore();

  // Get selected node directly from graph
  const selectedNode = focus.nodeId ? getNode(focus.nodeId) : null;

  if (!focus.nodeId || !selectedNode) {
    return (
      <div className="w-96 bg-gray-800 border-l border-gray-700 p-6 text-gray-400 text-sm flex flex-col items-center justify-center">
        <Target size={48} className="mb-3 opacity-50" />
        <p>Click on any goal to view details</p>
        <p className="text-xs mt-2 opacity-70">View execution plan, results, and metrics</p>
      </div>
    );
  }

  const node = selectedNode as any;

  // Create minimal context for mock data
  const context: InspectorContext = {
    node: node,
    systemState: {
      totalActiveGoals: 0,
      resourceUsage: 0,
      errorRate: 0,
      recentFailures: 0,
    },
    conflicts: [],
    history: [],
    suggestions: [],
  };

  return (
    <div className="w-96 bg-gray-800 border-l border-gray-700 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 flex items-center justify-between">
        <h2 className="text-white font-bold text-lg capitalize flex items-center gap-2">
          {node.type === 'goal' && <Target size={20} className="text-blue-400" />}
          {node.type} Details
        </h2>
        <button
          onClick={() => setFocus(null, null)}
          className="text-gray-400 hover:text-white transition-colors"
        >
          <X size={20} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Node-specific content - simplified for mock data */}
        {node.type === 'goal' && <GoalInspector node={node} context={context} />}
        {node.type === 'agent' && <AgentInspector node={node} context={context} />}
        {node.type === 'skill' && <SkillInspector node={node} context={context} />}
        {node.type === 'memory' && <MemoryInspector node={node} context={context} />}
      </div>
    </div>
  );
};

// Goal Inspector
const GoalInspector: React.FC<{ node: any; context: InspectorContext }> = ({ node, context }) => {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [showArtifacts, setShowArtifacts] = useState(true);
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);

  // Load artifacts when goal is selected
  useEffect(() => {
    const loadArtifacts = async () => {
      if (!node.id) return;

      setLoadingArtifacts(true);
      try {
        const response = await apiClient.fetchGoalArtifacts(node.id);
        if (response.status === 'ok' && response.artifacts) {
          setArtifacts(response.artifacts);
        }
      } catch (error) {
        console.error('[Inspector] Failed to load artifacts:', error);
      } finally {
        setLoadingArtifacts(false);
      }
    };

    loadArtifacts();
  }, [node.id]);

  // Mock execution plan based on goal status
  const executionPlan = useMemo(() => {
    if (node.status === 'done') {
      return [
        { step: 'Analysis', status: 'completed', result: 'Analyzed requirements and constraints' },
        { step: 'Planning', status: 'completed', result: 'Created detailed execution plan' },
        { step: 'Execution', status: 'completed', result: 'Successfully implemented all features' },
        { step: 'Verification', status: 'completed', result: 'Verified and tested implementation' },
      ];
    } else if (node.status === 'active') {
      return [
        { step: 'Analysis', status: 'completed', result: 'Analyzed requirements' },
        { step: 'Planning', status: 'completed', result: 'Created execution plan' },
        { step: 'Execution', status: 'in_progress', result: 'Currently implementing...' },
        { step: 'Verification', status: 'pending', result: 'Pending' },
      ];
    } else {
      return [
        { step: 'Analysis', status: 'pending', result: 'Pending' },
        { step: 'Planning', status: 'pending', result: 'Pending' },
        { step: 'Execution', status: 'pending', result: 'Pending' },
        { step: 'Verification', status: 'pending', result: 'Pending' },
      ];
    }
  }, [node.status]);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <span className="px-2 py-0.5 bg-green-900/50 text-green-400 text-xs rounded-full flex items-center gap-1">
          <CheckCircle size={10} /> Done
        </span>;
      case 'in_progress':
        return <span className="px-2 py-0.5 bg-blue-900/50 text-blue-400 text-xs rounded-full flex items-center gap-1">
          <Zap size={10} /> Active
        </span>;
      case 'pending':
        return <span className="px-2 py-0.5 bg-gray-700 text-gray-400 text-xs rounded-full flex items-center gap-1">
          <Clock size={10} /> Pending
        </span>;
      default:
        return <span className="px-2 py-0.5 bg-gray-700 text-gray-400 text-xs rounded-full">{status}</span>;
    }
  };

  return (
    <>
      {/* Intent */}
      <div>
        <h3 className="text-gray-400 text-xs uppercase mb-2 flex items-center gap-1">
          <FileText size={12} />
          Goal
        </h3>
        <p className="text-white text-sm font-medium">{node.intent}</p>
      </div>

      {/* Type & Status */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-gray-700 p-2 rounded">
          <div className="text-gray-400 text-xs">Type</div>
          <div className="text-white text-sm capitalize">{node.goalType}</div>
        </div>
        <div className="bg-gray-700 p-2 rounded">
          <div className="text-gray-400 text-xs">Status</div>
          <div className="text-white text-sm capitalize">{node.status}</div>
        </div>
      </div>

      {/* Dates */}
      <div className="bg-gray-700/50 p-3 rounded space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400 flex items-center gap-1">
            <Calendar size={12} />
            Created
          </span>
          <span className="text-white">{new Date(node.createdAt).toLocaleDateString()}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-400 flex items-center gap-1">
            <Clock size={12} />
            Updated
          </span>
          <span className="text-white">{new Date(node.updatedAt).toLocaleDateString()}</span>
        </div>
      </div>

      {/* Personality & Conflicts Section - NEW */}
      <ConflictsSection goalId={node.id} />
      <PersonalitySection userId={node.id} />

      {/* Progress */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-400">Progress</span>
          <span className="text-white">{(node.progress * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div
            className="bg-blue-500 h-2 rounded-full transition-all"
            style={{ width: `${node.progress * 100}%` }}
          />
        </div>
      </div>

      {/* Execution Plan */}
      <div>
        <h3 className="text-gray-400 text-xs uppercase mb-3 flex items-center gap-1">
          <Target size={12} />
          Execution Plan
        </h3>
        <div className="space-y-2">
          {executionPlan.map((step, idx) => (
            <div key={idx} className="bg-gray-700/50 p-2 rounded">
              <div className="flex items-center justify-between mb-1">
                <span className="text-white text-sm font-medium">Step {idx + 1}: {step.step}</span>
                {getStatusBadge(step.status)}
              </div>
              <div className="text-gray-400 text-xs">{step.result}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Sub-goals */}
      {node.childIds && node.childIds.length > 0 && (
        <div>
          <h3 className="text-gray-400 text-xs uppercase mb-2">Sub-goals</h3>
          <div className="bg-gray-700/50 p-2 rounded">
            <div className="text-white text-sm">{node.childIds.length} sub-goal{node.childIds.length > 1 ? 's' : ''}</div>
            <div className="text-gray-400 text-xs mt-1">Click on graph/tree to view details</div>
          </div>
        </div>
      )}

      {/* Metrics */}
      <div className="space-y-2">
        <h3 className="text-gray-400 text-xs uppercase">Metrics</h3>
        <div className="grid grid-cols-3 gap-2">
          <div className="text-center bg-gray-700/50 p-2 rounded">
            <div className="text-blue-400 font-bold text-sm">{(node.feasibility * 100).toFixed(0)}%</div>
            <div className="text-gray-400 text-xs">Feasibility</div>
          </div>
          <div className="text-center bg-gray-700/50 p-2 rounded">
            <div className="text-red-400 font-bold text-sm">{(node.conflictScore * 100).toFixed(0)}%</div>
            <div className="text-gray-400 text-xs">Conflict</div>
          </div>
          <div className="text-center bg-gray-700/50 p-2 rounded">
            <div className="text-yellow-400 font-bold text-sm">{(node.uncertainty * 100).toFixed(0)}%</div>
            <div className="text-gray-400 text-xs">Uncertainty</div>
          </div>
        </div>
      </div>

      {/* Artifacts Section */}
      <div>
        <div
          className="flex items-center justify-between cursor-pointer mb-2"
          onClick={() => setShowArtifacts(!showArtifacts)}
        >
          <h3 className="text-gray-400 text-xs uppercase flex items-center gap-1">
            <Package size={12} />
            Artifacts
            {artifacts.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 bg-blue-900/50 text-blue-400 text-xs rounded-full">
                {artifacts.length}
              </span>
            )}
          </h3>
          <button className="text-gray-400 hover:text-white">
            {showArtifacts ? <span>▼</span> : <span>▶</span>}
          </button>
        </div>

        {showArtifacts && (
          <div className="space-y-2">
            {loadingArtifacts ? (
              <div className="text-gray-400 text-xs">Loading artifacts...</div>
            ) : artifacts.length === 0 ? (
              <div className="text-gray-500 text-xs italic">
                {node.goalType === 'atomic'
                  ? 'No artifacts produced yet'
                  : 'No artifacts - this is a parent goal. Artifacts are produced by sub-goals.'}
              </div>
            ) : (
              <>
                {!node.goalType?.includes('atomic') && artifacts.length > 0 && (
                  <div className="text-blue-400 text-xs mb-2 px-2 py-1 bg-blue-900/30 rounded">
                    ℹ️ Showing {artifacts.length} artifacts from sub-goals
                  </div>
                )}
                {artifacts.map((artifact) => (
                  <ArtifactCard
                    key={artifact.id}
                    artifact={artifact}
                    onOpen={() => setSelectedArtifact(artifact)}
                  />
                ))}
              </>
            )}
          </div>
        )}
      </div>

      {/* Conflicts */}
      {context.conflicts.length > 0 && (
        <div>
          <h3 className="text-gray-400 text-xs uppercase mb-2 flex items-center gap-1">
            <AlertTriangle size={12} />
            Active Conflicts
          </h3>
          <div className="space-y-2">
            {context.conflicts.map((conflict) => (
              <div
                key={conflict.id}
                className={`p-2 rounded border-l-4 ${
                  conflict.severity > 0.7
                    ? 'border-red-500 bg-red-900/20'
                    : 'border-yellow-500 bg-yellow-900/20'
                }`}
              >
                <div className="flex items-center gap-2">
                  <AlertTriangle size={16} className="text-yellow-500" />
                  <span className="text-white text-sm">{conflict.description}</span>
                </div>
                <div className="text-gray-400 text-xs mt-1">
                  Severity: {(conflict.severity * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Suggestions */}
      {context.suggestions.length > 0 && (
        <div>
          <h3 className="text-gray-400 text-xs uppercase mb-2">Suggestions</h3>
          <div className="space-y-2">
            {context.suggestions.map((suggestion, i) => (
              <div key={i} className="p-2 bg-gray-700 rounded text-sm">
                <div className="text-white">{suggestion.action}</div>
                <div className="text-gray-400 text-xs mt-1">{suggestion.reason}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Artifact Modal */}
      {selectedArtifact && (
        <ArtifactModal
          artifact={selectedArtifact}
          onClose={() => setSelectedArtifact(null)}
        />
      )}
    </>
  );
};

// Agent Inspector
const AgentInspector: React.FC<{ node: any; context: InspectorContext }> = ({ node }) => (
  <>
    <div>
      <h3 className="text-gray-400 text-xs uppercase mb-1">Agent</h3>
      <p className="text-white">{node.role || 'Unnamed Agent'}</p>
    </div>
    {node.description && (
      <div>
        <h3 className="text-gray-400 text-xs uppercase mb-1">Description</h3>
        <p className="text-white text-sm">{node.description}</p>
      </div>
    )}
  </>
);

// Skill Inspector
const SkillInspector: React.FC<{ node: any; context: InspectorContext }> = ({ node }) => (
  <>
    <div>
      <h3 className="text-gray-400 text-xs uppercase mb-1">Skill</h3>
      <p className="text-white">{node.name || 'Unnamed Skill'}</p>
    </div>
    {node.description && (
      <div>
        <h3 className="text-gray-400 text-xs uppercase mb-1">Description</h3>
        <p className="text-white text-sm">{node.description}</p>
      </div>
    )}
  </>
);

// Memory Inspector
const MemoryInspector: React.FC<{ node: any; context: InspectorContext }> = ({ node }) => (
  <>
    <div>
      <h3 className="text-gray-400 text-xs uppercase mb-1">Memory</h3>
      <p className="text-white">{node.content || 'No content'}</p>
    </div>
  </>
);

// Artifact Card Component

// NEW: Conflicts Section Component
const ConflictsSection: React.FC<{ goalId: string }> = ({ goalId }) => {
  const [conflicts, setConflicts] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const loadConflicts = async () => {
      setLoading(true);
      try {
        const response = await apiClient.checkConflicts(goalId);
        setConflicts(response.conflict_result);
      } catch (error) {
        console.error('[Conflicts] Failed to load:', error);
      } finally {
        setLoading(false);
      }
    };

    if (expanded) {
      loadConflicts();
    }
  }, [goalId, expanded]);

  if (loading) {
    return (
      <div className="bg-gray-700/30 p-3 rounded">
        <div className="text-gray-400 text-xs">Checking conflicts...</div>
      </div>
    );
  }

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="w-full bg-gray-700/30 hover:bg-gray-700/50 p-3 rounded text-left transition-colors"
      >
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-xs uppercase flex items-center gap-1">
            <AlertTriangle size={12} />
            Check Conflicts
          </span>
          <span className="text-gray-500 text-xs">→</span>
        </div>
      </button>
    );
  }

  return (
    <div className="bg-gray-700/30 p-3 rounded space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-xs uppercase flex items-center gap-1">
          <AlertTriangle size={12} />
          Conflicts
        </span>
        <button
          onClick={() => setExpanded(false)}
          className="text-gray-500 hover:text-gray-300 text-xs"
        >
          ×
        </button>
      </div>

      {conflicts?.has_conflicts ? (
        <div className="space-y-2">
          {conflicts.conflicts.map((conflict: any) => (
            <div key={conflict.id} className="bg-red-900/20 border border-red-900/50 p-2 rounded">
              <div className="flex items-center gap-2 mb-1">
                <AlertTriangle size={12} className="text-red-400" />
                <span className="text-red-400 text-xs font-medium capitalize">
                  {conflict.severity}
                </span>
                <span className="text-gray-400 text-xs capitalize">
                  {conflict.conflict_type}
                </span>
              </div>
              <p className="text-gray-300 text-xs mb-1">{conflict.description}</p>
              <p className="text-gray-400 text-xs italic">{conflict.resolution_suggestion}</p>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-green-400 text-xs flex items-center gap-1">
          <CheckCircle size={12} />
          No conflicts detected
        </div>
      )}
    </div>
  );
};

// NEW: Personality Section Component
const PersonalitySection: React.FC<{ userId: string }> = ({ userId }) => {
  const [memory, setMemory] = useState<any>(null);
  const [snapshots, setSnapshots] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    const loadPersonality = async () => {
      setLoading(true);
      try {
        const [memoryRes, snapshotsRes] = await Promise.all([
          apiClient.getContextualMemory(userId),
          apiClient.getSnapshots(userId, 5)
        ]);
        setMemory(memoryRes.contextual_memory);
        setSnapshots(snapshotsRes);
      } catch (error) {
        console.error('[Personality] Failed to load:', error);
      } finally {
        setLoading(false);
      }
    };

    if (expanded) {
      loadPersonality();
    }
  }, [userId, expanded]);

  if (loading) {
    return (
      <div className="bg-gray-700/30 p-3 rounded">
        <div className="text-gray-400 text-xs">Loading personality data...</div>
      </div>
    );
  }

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="w-full bg-gray-700/30 hover:bg-gray-700/50 p-3 rounded text-left transition-colors"
      >
        <div className="flex items-center justify-between">
          <span className="text-gray-400 text-xs uppercase flex items-center gap-1">
            <Database size={12} />
            Personality Context
          </span>
          <span className="text-gray-500 text-xs">→</span>
        </div>
      </button>
    );
  }

  return (
    <div className="bg-gray-700/30 p-3 rounded space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-xs uppercase flex items-center gap-1">
          <Database size={12} />
          Personality Context
        </span>
        <button
          onClick={() => setExpanded(false)}
          className="text-gray-500 hover:text-gray-300 text-xs"
        >
          ×
        </button>
      </div>

      {memory && (
        <div className="space-y-2">
          <div className="bg-gray-700/50 p-2 rounded">
            <div className="text-gray-400 text-xs mb-1">Emotional Tone</div>
            <div className="text-white text-sm capitalize">{memory.emotional_tone_recent}</div>
          </div>

          <div className="bg-gray-700/50 p-2 rounded">
            <div className="text-gray-400 text-xs mb-1">Recent Goals</div>
            {memory.recent_goals?.length > 0 ? (
              <ul className="text-white text-xs space-y-1">
                {memory.recent_goals.slice(0, 3).map((goal: any, idx: number) => (
                  <li key={idx} className="truncate">• {goal.title}</li>
                ))}
              </ul>
            ) : (
              <div className="text-gray-500 text-xs">No recent goals</div>
            )}
          </div>

          {snapshots?.count > 0 && (
            <div className="bg-gray-700/50 p-2 rounded">
              <div className="text-gray-400 text-xs mb-1">Snapshots</div>
              <div className="text-white text-xs">{snapshots.count} version(s)</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InspectorPanel;
