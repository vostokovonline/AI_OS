/**
 * Autonomy Dashboard
 *
 * Displays the autonomous decision-making system
 * Shows decision engine, policy engine, safety constraints
 */

import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { Brain, Shield, AlertTriangle, CheckCircle, Clock, Zap } from 'lucide-react';

interface DecisionState {
  current_mode: string;
  active_policies: string[];
  safety_constraints: {
    ethics: string[];
    budget: number;
    time_horizon: string;
  };
  recent_decisions: Decision[];
  pending_overrides: number;
}

interface Decision {
  id: string;
  node_id: string;
  action: string;
  reasoning: string;
  confidence: number;
  timestamp: string;
  status: 'pending' | 'approved' | 'blocked' | 'executed';
}

const Autonomy: React.FC = () => {
  const [state, setState] = useState<DecisionState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadAutonomyState();
    const interval = setInterval(loadAutonomyState, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  const loadAutonomyState = async () => {
    try {
      // For now, mock data since API endpoints might not be ready
      // TODO: Replace with real API calls when endpoints are implemented
      // const response = await apiClient.get('/autonomy/state');
      // setState(response.data);

      setState({
        current_mode: 'autonomous',
        active_policies: ['ethical_bounds', 'budget_limits', 'safety_first'],
        safety_constraints: {
          ethics: ['no_harm', 'privacy_first', 'transparency'],
          budget: 10000,
          time_horizon: '30d'
        },
        recent_decisions: [
          {
            id: '1',
            node_id: 'goal-123',
            action: 'decompose',
            reasoning: 'Goal complexity requires decomposition',
            confidence: 0.92,
            timestamp: new Date().toISOString(),
            status: 'executed'
          }
        ],
        pending_overrides: 0
      });

      setLoading(false);
    } catch (err) {
      setError('Failed to load autonomy state');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="h-screen w-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Brain className="w-16 h-16 animate-pulse mx-auto mb-4 text-purple-600" />
          <p className="text-gray-600">Loading autonomy system...</p>
        </div>
      </div>
    );
  }

  if (error || !state) {
    return (
      <div className="h-screen w-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center text-red-600">
          <AlertTriangle className="w-16 h-16 mx-auto mb-4" />
          <p className="text-xl font-semibold">Error loading autonomy</p>
          <p className="text-sm mt-2">{error || 'Unknown error'}</p>
        </div>
      </div>
    );
  }

  const getStatusIcon = (status: Decision['status']) => {
    switch (status) {
      case 'executed':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'approved':
        return <Clock className="w-5 h-5 text-blue-600" />;
      case 'blocked':
        return <Shield className="w-5 h-5 text-red-600" />;
      default:
        return <Zap className="w-5 h-5 text-yellow-600" />;
    }
  };

  return (
    <div className="h-screen w-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Brain className="w-8 h-8 text-purple-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Autonomy System</h1>
              <p className="text-sm text-gray-500">Autonomous decision-making and policy engine</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${
              state.current_mode === 'autonomous'
                ? 'bg-green-100 text-green-800'
                : 'bg-gray-100 text-gray-800'
            }`}>
              {state.current_mode}
            </span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-8">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Safety Constraints */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Shield className="w-6 h-6 text-purple-600" />
              <h2 className="text-xl font-semibold text-gray-900">Safety Constraints</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border border-gray-200 rounded p-4">
                <h3 className="font-medium text-gray-700 mb-2">Ethics</h3>
                <div className="space-y-1">
                  {state.safety_constraints.ethics.map((ethic, idx) => (
                    <div key={idx} className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4 text-green-600" />
                      <span className="text-sm text-gray-600">{ethic}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="border border-gray-200 rounded p-4">
                <h3 className="font-medium text-gray-700 mb-2">Budget</h3>
                <p className="text-2xl font-bold text-gray-900">
                  ${state.safety_constraints.budget.toLocaleString()}
                </p>
              </div>
              <div className="border border-gray-200 rounded p-4">
                <h3 className="font-medium text-gray-700 mb-2">Time Horizon</h3>
                <p className="text-2xl font-bold text-gray-900">
                  {state.safety_constraints.time_horizon}
                </p>
              </div>
            </div>
          </div>

          {/* Active Policies */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Zap className="w-6 h-6 text-yellow-600" />
              <h2 className="text-xl font-semibold text-gray-900">Active Policies</h2>
            </div>
            <div className="flex flex-wrap gap-2">
              {state.active_policies.map((policy, idx) => (
                <span
                  key={idx}
                  className="px-4 py-2 bg-purple-100 text-purple-800 rounded-full text-sm font-medium"
                >
                  {policy}
                </span>
              ))}
            </div>
          </div>

          {/* Recent Decisions */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Clock className="w-6 h-6 text-blue-600" />
                <h2 className="text-xl font-semibold text-gray-900">Recent Decisions</h2>
              </div>
              {state.pending_overrides > 0 && (
                <span className="px-3 py-1 bg-red-100 text-red-800 rounded-full text-sm font-medium">
                  {state.pending_overrides} pending overrides
                </span>
              )}
            </div>
            <div className="space-y-3">
              {state.recent_decisions.map((decision) => (
                <div
                  key={decision.id}
                  className="border border-gray-200 rounded p-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-2">
                        <span className="font-medium text-gray-900">{decision.action}</span>
                        <span className="text-sm text-gray-500">on {decision.node_id}</span>
                        {getStatusIcon(decision.status)}
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{decision.reasoning}</p>
                      <div className="flex items-center space-x-4 text-xs text-gray-500">
                        <span>Confidence: {(decision.confidence * 100).toFixed(0)}%</span>
                        <span>{new Date(decision.timestamp).toLocaleString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Autonomy;
