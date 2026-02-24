/**
 * Autonomy Dashboard
 *
 * Displays the autonomous decision-making system
 * Shows decision engine, policy engine, safety constraints, and system alerts
 */

import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { Brain, Shield, AlertTriangle, CheckCircle, Clock, Zap, Activity } from 'lucide-react';

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

interface AlertSummary {
  total_alerts: number;
  active_alerts: number;
  resolved_alerts: number;
  active_by_type: Array<{
    alert_type: string;
    severity: string;
    count: number;
  }>;
}

const Autonomy: React.FC = () => {
  const [state, setState] = useState<DecisionState | null>(null);
  const [alerts, setAlerts] = useState<AlertSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      // Load alerts summary from backend
      const alertsResponse = await apiClient.get('/alerts/summary');
      setAlerts(alertsResponse.data.summary);

      // For now, use enhanced mock data for autonomy state
      // TODO: Integrate with actual autonomy API when available
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
    } catch (err: any) {
      console.error('Failed to load autonomy state:', err);
      setError(err.message || 'Failed to load autonomy state');
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

  if (error) {
    return (
      <div className="h-screen w-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center text-red-600">
          <AlertTriangle className="w-16 h-16 mx-auto mb-4" />
          <p className="text-xl font-semibold">Error loading autonomy</p>
          <p className="text-sm mt-2">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-gray-50 overflow-auto">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Autonomy System</h1>
          <p className="text-gray-600">Autonomous decision-making and safety monitoring</p>
        </div>

        {/* Alerts Summary */}
        {alerts && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Total Alerts</p>
                  <p className="text-2xl font-bold text-gray-900">{alerts.total_alerts}</p>
                </div>
                <Activity className="w-8 h-8 text-blue-500" />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Active Alerts</p>
                  <p className="text-2xl font-bold text-orange-600">{alerts.active_alerts}</p>
                </div>
                <AlertTriangle className="w-8 h-8 text-orange-500" />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Resolved</p>
                  <p className="text-2xl font-bold text-green-600">{alerts.resolved_alerts}</p>
                </div>
                <CheckCircle className="w-8 h-8 text-green-500" />
              </div>
            </div>
          </div>
        )}

        {/* Decision Engine Status */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-600" />
              Decision Engine
            </h2>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-gray-500 mb-2">Current Mode</p>
                <div className="flex items-center gap-2">
                  <Zap className="w-5 h-5 text-yellow-500" />
                  <span className="text-lg font-semibold capitalize">{state?.current_mode}</span>
                </div>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-2">Pending Overrides</p>
                <div className="flex items-center gap-2">
                  <Clock className="w-5 h-5 text-blue-500" />
                  <span className="text-lg font-semibold">{state?.pending_overrides}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Safety Constraints */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
              <Shield className="w-5 h-5 text-green-600" />
              Safety Constraints
            </h2>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <p className="text-sm text-gray-500 mb-2">Ethics</p>
                <div className="flex flex-wrap gap-2">
                  {state?.safety_constraints.ethics.map((ethic, i) => (
                    <span key={i} className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm">
                      {ethic}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-2">Budget</p>
                <p className="text-lg font-semibold text-gray-900">${state?.safety_constraints.budget.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500 mb-2">Time Horizon</p>
                <p className="text-lg font-semibold text-gray-900">{state?.safety_constraints.time_horizon}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Active Policies */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">Active Policies</h2>
          </div>
          <div className="p-6">
            <div className="flex flex-wrap gap-2">
              {state?.active_policies.map((policy, i) => (
                <span key={i} className="px-4 py-2 bg-purple-100 text-purple-800 rounded-lg font-medium">
                  {policy}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Decisions */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">Recent Decisions</h2>
          </div>
          <div className="p-6">
            {state?.recent_decisions.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No recent decisions</p>
            ) : (
              <div className="space-y-4">
                {state?.recent_decisions.map((decision) => (
                  <div key={decision.id} className="border border-gray-200 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-gray-900">{decision.action}</span>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        decision.status === 'executed' ? 'bg-green-100 text-green-800' :
                        decision.status === 'pending' ? 'bg-yellow-100 text-yellow-800' :
                        decision.status === 'blocked' ? 'bg-red-100 text-red-800' :
                        'bg-blue-100 text-blue-800'
                      }`}>
                        {decision.status}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{decision.reasoning}</p>
                    <div className="flex items-center justify-between text-xs text-gray-500">
                      <span>Confidence: {(decision.confidence * 100).toFixed(0)}%</span>
                      <span>{new Date(decision.timestamp).toLocaleString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Autonomy;
