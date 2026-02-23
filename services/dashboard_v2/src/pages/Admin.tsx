/**
 * Admin Dashboard
 *
 * System administration with:
 * - Pending Approvals (manual goal completion approvals)
 * - Reflections (lessons learned from completed goals)
 * - System Observer (real-time monitoring)
 */

import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { Shield, CheckCircle, XCircle, Eye, Activity, Clock, TrendingUp } from 'lucide-react';

interface AdminStats {
  pending_approvals: number;
  completed_goals: number;
  active_goals: number;
  system_health: number;
}

interface PendingApproval {
  id: string;
  goal_id: string;
  goal_title: string;
  goal_type: string;
  status: string;
  progress: number;
  created_at: string;
  requires_approval: boolean;
}

interface Reflection {
  id: string;
  goal_id: string;
  goal_title: string;
  outcome: 'success' | 'failure';
  lessons_learned: string[];
  created_at: string;
}

const Admin: React.FC = () => {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<PendingApproval[]>([]);
  const [reflections, setReflections] = useState<Reflection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'approvals' | 'reflections' | 'observer'>('approvals');

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      // Load goals for pending approvals
      const goalsResponse = await apiClient.get('/goals/list');
      const allGoals = goalsResponse.data.goals || [];

      // Filter for manual goals that might need approval
      const manualGoals = allGoals.filter((g: any) =>
        g.completion_mode === 'manual' &&
        (g.status === 'active' || g.status === 'done')
      );

      setPendingApprovals(manualGoals.map((g: any) => ({
        id: g.id,
        goal_id: g.id,
        goal_title: g.title,
        goal_type: g.goal_type,
        status: g.status,
        progress: g.progress,
        created_at: g.created_at,
        requires_approval: g.completion_mode === 'manual' && g.status === 'done'
      })));

      // Calculate stats
      const activeCount = allGoals.filter((g: any) => g.status === 'active').length;
      const doneCount = allGoals.filter((g: any) => g.status === 'done').length;
      const pendingApprovalCount = manualGoals.filter((g: any) =>
        g.completion_mode === 'manual' && g.status === 'done'
      ).length;

      setStats({
        pending_approvals: pendingApprovalCount,
        completed_goals: doneCount,
        active_goals: activeCount,
        system_health: 95 // Mock health metric
      });

      // For reflections, we'd need a dedicated API endpoint
      // For now, show empty list
      setReflections([]);

      setLoading(false);
    } catch (err: any) {
      console.error('Failed to load admin data:', err);
      setError(err.message || 'Failed to load admin data');
      setLoading(false);
    }
  };

  const handleApprove = async (goalId: string) => {
    try {
      await apiClient.post(`/goals/${goalId}/approve_completion`, {
        approved_by: 'admin',
        authority_level: 4,
        comment: 'Approved via admin dashboard'
      });
      // Reload data
      loadData();
    } catch (err: any) {
      console.error('Failed to approve goal:', err);
      alert(`Failed to approve: ${err.response?.data?.error?.message || err.message}`);
    }
  };

  const handleReject = async (goalId: string) => {
    try {
      // For rejection, we might need to transition to a different state
      await apiClient.post(`/goals/${goalId}/mutate`, {
        mutation_type: 'weaken',
        reason: 'Rejected via admin dashboard'
      });
      loadData();
    } catch (err: any) {
      console.error('Failed to reject goal:', err);
      alert(`Failed to reject: ${err.message}`);
    }
  };

  if (loading) {
    return (
      <div className="h-screen w-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Shield className="w-16 h-16 animate-pulse mx-auto mb-4 text-cyan-600" />
          <p className="text-gray-600">Loading admin panel...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen w-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center text-red-600">
          <XCircle className="w-16 h-16 mx-auto mb-4" />
          <p className="text-xl font-semibold">Error loading admin</p>
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
          <h1 className="text-3xl font-bold text-gray-900 mb-2">System Administration</h1>
          <p className="text-gray-600">Goal approvals, reflections, and system monitoring</p>
        </div>

        {/* Stats Bar */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Pending Approvals</p>
                  <p className="text-2xl font-bold text-orange-600">{stats.pending_approvals}</p>
                </div>
                <Clock className="w-8 h-8 text-orange-500" />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Completed</p>
                  <p className="text-2xl font-bold text-green-600">{stats.completed_goals}</p>
                </div>
                <CheckCircle className="w-8 h-8 text-green-500" />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Active</p>
                  <p className="text-2xl font-bold text-blue-600">{stats.active_goals}</p>
                </div>
                <Activity className="w-8 h-8 text-blue-500" />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">System Health</p>
                  <p className="text-2xl font-bold text-purple-600">{stats.system_health}%</p>
                </div>
                <TrendingUp className="w-8 h-8 text-purple-500" />
              </div>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="bg-white rounded-lg shadow mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('approvals')}
                className={`px-6 py-4 border-b-2 font-medium text-sm ${
                  activeTab === 'approvals'
                    ? 'border-cyan-500 text-cyan-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Pending Approvals
              </button>
              <button
                onClick={() => setActiveTab('reflections')}
                className={`px-6 py-4 border-b-2 font-medium text-sm ${
                  activeTab === 'reflections'
                    ? 'border-cyan-500 text-cyan-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Reflections
              </button>
              <button
                onClick={() => setActiveTab('observer')}
                className={`px-6 py-4 border-b-2 font-medium text-sm ${
                  activeTab === 'observer'
                    ? 'border-cyan-500 text-cyan-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                System Observer
              </button>
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'approvals' && (
              <div>
                {pendingApprovals.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No pending approvals</p>
                ) : (
                  <div className="space-y-4">
                    {pendingApprovals.map((approval) => (
                      <div key={approval.id} className="border border-gray-200 rounded-lg p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex-1">
                            <h3 className="font-semibold text-gray-900 mb-1">{approval.goal_title}</h3>
                            <div className="flex items-center gap-4 text-sm text-gray-600">
                              <span>Type: {approval.goal_type}</span>
                              <span>Status: {approval.status}</span>
                              <span>Progress: {(approval.progress * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                          {approval.requires_approval && (
                            <span className="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-xs font-medium">
                              Requires Approval
                            </span>
                          )}
                        </div>
                        {approval.requires_approval && (
                          <div className="flex gap-2 mt-4">
                            <button
                              onClick={() => handleApprove(approval.goal_id)}
                              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                            >
                              <CheckCircle className="w-4 h-4" />
                              Approve
                            </button>
                            <button
                              onClick={() => handleReject(approval.goal_id)}
                              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                            >
                              <XCircle className="w-4 h-4" />
                              Reject
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'reflections' && (
              <div>
                {reflections.length === 0 ? (
                  <div className="text-center py-8">
                    <Eye className="w-16 h-16 mx-auto mb-4 text-gray-400" />
                    <p className="text-gray-500">No reflections yet</p>
                    <p className="text-sm text-gray-400 mt-2">Reflections appear here after goals are completed</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {reflections.map((reflection) => (
                      <div key={reflection.id} className="border border-gray-200 rounded-lg p-4">
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <h3 className="font-semibold text-gray-900 mb-1">{reflection.goal_title}</h3>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              reflection.outcome === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                            }`}>
                              {reflection.outcome}
                            </span>
                          </div>
                          <span className="text-sm text-gray-500">
                            {new Date(reflection.created_at).toLocaleString()}
                          </span>
                        </div>
                        <div className="mt-3">
                          <p className="text-sm font-medium text-gray-700 mb-2">Lessons Learned:</p>
                          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                            {reflection.lessons_learned.map((lesson, i) => (
                              <li key={i}>{lesson}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {activeTab === 'observer' && (
              <div>
                <div className="text-center py-8">
                  <Activity className="w-16 h-16 mx-auto mb-4 text-cyan-600" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">System Observer</h3>
                  <p className="text-gray-500 mb-4">Real-time system monitoring</p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500">CPU Usage</p>
                      <p className="text-2xl font-bold text-gray-900">--%</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500">Memory</p>
                      <p className="text-2xl font-bold text-gray-900">--%</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500">Active Tasks</p>
                      <p className="text-2xl font-bold text-gray-900">--</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400 mt-6">
                    System observer metrics coming soon
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Admin;
