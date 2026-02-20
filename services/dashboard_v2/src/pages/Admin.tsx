/**
 * Admin Dashboard
 *
 * Displays system administration features
 * Observer: System state monitoring
 * Reflection: Goal completion approval and reflection management
 */

import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { Eye, CheckCircle, XCircle, Clock, AlertCircle, FileText } from 'lucide-react';

interface AdminStats {
  pending_approvals: number;
  completed_today: number;
  active_goals: number;
  system_health: 'healthy' | 'degraded' | 'critical';
}

interface PendingApproval {
  id: string;
  goal_id: string;
  title: string;
  completion_mode: 'manual' | 'aggregate';
  progress: number;
  created_at: string;
  artifacts_count: number;
}

interface Reflection {
  id: string;
  goal_id: string;
  outcome: 'success' | 'failure';
  lessons_learned: string[];
  created_at: string;
}

const Admin: React.FC = () => {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [pendingApprovals, setPendingApprovals] = useState<PendingApproval[]>([]);
  const [reflections, setReflections] = useState<Reflection[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'approvals' | 'reflections' | 'observer'>('approvals');

  useEffect(() => {
    loadAdminData();
    const interval = setInterval(loadAdminData, 10000); // Poll every 10s
    return () => clearInterval(interval);
  }, [activeTab]);

  const loadAdminData = async () => {
    try {
      // TODO: Replace with real API calls
      // const statsRes = await apiClient.get('/admin/stats');
      // const approvalsRes = await apiClient.get('/admin/approvals/pending');
      // const reflectionsRes = await apiClient.get('/admin/reflections');

      // Mock data for now
      setStats({
        pending_approvals: 3,
        completed_today: 7,
        active_goals: 12,
        system_health: 'healthy'
      });

      if (activeTab === 'approvals') {
        setPendingApprovals([
          {
            id: '1',
            goal_id: 'goal-123',
            title: 'Implement new feature',
            completion_mode: 'manual',
            progress: 1.0,
            created_at: new Date().toISOString(),
            artifacts_count: 3
          },
          {
            id: '2',
            goal_id: 'goal-456',
            title: 'Fix critical bug',
            completion_mode: 'manual',
            progress: 1.0,
            created_at: new Date().toISOString(),
            artifacts_count: 2
          }
        ]);
      } else if (activeTab === 'reflections') {
        setReflections([
          {
            id: '1',
            goal_id: 'goal-789',
            outcome: 'success',
            lessons_learned: ['Start with tests', 'Break into smaller tasks'],
            created_at: new Date().toISOString()
          }
        ]);
      }

      setLoading(false);
    } catch (err) {
      console.error('Failed to load admin data:', err);
      setLoading(false);
    }
  };

  const handleApprove = async (approvalId: string) => {
    try {
      // TODO: Implement real approval
      // await apiClient.post(`/admin/approvals/${approvalId}/approve`, {
      //   approved_by: 'admin',
      //   authority_level: 4
      // });
      console.log('Approving:', approvalId);
      loadAdminData();
    } catch (err) {
      console.error('Failed to approve:', err);
    }
  };

  const handleReject = async (approvalId: string) => {
    try {
      // TODO: Implement real rejection
      // await apiClient.post(`/admin/approvals/${approvalId}/reject`, {
      //   decided_by: 'admin'
      // });
      console.log('Rejecting:', approvalId);
      loadAdminData();
    } catch (err) {
      console.error('Failed to reject:', err);
    }
  };

  const getHealthColor = (health: AdminStats['system_health']) => {
    switch (health) {
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'degraded':
        return 'text-yellow-600 bg-yellow-100';
      case 'critical':
        return 'text-red-600 bg-red-100';
    }
  };

  if (loading) {
    return (
      <div className="h-screen w-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Eye className="w-16 h-16 animate-pulse mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600">Loading admin dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Eye className="w-8 h-8 text-blue-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
              <p className="text-sm text-gray-500">System administration and approvals</p>
            </div>
          </div>
          {stats && (
            <div className="flex items-center space-x-2">
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${getHealthColor(stats.system_health)}`}>
                {stats.system_health}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Stats Bar */}
      {stats && (
        <div className="bg-white border-b border-gray-200 px-8 py-4">
          <div className="grid grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">{stats.pending_approvals}</p>
              <p className="text-sm text-gray-500">Pending Approvals</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">{stats.completed_today}</p>
              <p className="text-sm text-gray-500">Completed Today</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600">{stats.active_goals}</p>
              <p className="text-sm text-gray-500">Active Goals</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-gray-600">{stats.system_health}</p>
              <p className="text-sm text-gray-500">System Health</p>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="bg-white border-b border-gray-200 px-8">
        <div className="flex space-x-8">
          <button
            onClick={() => setActiveTab('approvals')}
            className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'approvals'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Pending Approvals
          </button>
          <button
            onClick={() => setActiveTab('reflections')}
            className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'reflections'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            Reflections
          </button>
          <button
            onClick={() => setActiveTab('observer')}
            className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
              activeTab === 'observer'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            System Observer
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-8">
        <div className="max-w-7xl mx-auto">
          {activeTab === 'approvals' && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Manual Goal Approvals</h2>
              {pendingApprovals.length === 0 ? (
                <div className="bg-white rounded-lg shadow p-8 text-center">
                  <CheckCircle className="w-12 h-12 text-green-600 mx-auto mb-4" />
                  <p className="text-gray-600">No pending approvals</p>
                </div>
              ) : (
                pendingApprovals.map((approval) => (
                  <div key={approval.id} className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <h3 className="text-lg font-semibold text-gray-900">{approval.title}</h3>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            approval.completion_mode === 'manual'
                              ? 'bg-purple-100 text-purple-800'
                              : 'bg-blue-100 text-blue-800'
                          }`}>
                            {approval.completion_mode}
                          </span>
                        </div>
                        <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                          <span>Goal ID: {approval.goal_id}</span>
                          <span>Progress: {(approval.progress * 100).toFixed(0)}%</span>
                          <span>Artifacts: {approval.artifacts_count}</span>
                        </div>
                        <p className="text-xs text-gray-500">
                          Created: {new Date(approval.created_at).toLocaleString()}
                        </p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => handleApprove(approval.id)}
                          className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors flex items-center space-x-1"
                        >
                          <CheckCircle className="w-4 h-4" />
                          <span>Approve</span>
                        </button>
                        <button
                          onClick={() => handleReject(approval.id)}
                          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors flex items-center space-x-1"
                        >
                          <XCircle className="w-4 h-4" />
                          <span>Reject</span>
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === 'reflections' && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Reflections</h2>
              {reflections.length === 0 ? (
                <div className="bg-white rounded-lg shadow p-8 text-center">
                  <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">No reflections yet</p>
                </div>
              ) : (
                reflections.map((reflection) => (
                  <div key={reflection.id} className="bg-white rounded-lg shadow p-6">
                    <div className="flex items-start space-x-3">
                      {reflection.outcome === 'success' ? (
                        <CheckCircle className="w-6 h-6 text-green-600 mt-1" />
                      ) : (
                        <AlertCircle className="w-6 h-6 text-red-600 mt-1" />
                      )}
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">
                          {reflection.goal_id}
                        </h3>
                        <div className="mb-3">
                          <p className="text-sm font-medium text-gray-700 mb-1">Lessons Learned:</p>
                          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                            {reflection.lessons_learned.map((lesson, idx) => (
                              <li key={idx}>{lesson}</li>
                            ))}
                          </ul>
                        </div>
                        <p className="text-xs text-gray-500">
                          {new Date(reflection.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === 'observer' && (
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">System Observer</h2>
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center space-x-2 mb-4">
                  <Clock className="w-5 h-5 text-blue-600" />
                  <h3 className="font-semibold text-gray-900">Real-time Monitoring</h3>
                </div>
                <p className="text-gray-600 mb-4">
                  System state observer will show live metrics, resource usage, and performance indicators.
                </p>
                <div className="grid grid-cols-3 gap-4">
                  <div className="border border-gray-200 rounded p-4">
                    <p className="text-sm text-gray-500">CPU Usage</p>
                    <p className="text-2xl font-bold text-gray-900">--</p>
                  </div>
                  <div className="border border-gray-200 rounded p-4">
                    <p className="text-sm text-gray-500">Memory</p>
                    <p className="text-2xl font-bold text-gray-900">--</p>
                  </div>
                  <div className="border border-gray-200 rounded p-4">
                    <p className="text-sm text-gray-500">Active Tasks</p>
                    <p className="text-2xl font-bold text-gray-900">--</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Admin;
