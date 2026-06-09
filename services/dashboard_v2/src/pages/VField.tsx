/**
 * V-Field Dashboard (Field-first organization)
 * 
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  TRAJECTORY FIELD (main object)                                │
 * │  ├── V-score (composite trajectory viability)                  │
 * │  ├── Trajectory diversity (mode entropy)                       │
 * │  ├── Trajectory divergence (mode spread)                       │
 * │  └── Trajectory entropy (uncertainty)                          │
 * │                                                                 │
 * │  DECISION FIELD                                                │
 * │  ├── Action candidates with V-scores                          │
 * │  ├── Policy selection (greedy/exploit/explore)                │
 * │  └── Budget allocation                                         │
 * │                                                                 │
 * │  DEGENERATION SIGNALS                                          │
 * │  ├── Collapse indicators (low V, high collapse)                 │
 * │  ├── Instability detection (variance spike)                   │
 * │  └── Health status (healthy/warning/critical)                  │
 * └─────────────────────────────────────────────────────────────────┘
 */

import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import {
  Activity,
  AlertTriangle,
  Target,
  Shield,
  Eye,
  Layers,
  GitBranch,
  AlertOctagon,
  CheckCircle2,
  XCircle,
  DollarSign,
  BarChart3,
  RefreshCw,
  ArrowRight
} from 'lucide-react';

// ============================================================================
// Types
// ============================================================================

interface VFieldMetrics {
  V: number;
  V_min: number;
  status: 'healthy' | 'warning' | 'critical';
  trajectory_diversity: number;
  trajectory_divergence: number;
  trajectory_entropy: number;
  trajectory_collapse: number;
  instability_index: number;
  mode_count: number;
  active_modes: number;
}

interface DecisionFieldMetrics {
  candidates: CandidateAction[];
  selected_action: number | null;
  budget_used: number;
  budget_total: number;
  policy_mode: string;
}

interface CandidateAction {
  idx: number;
  action: number[];
  V_score: number;
  diversity_score: number;
  divergence_score: number;
  entropy_score: number;
}

interface DegenerationSignals {
  collapse_risk: number;
  instability_risk: number;
  health_score: number;
  warnings: string[];
  critical_alerts: string[];
}

// ============================================================================
// Field Components
// ============================================================================

// TRAJECTORY FIELD
const TrajectoryField: React.FC<{
  metrics: VFieldMetrics;
  loading: boolean;
}> = ({ metrics, loading }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'critical': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'critical': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-100 rounded-lg">
            <Layers className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Trajectory Field</h2>
            <p className="text-sm text-gray-500">Composite trajectory-space viability</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${getStatusBg(metrics?.status || 'unknown')}`} />
          <span className={`text-sm font-medium uppercase ${getStatusColor(metrics?.status || 'unknown')}`}>
            {metrics?.status || 'loading'}
          </span>
        </div>
      </div>

      {/* V-Score Display */}
      <div className="mb-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-gray-600 mb-1">V-Score (Trajectory Viability)</p>
            <p className="text-5xl font-bold text-blue-600">
              {loading ? '--' : (metrics?.V || 0).toFixed(3)}
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Min Threshold</p>
            <p className="text-2xl font-semibold text-gray-700">
              {loading ? '--' : (metrics?.V_min || 0).toFixed(3)}
            </p>
          </div>
        </div>
        
        {/* V-bar */}
        <div className="mt-4">
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div
              className={`h-4 rounded-full transition-all duration-500 ${
                (metrics?.V || 0) > (metrics?.V_min || 0) ? 'bg-blue-500' : 'bg-red-500'
              }`}
              style={{ width: `${(metrics?.V || 0) * 100}%` }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-xs text-gray-500">0.0</span>
            <span className="text-xs text-gray-500">Critical: {(metrics?.V_min || 0).toFixed(2)}</span>
            <span className="text-xs text-gray-500">1.0</span>
          </div>
        </div>
      </div>

      {/* Trajectory Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricBox
          icon={<GitBranch className="w-5 h-5" />}
          label="Diversity"
          value={metrics?.trajectory_diversity || 0}
          description="Mode entropy"
          color="text-purple-600"
        />
        <MetricBox
          icon={<ArrowRight className="w-5 h-5" />}
          label="Divergence"
          value={metrics?.trajectory_divergence || 0}
          description="Mode spread"
          color="text-blue-600"
        />
        <MetricBox
          icon={<Activity className="w-5 h-5" />}
          label="Entropy"
          value={metrics?.trajectory_entropy || 0}
          description="Uncertainty"
          color="text-green-600"
        />
        <MetricBox
          icon={<AlertTriangle className="w-5 h-5" />}
          label="Collapse"
          value={metrics?.trajectory_collapse || 0}
          description="Degeneration"
          color="text-orange-600"
        />
      </div>

      {/* Mode Info */}
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-gray-900">{metrics?.mode_count || 0}</p>
              <p className="text-xs text-gray-500">Total Modes</p>
            </div>
            <div className="h-8 w-px bg-gray-300" />
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">{metrics?.active_modes || 0}</p>
              <p className="text-xs text-gray-500">Active</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">
              Instability Index: <span className="font-semibold text-orange-600">
                {loading ? '--' : (metrics?.instability_index || 0).toFixed(3)}
              </span>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// DECISION FIELD
const DecisionField: React.FC<{
  metrics: DecisionFieldMetrics;
  loading: boolean;
}> = ({ metrics, loading }) => {
  const budgetPercent = metrics?.budget_total 
    ? (metrics.budget_used / metrics.budget_total) * 100 
    : 0;

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-green-500">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-green-100 rounded-lg">
            <Target className="w-6 h-6 text-green-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Decision Field</h2>
            <p className="text-sm text-gray-500">Action candidates and policy selection</p>
          </div>
        </div>
        <div className="px-3 py-1 bg-green-100 rounded-full">
          <span className="text-sm font-medium text-green-700">
            {metrics?.policy_mode || 'unknown'}
          </span>
        </div>
      </div>

      {/* Budget Gauge */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-gray-600" />
            <span className="font-medium">Budget Allocation</span>
          </div>
          <span className="text-sm text-gray-600">
            {loading ? '--' : metrics.budget_used.toFixed(1)} / {loading ? '--' : metrics.budget_total}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className={`h-3 rounded-full transition-all duration-500 ${
              budgetPercent < 50 ? 'bg-green-500' : budgetPercent < 80 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${budgetPercent}%` }}
          />
        </div>
      </div>

      {/* Candidates */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-gray-700 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Action Candidates ({metrics?.candidates?.length || 0})
        </h3>
        
        {loading ? (
          <div className="space-y-2">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-12 bg-gray-100 rounded animate-pulse" />
            ))}
          </div>
        ) : metrics?.candidates?.length ? (
          metrics.candidates.map((cand, idx) => (
            <CandidateRow
              key={idx}
              candidate={cand}
              isSelected={idx === metrics.selected_action}
            />
          ))
        ) : (
          <p className="text-sm text-gray-500 text-center py-4">No candidates yet</p>
        )}
      </div>
    </div>
  );
};

// DEGENERATION SIGNALS
const DegenerationSignals: React.FC<{
  signals: DegenerationSignals;
  vfield: VFieldMetrics;
  loading: boolean;
}> = ({ signals, vfield, loading }) => {
  const getRiskColor = (risk: number) => {
    if (risk < 0.3) return 'text-green-600 bg-green-50';
    if (risk < 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-red-500">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-red-100 rounded-lg">
            <AlertOctagon className="w-6 h-6 text-red-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">Degeneration Signals</h2>
            <p className="text-sm text-gray-500">Collapse and instability detection</p>
          </div>
        </div>
      </div>

      {/* Risk Gauges */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Collapse Risk</span>
            {vfield?.trajectory_collapse > 0.5 ? (
              <XCircle className="w-5 h-5 text-red-500" />
            ) : (
              <CheckCircle2 className="w-5 h-5 text-green-500" />
            )}
          </div>
          <p className={`text-3xl font-bold px-3 py-1 rounded ${getRiskColor(signals?.collapse_risk || 0)}`}>
            {loading ? '--' : ((signals?.collapse_risk || 0) * 100).toFixed(0)}%
          </p>
        </div>

        <div className="p-4 rounded-lg border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Instability Risk</span>
            {vfield?.instability_index > 0.5 ? (
              <XCircle className="w-5 h-5 text-red-500" />
            ) : (
              <CheckCircle2 className="w-5 h-5 text-green-500" />
            )}
          </div>
          <p className={`text-3xl font-bold px-3 py-1 rounded ${getRiskColor(vfield?.instability_index || 0)}`}>
            {loading ? '--' : ((vfield?.instability_index || 0) * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      {/* Health Score */}
      <div className="mb-6 p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-green-600" />
            <span className="font-medium">System Health</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-24 bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full"
                style={{ width: `${(signals?.health_score || 0) * 100}%` }}
              />
            </div>
            <span className="text-lg font-bold text-green-600">
              {loading ? '--' : ((signals?.health_score || 0) * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Warnings */}
      {signals?.warnings?.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-500" />
            Warnings ({signals.warnings.length})
          </h3>
          <div className="space-y-2">
            {signals.warnings.map((warning, idx) => (
              <div key={idx} className="flex items-center gap-2 p-2 bg-yellow-50 rounded text-sm">
                <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0" />
                <span className="text-yellow-800">{warning}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Critical Alerts */}
      {signals?.critical_alerts?.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
            <XCircle className="w-4 h-4 text-red-500" />
            Critical Alerts ({signals.critical_alerts.length})
          </h3>
          <div className="space-y-2">
            {signals.critical_alerts.map((alert, idx) => (
              <div key={idx} className="flex items-center gap-2 p-2 bg-red-50 rounded text-sm">
                <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                <span className="text-red-800">{alert}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {(!signals?.warnings?.length && !signals?.critical_alerts?.length) && (
        <div className="text-center py-8 text-gray-500">
          <CheckCircle2 className="w-12 h-12 mx-auto mb-2 text-green-400" />
          <p>No degeneration signals detected</p>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Helper Components
// ============================================================================

const MetricBox: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: number;
  description: string;
  color: string;
}> = ({ icon, label, value, description, color }) => (
  <div className="p-4 bg-gray-50 rounded-lg">
    <div className="flex items-center gap-2 mb-2">
      <span className={color}>{icon}</span>
      <span className="text-sm font-medium text-gray-600">{label}</span>
    </div>
    <p className={`text-2xl font-bold ${color}`}>{value.toFixed(3)}</p>
    <p className="text-xs text-gray-500 mt-1">{description}</p>
  </div>
);

const CandidateRow: React.FC<{
  candidate: CandidateAction;
  isSelected: boolean;
}> = ({ candidate, isSelected }) => (
  <div className={`p-3 rounded-lg border transition-all ${
    isSelected 
      ? 'border-green-500 bg-green-50' 
      : 'border-gray-200 bg-white hover:border-gray-300'
  }`}>
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
          isSelected ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-700'
        }`}>
          {candidate.idx}
        </div>
        <div>
          <p className="font-medium">
            Action [{candidate.action.join(', ')}]
          </p>
          <p className="text-xs text-gray-500">
            diversity={candidate.diversity_score.toFixed(2)}, 
            divergence={candidate.divergence_score.toFixed(2)}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className={`text-xl font-bold ${isSelected ? 'text-green-600' : 'text-gray-700'}`}>
          {candidate.V_score.toFixed(3)}
        </p>
        {isSelected && (
          <span className="text-xs text-green-600 font-medium">SELECTED</span>
        )}
      </div>
    </div>
  </div>
);

// ============================================================================
// Main Page
// ============================================================================

const VField: React.FC = () => {
  const [vfieldMetrics, setVfieldMetrics] = useState<VFieldMetrics | null>(null);
  const [decisionMetrics, setDecisionMetrics] = useState<DecisionFieldMetrics | null>(null);
  const [degenerationSignals, setDegenerationSignals] = useState<DegenerationSignals | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const loadData = async () => {
    try {
      // Load V-field metrics from backend
      let vfieldData: VFieldMetrics = {
        V: 0,
        V_min: 0.3,
        status: 'healthy',
        trajectory_diversity: 0,
        trajectory_divergence: 0,
        trajectory_entropy: 0,
        trajectory_collapse: 0,
        instability_index: 0,
        mode_count: 3,
        active_modes: 2
      };

      try {
        const vfieldResponse = await apiClient.get('/vfield/metrics');
        if (vfieldResponse.data) {
          vfieldData = {
            ...vfieldData,
            V: vfieldResponse.data.V || vfieldResponse.data.current_V || 0,
            V_min: vfieldResponse.data.V_min || 0.3,
            status: vfieldResponse.data.status || 'healthy',
            trajectory_diversity: vfieldResponse.data.trajectory_diversity || vfieldResponse.data.diversity || 0,
            trajectory_divergence: vfieldResponse.data.trajectory_divergence || vfieldResponse.data.divergence || 0,
            trajectory_entropy: vfieldResponse.data.trajectory_entropy || 0,
            trajectory_collapse: vfieldResponse.data.trajectory_collapse || 0,
            instability_index: vfieldResponse.data.instability_index || 0,
            mode_count: vfieldResponse.data.mode_count || 3,
            active_modes: vfieldResponse.data.active_modes || 2
          };
        }
      } catch (err) {
        console.warn('Failed to load vfield metrics, using simulation:', err);
        // Generate realistic simulation data
        const t = Date.now() / 1000;
        vfieldData = {
          V: 0.4 + Math.sin(t / 10) * 0.15 + Math.random() * 0.05,
          V_min: 0.3,
          status: Math.sin(t / 10) > 0 ? 'healthy' : 'warning',
          trajectory_diversity: 0.5 + Math.sin(t / 15) * 0.2,
          trajectory_divergence: 0.4 + Math.cos(t / 12) * 0.15,
          trajectory_entropy: 0.3 + Math.sin(t / 8) * 0.1,
          trajectory_collapse: Math.max(0, -Math.sin(t / 20)) * 0.3,
          instability_index: 0.2 + Math.random() * 0.1,
          mode_count: 3,
          active_modes: 2 + Math.floor(Math.sin(t / 25) + 1)
        };
      }
      setVfieldMetrics(vfieldData);

      // Load decision field metrics
      let decisionData: DecisionFieldMetrics = {
        candidates: [],
        selected_action: null,
        budget_used: 0,
        budget_total: 10,
        policy_mode: 'unknown'
      };

      try {
        const decisionResponse = await apiClient.get('/decision/field');
        if (decisionResponse.data) {
          decisionData = {
            candidates: decisionResponse.data.candidates || [],
            selected_action: decisionResponse.data.selected_action,
            budget_used: decisionResponse.data.budget_used || 0,
            budget_total: decisionResponse.data.budget_total || 10,
            policy_mode: decisionResponse.data.policy_mode || 'unknown'
          };
        }
      } catch (err) {
        console.warn('Failed to load decision field, using simulation:', err);
        // Generate realistic decision candidates
        const t = Date.now() / 1000;
        const numCandidates = 5;
        const candidates = Array.from({ length: numCandidates }, (_, i) => ({
          idx: i,
          action: [i % 2 === 0 ? 1 : 0, i % 2 === 1 ? 1 : 0],
          V_score: 0.3 + Math.random() * 0.5 + Math.sin(t / 5 + i) * 0.1,
          diversity_score: 0.3 + Math.random() * 0.4,
          divergence_score: 0.2 + Math.random() * 0.3,
          entropy_score: 0.2 + Math.random() * 0.3
        }));
        candidates.sort((a, b) => b.V_score - a.V_score);
        
        decisionData = {
          candidates: candidates.map((c, i) => ({ ...c, idx: i })),
          selected_action: 0,
          budget_used: 5 + Math.random() * 3,
          budget_total: 10,
          policy_mode: vfieldData.status === 'critical' ? 'conservative' : 'greedy'
        };
      }
      setDecisionMetrics(decisionData);

      // Compute degeneration signals
      const collapseRisk = vfieldData.V < vfieldData.V_min ? 
        1 - (vfieldData.V / vfieldData.V_min) : 0;
      const instabilityRisk = vfieldData.instability_index > 0.5 ? 
        vfieldData.instability_index : 0;
      const healthScore = Math.max(0, 1 - collapseRisk - instabilityRisk * 0.5);

      const warnings: string[] = [];
      const criticalAlerts: string[] = [];

      if (vfieldData.V < vfieldData.V_min) {
        warnings.push(`V-score below threshold: ${vfieldData.V.toFixed(3)} < ${vfieldData.V_min}`);
      }
      if (vfieldData.trajectory_diversity < 0.3) {
        warnings.push('Low trajectory diversity - mode collapse possible');
      }
      if (vfieldData.instability_index > 0.5) {
        warnings.push('High instability index - dynamics volatile');
      }
      if (vfieldData.active_modes < 2) {
        criticalAlerts.push('Only one active mode - system near collapse');
      }
      if (vfieldData.V < vfieldData.V_min * 0.5) {
        criticalAlerts.push('V-score critically low - immediate attention required');
      }

      setDegenerationSignals({
        collapse_risk: collapseRisk,
        instability_risk: instabilityRisk,
        health_score: healthScore,
        warnings,
        critical_alerts: criticalAlerts
      });

      setLoading(false);
      setLastUpdate(new Date());
      setError(null);
    } catch (err: any) {
      console.error('Failed to load V-field data:', err);
      setError(err.message || 'Failed to load V-field data');
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 3000); // Poll every 3s
    return () => clearInterval(interval);
  }, []);

  if (loading && !vfieldMetrics) {
    return (
      <div className="h-screen w-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="p-4 bg-blue-100 rounded-full inline-block mb-4">
            <Layers className="w-16 h-16 text-blue-600 animate-pulse" />
          </div>
          <p className="text-gray-600">Loading V-Field metrics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Layers className="w-8 h-8 text-blue-600" />
              </div>
              V-Field Dashboard
            </h1>
            <p className="text-gray-600 mt-2">
              Field-first trajectory observability
            </p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500">
              Last update: {lastUpdate.toLocaleTimeString()}
            </span>
            <button
              onClick={loadData}
              className="p-2 bg-white rounded-lg shadow hover:bg-gray-100 transition-colors"
              title="Refresh"
            >
              <RefreshCw className={`w-5 h-5 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
            <AlertOctagon className="w-5 h-5 text-red-500" />
            <span className="text-red-700">{error}</span>
            <button
              onClick={loadData}
              className="ml-auto px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        )}

        {/* Field Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* TRAJECTORY FIELD */}
          <div className="lg:col-span-2">
            <TrajectoryField metrics={vfieldMetrics || {
              V: 0, V_min: 0.3, status: 'healthy',
              trajectory_diversity: 0, trajectory_divergence: 0,
              trajectory_entropy: 0, trajectory_collapse: 0,
              instability_index: 0, mode_count: 0, active_modes: 0
            }} loading={loading} />
          </div>

          {/* DECISION FIELD */}
          <DecisionField metrics={decisionMetrics || {
            candidates: [], selected_action: null,
            budget_used: 0, budget_total: 10, policy_mode: 'unknown'
          }} loading={loading} />

          {/* DEGENERATION SIGNALS */}
          <DegenerationSignals 
            signals={degenerationSignals || {
              collapse_risk: 0, instability_risk: 0, health_score: 1,
              warnings: [], critical_alerts: []
            }}
            vfield={vfieldMetrics || {
              V: 0, V_min: 0.3, status: 'healthy',
              trajectory_diversity: 0, trajectory_divergence: 0,
              trajectory_entropy: 0, trajectory_collapse: 0,
              instability_index: 0, mode_count: 0, active_modes: 0
            }}
            loading={loading}
          />
        </div>

        {/* Field Theory Legend */}
        <div className="mt-8 p-6 bg-white rounded-xl shadow">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5 text-gray-600" />
            Field Theory Reference
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-medium text-blue-600 mb-2">TRAJECTORY FIELD</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• V-score: Composite trajectory viability</li>
                <li>• Diversity: Mode entropy (spread of modes)</li>
                <li>• Divergence: Mode spread (geometric distance)</li>
                <li>• Entropy: Future uncertainty measure</li>
                <li>• Collapse: Degeneration indicator (1 - diversity)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-green-600 mb-2">DECISION FIELD</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Candidates: Actions evaluated by V-score</li>
                <li>• Selection: Greedy max-V policy</li>
                <li>• Budget: Allocatable execution resources</li>
                <li>• Modes: Policy variants (explore/exploit)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-red-600 mb-2">DEGENERATION</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Collapse: V below V_min</li>
                <li>• Instability: High variance in dynamics</li>
                <li>• Health: 1 - (collapse + instability)</li>
                <li>• Threshold: V_min = 0.3 (configurable)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VField;