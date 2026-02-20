/**
 * Screen 3: Alert & IRL Monitor
 *
 * Question: Where is the system asking to be noticed?
 * Frequency: Continuous
 *
 * READ-ONLY - No action buttons
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, Eye, Clock, TrendingUp, Shield } from 'lucide-react';
import { apiClient } from '../../../api/client';

interface Alert {
  id: string;
  alert_type: string;
  severity: 'low' | 'medium' | 'high';
  trigger_data: any;
  explanation: string;
  context: any;
  created_at: string;
}

interface IRLStatus {
  invariants_status: string;
  health_status: string;
  candidates: {
    total: number;
    approved: number;
    pending: number;
  };
  summary: {
    invariant_violations: number;
    critical_failure_modes: number;
  };
}

export function AlertIRLMonitor() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [irlStatus, setIRLStatus] = useState<IRLStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [alertsResp, irlResp] = await Promise.all([
        apiClient.get('/alerts?resolved=false&limit=20'),
        apiClient.get('/irl/status')
      ]);

      setAlerts(alertsResp.alerts || []);
      setIRLStatus(irlResp);
    } catch (error) {
      console.error('Failed to load alerts/IRL status:', error);
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'text-red-400 bg-red-900/10 border-red-900/50';
      case 'medium': return 'text-yellow-400 bg-yellow-900/10 border-yellow-900/50';
      case 'low': return 'text-blue-400 bg-blue-900/10 border-blue-900/50';
      default: return 'text-gray-400 bg-gray-900/10 border-gray-700';
    }
  };

  const getTimeSince = (timestamp: string) => {
    const now = new Date();
    const then = new Date(timestamp);
    const diff = Math.floor((now.getTime() - then.getTime()) / 1000 / 60);
    if (diff < 60) return `${diff}m ago`;
    if (diff < 1440) return `${Math.floor(diff / 60)}h ago`;
    return `${Math.floor(diff / 1440)}d ago`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Loading system signals...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* IRL Status Block */}
      {irlStatus && (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
          <div className="flex items-center gap-2 mb-4">
            <Shield className="text-blue-400" size={20} />
            <h3 className="text-lg font-semibold text-gray-100">Архитектурный Статус IRL</h3>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-950 border border-gray-800 rounded p-3">
              <div className="text-xs text-gray-500 mb-1">Инварианты</div>
              <div className="text-lg font-mono text-gray-200">{irlStatus.invariants_status}</div>
            </div>

            <div className="bg-gray-950 border border-gray-800 rounded p-3">
              <div className="text-xs text-gray-500 mb-1">Здоровье</div>
              <div className="text-lg font-mono text-gray-200">{irlStatus.health_status}</div>
            </div>

            <div className="bg-gray-950 border border-gray-800 rounded p-3">
              <div className="text-xs text-gray-500 mb-1">Нарушения</div>
              <div className="text-lg font-mono text-gray-200">{irlStatus.summary.invariant_violations}</div>
            </div>

            <div className="bg-gray-950 border border-gray-800 rounded p-3">
              <div className="text-xs text-gray-500 mb-1">Критичные FM</div>
              <div className="text-lg font-mono text-gray-200">{irlStatus.summary.critical_failure_modes}</div>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-gray-800">
            <div className="text-xs text-gray-600 mb-2">КАНДИДАТЫ ИНТЕРВЕНЦИЙ</div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-xs text-gray-500">Всего</div>
                <div className="text-sm font-mono text-gray-300">{irlStatus.candidates.total}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Одобрено</div>
                <div className="text-sm font-mono text-gray-300">{irlStatus.candidates.approved}</div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Ожидает</div>
                <div className="text-sm font-mono text-gray-300">{irlStatus.candidates.pending}</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Active Alerts Block */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="text-yellow-400" size={20} />
            <h3 className="text-lg font-semibold text-gray-100">Активные Системные Тревоги</h3>
          </div>
          <div className="text-xs text-gray-500">
            {alerts.length} активно
          </div>
        </div>

        {alerts.length === 0 ? (
          <div className="text-center py-12">
            <Eye className="text-gray-700 mx-auto mb-2" size={32} />
            <div className="text-sm text-gray-600">Нет активных тревог</div>
            <div className="text-xs text-gray-700 mt-1">Система не подаёт сигналов</div>
          </div>
        ) : (
          <div className="space-y-3">
            {alerts.map(alert => (
              <div
                key={alert.id}
                className="bg-gray-950 border border-gray-800 rounded-lg p-4"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`px-2 py-0.5 rounded text-xs font-mono border ${getSeverityColor(alert.severity)}`}>
                        {alert.severity.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500 font-mono">{alert.alert_type}</span>
                    </div>

                    <div className="text-sm text-gray-300 mb-2">
                      {alert.explanation}
                    </div>

                    {alert.trigger_data && Object.keys(alert.trigger_data).length > 0 && (
                      <details className="mt-2">
                        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
                          Данные триггера
                        </summary>
                        <pre className="mt-2 text-xs text-gray-600 bg-gray-900 p-2 rounded overflow-x-auto">
                          {JSON.stringify(alert.trigger_data, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>

                  <div className="flex items-center gap-1 text-xs text-gray-500">
                    <Clock size={12} />
                    {getTimeSince(alert.created_at)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Alert Fatigue Indicator */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="text-gray-400" size={18} />
          <h4 className="text-sm font-semibold text-gray-300">Индикатор Усталости от Тревог</h4>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-950 border border-gray-800 rounded p-3">
            <div className="text-xs text-gray-500 mb-1">Активно сейчас</div>
            <div className="text-2xl font-mono text-gray-200">{alerts.length}</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-3">
            <div className="text-xs text-gray-500 mb-1">Повторяется (7д)</div>
            <div className="text-2xl font-mono text-gray-200">-</div>
            <div className="text-xs text-gray-600 mt-1">Не отслеживается</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-3">
            <div className="text-xs text-gray-500 mb-1">Тренд (7д)</div>
            <div className="text-2xl font-mono text-gray-200">-</div>
            <div className="text-xs text-gray-600 mt-1">Не отслеживается</div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-gray-950/50 border border-gray-800 rounded text-xs text-gray-500">
          <div className="font-mono mb-1">ЗАМЕЧАНИЕ ПО НАБЛЮДЕНИЮ:</div>
          Если количество тревог растёт без разрешения → возможная усталость
          <br />
          Если тревоги повторяются без вмешательства → деградация сигналов
        </div>
      </div>

      {/* READ-ONLY Notice */}
      <div className="bg-blue-900/5 border border-blue-900/50 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <Eye className="text-blue-400 mt-0.5" size={16} />
          <div className="text-xs text-blue-400">
            <div className="font-semibold mb-1">РЕЖИМ ТОЛЬКО ЧТЕНИЕ</div>
            <div className="text-blue-400/70">
              Этот экран показывает системные сигналы. Кнопки действий отсутствуют намеренно.
              Все интервенции требуют отдельного осознанного решения.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
