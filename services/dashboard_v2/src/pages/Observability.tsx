import React, { useState, useEffect } from 'react';
import { occpApi, Metric } from '../api/occpApi';

/**
 * Observability Dashboard
 * Real-time metrics and system health monitoring
 */
export const Observability: React.FC = () => {
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      const data = await occpApi.getMetrics();
      setMetrics(data.slice(-50)); // Last 50 metrics
    } catch (err) {
      console.error('Failed to load metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  // Calculate RED metrics
  const calculateREDMetrics = () => {
    if (metrics.length === 0) return null;

    const total = metrics.length;
    const errors = metrics.filter(m => m.status !== 'passed').length;
    const errorRate = total > 0 ? (errors / total) * 100 : 0;

    const durations = metrics.map(m => m.duration_ms);
    const avgDuration = durations.length > 0
      ? durations.reduce((a, b) => a + b, 0) / durations.length
      : 0;

    const sortedDurations = [...durations].sort((a, b) => a - b);
    const p95Duration = sortedDurations[Math.floor(sortedDurations.length * 0.95)] || 0;

    const requestRate = total / 300; // Requests per second (assuming 5min window)

    return {
      requestRate: requestRate.toFixed(2),
      errorRate: errorRate.toFixed(1),
      avgDuration: avgDuration.toFixed(1),
      p95Duration: p95Duration.toFixed(1)
    };
  };

  const redMetrics = calculateREDMetrics();

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Observability Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Real-time metrics and system health monitoring
        </p>
      </div>

      {/* RED Metrics Cards */}
      {redMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Request Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-2">
                  {redMetrics.requestRate}
                  <span className="text-sm font-normal text-gray-500">req/s</span>
                </p>
              </div>
              <div className="text-3xl text-blue-500">📊</div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Error Rate</p>
                <p className="text-2xl font-bold text-gray-900 mt-2">
                  {redMetrics.errorRate}
                  <span className="text-sm font-normal text-gray-500">%</span>
                </p>
              </div>
              <div className="text-3xl text-red-500">❌</div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg Duration</p>
                <p className="text-2xl font-bold text-gray-900 mt-2">
                  {redMetrics.avgDuration}
                  <span className="text-sm font-normal text-gray-500">ms</span>
                </p>
              </div>
              <div className="text-3xl text-yellow-500">⏱️</div>
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">P95 Duration</p>
                <p className="text-2xl font-bold text-gray-900 mt-2">
                  {redMetrics.p95Duration}
                  <span className="text-sm font-normal text-gray-500">ms</span>
                </p>
              </div>
              <div className="text-3xl text-purple-500">📈</div>
            </div>
          </div>
        </div>
      )}

      {/* Metrics Stream */}
      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="bg-gray-50 px-6 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Recent Metrics</h2>
        </div>

        <div className="overflow-x-auto max-h-96 overflow-y-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Skill
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Action
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Duration
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {metrics.slice(-20).map((metric, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {new Date(metric.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {metric.skill_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {metric.action}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                      metric.status === 'passed'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {metric.status.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {metric.duration_ms}ms
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {metrics.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">No metrics available</p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center gap-6 text-sm text-gray-600">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-green-100 border border-green-200 rounded"></span>
          <span>Success</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-red-100 border border-red-200 rounded"></span>
          <span>Error</span>
        </div>
      </div>
    </div>
  );
};
