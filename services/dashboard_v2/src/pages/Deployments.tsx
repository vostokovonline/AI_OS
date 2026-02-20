import React, { useState, useEffect } from 'react';
import { occpApi, Deployment } from '../api/occpApi';

/**
 * Deployments Page
 * Display all deployments with status and traffic information
 */
export const Deployments: React.FC = () => {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDeployments();
  }, []);

  const loadDeployments = async () => {
    try {
      setLoading(true);
      const data = await occpApi.getDeployments();
      setDeployments(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load deployments');
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const styles = {
      canary: 'bg-yellow-100 text-yellow-800',
      stable: 'bg-green-100 text-green-800',
      rolled_back: 'bg-red-100 text-red-800',
      green: 'bg-blue-100 text-blue-800'
    };

    const style = styles[status as keyof typeof styles] || 'bg-gray-100 text-gray-800';

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${style}`}>
        {status.toUpperCase()}
      </span>
    );
  };

  const getTrafficBar = (percentage: number) => {
    const width = `${percentage}%`;
    let color = 'bg-green-500';

    if (percentage < 10) color = 'bg-yellow-500';
    else if (percentage < 50) color = 'bg-blue-500';
    else if (percentage === 100) color = 'bg-green-500';

    return (
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`${color} h-2 rounded-full transition-all duration-300`}
          style={{ width }}
        ></div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <div className="h-12 bg-gray-100 border-b"></div>
            <div className="p-4 space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-16 bg-gray-200 rounded"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">Error: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Deployments</h1>
        <p className="text-gray-600 mt-2">
          Monitor all skill deployments and traffic distribution
        </p>
      </div>

      <div className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden">
        <div className="bg-gray-50 px-6 py-3 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Active Deployments</h2>
            <span className="text-sm text-gray-600">{deployments.length} total</span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Deployment
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Traffic
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Updated
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {deployments.map((deployment) => (
                <tr key={deployment.deployment_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {deployment.skill_id}:{deployment.version}
                    </div>
                    <div className="text-xs text-gray-500 font-mono mt-1">
                      {deployment.deployment_id.slice(0, 8)}...
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {getStatusBadge(deployment.status)}
                  </td>
                  <td className="px-6 py-4">
                    <div className="w-32">
                      <div className="flex items-center justify-between text-xs mb-1">
                        <span className="text-gray-600">Traffic</span>
                        <span className="font-medium text-gray-900">{deployment.traffic_percentage}%</span>
                      </div>
                      {getTrafficBar(deployment.traffic_percentage)}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {new Date(deployment.created_at).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {new Date(deployment.updated_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {deployments.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-500">No deployments found</p>
          </div>
        )}
      </div>
    </div>
  );
};
