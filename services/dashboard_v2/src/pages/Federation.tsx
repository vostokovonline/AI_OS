import React, { useState, useEffect } from 'react';
import { occpApi, Node } from '../api/occpApi';

/**
 * Federation Page
 * Display multi-node topology and skill replication
 */
export const Federation: React.FC = () => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadNodes();
  }, []);

  const loadNodes = async () => {
    try {
      const data = await occpApi.getFederationNodes();
      setNodes(data);
    } catch (err) {
      console.error('Failed to load nodes:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-100 text-green-800 border-green-200';
      case 'degraded': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'inactive': return 'bg-gray-100 text-gray-800 border-gray-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRoleBadge = (role: string) => {
    return role === 'primary'
      ? 'bg-purple-100 text-purple-800 border-purple-200'
      : 'bg-blue-100 text-blue-800 border-blue-200';
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Federation</h1>
        <p className="text-gray-600 mt-2">
          Multi-node skill replication and topology
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Nodes</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">{nodes.length}</p>
            </div>
            <div className="text-4xl">🌐</div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Nodes</p>
              <p className="text-3xl font-bold text-green-600 mt-2">
                {nodes.filter(n => n.status === 'active').length}
              </p>
            </div>
            <div className="text-4xl">✅</div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Replicated Skills</p>
              <p className="text-3xl font-bold text-blue-600 mt-2">
                {nodes.reduce((sum, n) => sum + n.skills_count, 0)}
              </p>
            </div>
            <div className="text-4xl">📦</div>
          </div>
        </div>
      </div>

      {/* Topology Visualization */}
      <div className="bg-white border border-gray-200 rounded-lg shadow-sm mb-6">
        <div className="bg-gray-50 px-6 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Federation Topology</h2>
        </div>

        <div className="p-6">
          <div className="flex flex-col items-center gap-8">
            {nodes.map((node, idx) => (
              <div key={node.node_id} className="flex items-center gap-8">
                {/* Node */}
                <div className={`flex items-center gap-4 p-4 border-2 rounded-lg ${getStatusColor(node.status)}`}>
                  <div className="text-2xl">🖥️</div>
                  <div>
                    <p className="font-semibold text-gray-900">{node.node_id}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className={`text-xs px-2 py-0.5 rounded border ${getRoleBadge(node.role)}`}>
                        {node.role.toUpperCase()}
                      </span>
                      <span className={`text-xs px-2 py-0.5 rounded border ${getStatusColor(node.status)}`}>
                        {node.status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Skills */}
                <div className="flex-1">
                  <p className="text-sm text-gray-600 mb-2">
                    {node.skills_count} skill{node.skills_count !== 1 ? 's' : ''}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {/* Show skill indicators */}
                    <div className="w-24 h-6 bg-blue-100 border border-blue-300 rounded flex items-center justify-center text-xs text-blue-700">
                      hello_world
                    </div>
                    <div className="w-24 h-6 bg-green-100 border border-green-300 rounded flex items-center justify-center text-xs text-green-700">
                      calculator
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Nodes Table */}
      <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <div className="bg-gray-50 px-6 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Node Details</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Node
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Role
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Skills
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Last Seen
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {nodes.map((node) => (
                <tr key={node.node_id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {node.node_id}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-xs px-2 py-1 rounded border ${getRoleBadge(node.role)}`}>
                      {node.role}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`text-xs px-2 py-1 rounded border ${getStatusColor(node.status)}`}>
                      {node.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {node.skills_count}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                    {new Date(node.last_seen).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
