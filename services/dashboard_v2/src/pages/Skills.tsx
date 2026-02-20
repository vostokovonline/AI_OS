import React, { useState, useEffect } from 'react';
import { occpApi, Skill } from '../api/occpApi';

/**
 * Skills Management Page
 * Display all deployed skills with their capabilities and metrics
 */
export const Skills: React.FC = () => {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSkills();
  }, []);

  const loadSkills = async () => {
    try {
      setLoading(true);
      const data = await occpApi.getSkills();
      setSkills(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load skills');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
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
        <h1 className="text-3xl font-bold text-gray-900">Skills Management</h1>
        <p className="text-gray-600 mt-2">
          Manage and monitor all deployed OCCP skills
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {skills.map((skill) => (
          <div
            key={`${skill.skill_id}-${skill.version}`}
            className="bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow p-6"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-xl font-semibold text-gray-900">
                  {skill.skill_id}
                </h3>
                <p className="text-sm text-gray-500">v{skill.version}</p>
              </div>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                Active
              </span>
            </div>

            <p className="text-gray-600 mb-4">{skill.description}</p>

            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Capabilities</h4>
              <div className="flex flex-wrap gap-2">
                {skill.capabilities.map((capability) => (
                  <span
                    key={capability}
                    className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-50 text-blue-700"
                  >
                    {capability}
                  </span>
                ))}
              </div>
            </div>

            <div className="border-t border-gray-200 pt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Contracts</h4>
              <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Max Time:</span>
                  <span className="font-mono">{skill.contracts.max_execution_time_seconds}s</span>
                </div>
                <div className="flex justify-between">
                  <span>Max Memory:</span>
                  <span className="font-mono">{skill.contracts.max_memory_mb}MB</span>
                </div>
                <div className="flex justify-between">
                  <span>Max Tokens:</span>
                  <span className="font-mono">{skill.contracts.max_tokens}</span>
                </div>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-xs text-gray-500">
                Author: {skill.author}
              </p>
            </div>
          </div>
        ))}
      </div>

      {skills.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No skills deployed yet</p>
        </div>
      )}
    </div>
  );
};
