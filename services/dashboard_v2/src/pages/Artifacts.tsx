import React, { useState, useEffect } from 'react';
import { FileText, Eye, CheckCircle, XCircle, Clock, Download, ChevronDown, ChevronUp } from 'lucide-react';
import { apiClient } from '../api/client';

interface Artifact {
  id: string;
  type: string;
  goal_id: string;
  skill_name: string;
  content_kind: string;
  content_location: string;
  description: string | null;
  verification_status: string;
  created_at: string;
  file_size?: number;
}

interface ArtifactContent {
  status: string;
  artifact_id: string;
  file_path: string;
  file_content?: string;
  file_size?: number;
  message?: string;
}

export const Artifacts: React.FC = () => {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);
  const [artifactContent, setArtifactContent] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(false);
  const [contentError, setContentError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'passed' | 'failed' | 'pending'>('all');
  const [expandedArtifact, setExpandedArtifact] = useState<string | null>(null);

  useEffect(() => {
    loadArtifacts();
  }, [filter]);

  const loadArtifacts = async () => {
    try {
      setLoading(true);
      const status = filter === 'all' ? undefined : filter;
      const response = await apiClient.get<{ artifacts: Artifact[] }>('/artifacts', {
        limit: 50,
        status
      });

      if (response && response.artifacts) {
        setArtifacts(response.artifacts);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load artifacts');
    } finally {
      setLoading(false);
    }
  };

  const loadArtifactContent = async (artifact: Artifact) => {
    if (expandedArtifact === artifact.id) {
      setExpandedArtifact(null);
      setArtifactContent(null);
      return;
    }

    setExpandedArtifact(artifact.id);
    setSelectedArtifact(artifact);
    setContentLoading(true);
    setContentError(null);

    try {
      const response = await apiClient.get<ArtifactContent>(`/artifacts/${artifact.id}/content`);

      if (response && response.status === 'ok' && response.file_content) {
        setArtifactContent(response.file_content);
      } else {
        setContentError(response?.message || 'Failed to load content');
      }
    } catch (err) {
      setContentError(err instanceof Error ? err.message : 'Failed to load artifact content');
    } finally {
      setContentLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return <CheckCircle size={16} className="text-green-500" />;
      case 'failed':
        return <XCircle size={16} className="text-red-500" />;
      case 'pending':
        return <Clock size={16} className="text-yellow-500" />;
      default:
        return <Clock size={16} className="text-gray-500" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const styles = {
      passed: 'bg-green-100 text-green-800 border-green-200',
      failed: 'bg-red-100 text-red-800 border-red-200',
      pending: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    };

    const style = styles[status as keyof typeof styles] || 'bg-gray-100 text-gray-800 border-gray-200';

    return (
      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${style}`}>
        {status.toUpperCase()}
      </span>
    );
  };

  const getTypeBadge = (type: string) => {
    const colors = {
      FILE: 'bg-blue-100 text-blue-800',
      KNOWLEDGE: 'bg-purple-100 text-purple-800',
      DATASET: 'bg-orange-100 text-orange-800',
      REPORT: 'bg-teal-100 text-teal-800',
      LINK: 'bg-pink-100 text-pink-800',
    };

    const color = colors[type as keyof typeof colors] || 'bg-gray-100 text-gray-800';

    return (
      <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${color}`}>
        {type}
      </span>
    );
  };

  const getFileName = (location: string) => {
    return location.split('/').pop() || location;
  };

  if (loading && artifacts.length === 0) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
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
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="p-6 pb-4 flex-shrink-0">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">Артефакты</h1>
          <p className="text-gray-600 mt-2">
            Просмотр результатов выполнения целей (файлы, знания, отчеты)
          </p>
        </div>

        {/* Filters */}
        <div className="mb-4 flex items-center gap-4">
        <span className="text-sm text-gray-600">Фильтр:</span>
        <div className="flex gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === 'all'
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            Все ({artifacts.length})
          </button>
          <button
            onClick={() => setFilter('passed')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === 'passed'
                ? 'bg-green-600 text-white'
                : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            Успешные
          </button>
          <button
            onClick={() => setFilter('failed')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === 'failed'
                ? 'bg-red-600 text-white'
                : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
            }`}
          >
            Ошибки
          </button>
        </div>
      </div>
      </div>

      {/* Scrollable Artifacts List */}
      <div className="flex-1 overflow-y-auto px-6 pb-6">
        <div className="space-y-4">
        {artifacts.map((artifact) => (
          <div
            key={artifact.id}
            className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden"
          >
            <div className="p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-2">
                    <FileText size={20} className="text-blue-500 flex-shrink-0" />
                    <h3 className="text-lg font-semibold text-gray-900 truncate">
                      {getFileName(artifact.content_location)}
                    </h3>
                    {getStatusBadge(artifact.verification_status)}
                    {getTypeBadge(artifact.type)}
                  </div>

                  <div className="ml-9 space-y-2">
                    <div className="flex items-center gap-4 text-sm text-gray-600">
                      <span>🎯 Goal: <code className="text-xs">{artifact.goal_id.slice(0, 8)}...</code></span>
                      <span>🔧 Skill: <span className="font-mono text-xs">{artifact.skill_name}</span></span>
                      <span>📅 {new Date(artifact.created_at).toLocaleString('ru-RU')}</span>
                    </div>

                    {artifact.description && (
                      <p className="text-sm text-gray-500 line-clamp-2">
                        {artifact.description}
                      </p>
                    )}
                  </div>
                </div>

                <button
                  onClick={() => loadArtifactContent(artifact)}
                  disabled={contentLoading}
                  className="ml-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 flex items-center gap-2"
                >
                  {expandedArtifact === artifact.id ? (
                    <>
                      <ChevronUp size={16} />
                      Скрыть
                    </>
                  ) : (
                    <>
                      <Eye size={16} />
                      Показать
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Content Preview */}
            {expandedArtifact === artifact.id && (
              <div className="border-t border-gray-200 bg-gray-50 max-h-96 overflow-y-auto">
                {contentLoading ? (
                  <div className="p-4 text-center text-gray-500">
                    Загрузка содержимого...
                  </div>
                ) : contentError ? (
                  <div className="p-4 bg-red-50 text-red-700">
                    Ошибка: {contentError}
                  </div>
                ) : artifactContent ? (
                  <div className="p-4">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs text-gray-500 font-mono">
                        {artifact.content_location}
                      </span>
                      <span className="text-xs text-gray-500">
                        {artifactContent.length} символов
                      </span>
                    </div>
                    <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm leading-relaxed">
                      {artifactContent}
                    </pre>
                  </div>
                ) : null}
              </div>
            )}
          </div>
        ))}
        </div>

        {artifacts.length === 0 && !loading && (
          <div className="text-center py-12">
            <FileText size={48} className="text-gray-300 mx-auto mb-4" />
            <p className="text-gray-500">Артефакты не найдены</p>
          </div>
        )}
      </div>
    </div>
  );
};
