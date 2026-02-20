/**
 * v2 UI - Control Panel
 *
 * Left-side panel for mode switching and constraint management
 */

import React, { useState } from 'react';
import { useUIStore } from '../../store/uiStore';
import { useGraphStore } from '../../store/graphStore';
import { useToastStore } from '../../store/toastStore';
import { useExecutionLogStore } from '../../store/executionLogStore';
import { FilterState } from '../../store/graphStore';
import {
  Brain,
  Zap,
  Eye,
  Shield,
  DollarSign,
  Clock,
  Ban,
  Flame,
  AlertTriangle,
  Microscope,
  Search,
  CheckCircle,
  XCircle,
  Filter,
  Network,
  Calendar,
  GitBranch,
  Heart,
  MessageCircle,
  Package,
  Rocket,
  Activity,
  Globe,
  FileText,
} from 'lucide-react';

interface ControlPanelProps {
  onToggleEmotionalLayer?: () => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ onToggleEmotionalLayer }) => {
  const {
    mode,
    view,
    overlay,
    constraints,
    override,
    setMode,
    setView,
    setOverlay,
    clearOverlay,
    setOverride,
  } = useUIStore();

  const { nodes, filters, setFilters, getFilteredNodes } = useGraphStore();
  const addToast = useToastStore((state) => state.add);
  const addLog = useExecutionLogStore((state) => state.add);
  const [searchQuery, setSearchQueryLocal] = useState(filters.searchQuery);

  const filteredCount = getFilteredNodes().length;
  const totalCount = nodes.size;

  const handleSearch = (value: string) => {
    setSearchQueryLocal(value);
    setFilters({ searchQuery: value });
  };

  const toggleStatusFilter = (status: keyof FilterState) => {
    setFilters({ [status]: !filters[status] });
  };

  const quickFilters = [
    { key: 'showActive' as const, label: 'Active', icon: Zap, color: 'text-blue-400' },
    { key: 'showPending' as const, label: 'Pending', icon: Clock, color: 'text-gray-400' },
    { key: 'showDone' as const, label: 'Done', icon: CheckCircle, color: 'text-green-400' },
    { key: 'showBlocked' as const, label: 'Blocked', icon: XCircle, color: 'text-red-400' },
  ];

  const modes = [
    { id: 'explore' as const, icon: Brain, label: 'Explore', color: 'text-blue-400' },
    { id: 'exploit' as const, icon: Zap, label: 'Exploit', color: 'text-yellow-400' },
    { id: 'reflect' as const, icon: Eye, label: 'Reflect', color: 'text-purple-400' },
  ];

  const overlays = [
    { id: 'none' as const, icon: null, label: 'None' },
    { id: 'heatmap' as const, icon: Flame, label: 'Heatmap', color: 'text-orange-400' },
    { id: 'conflicts' as const, icon: AlertTriangle, label: 'Conflicts', color: 'text-red-400' },
    {
      id: 'memory_traces' as const,
      icon: Microscope,
      label: 'Memory Traces',
      color: 'text-purple-400',
    },
  ];

  return (
    <div className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
      {/* View Selector */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-gray-400 text-xs uppercase mb-3 font-bold">View</h3>
        <div className="space-y-2">
          <button
            onClick={() => setView('graph')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'graph'
                  ? 'bg-purple-900/50 border border-purple-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Network size={18} className={view === 'graph' ? 'text-purple-400' : ''} />
            <span className="text-sm font-medium">Graph</span>
          </button>
          <button
            onClick={() => setView('gantt')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'gantt'
                  ? 'bg-purple-900/50 border border-purple-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Calendar size={18} className={view === 'gantt' ? 'text-purple-400' : ''} />
            <span className="text-sm font-medium">Timeline (Gantt)</span>
          </button>
          <button
            onClick={() => setView('tree')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'tree'
                  ? 'bg-purple-900/50 border border-purple-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <GitBranch size={18} className={view === 'tree' ? 'text-purple-400' : ''} />
            <span className="text-sm font-medium">Dependency Tree</span>
          </button>
          <button
            onClick={() => setView('observability')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'observability'
                  ? 'bg-blue-900/50 border border-blue-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Eye size={18} className={view === 'observability' ? 'text-blue-400' : ''} />
            <span className="text-sm font-medium">Наблюдаемость</span>
          </button>
          <button
            onClick={() => setView('questions')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'questions'
                  ? 'bg-green-900/50 border border-green-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <MessageCircle size={18} className={view === 'questions' ? 'text-green-400' : ''} />
            <span className="text-sm font-medium">Вопросы</span>
          </button>
          <button
            onClick={() => setView('decomposition')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'decomposition'
                  ? 'bg-purple-900/50 border border-purple-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Brain size={18} className={view === 'decomposition' ? 'text-purple-400' : ''} />
            <span className="text-sm font-medium">Декомпозиция</span>
          </button>
        </div>
      </div>

      {/* OCCP Section */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-gray-400 text-xs uppercase mb-3 font-bold">OCCP v1.0</h3>
        <div className="space-y-2">
          <button
            onClick={() => setView('skills')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'skills'
                  ? 'bg-green-900/50 border border-green-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Package size={18} className={view === 'skills' ? 'text-green-400' : ''} />
            <span className="text-sm font-medium">Skills</span>
          </button>
          <button
            onClick={() => setView('deployments')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'deployments'
                  ? 'bg-yellow-900/50 border border-yellow-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Rocket size={18} className={view === 'deployments' ? 'text-yellow-400' : ''} />
            <span className="text-sm font-medium">Deployments</span>
          </button>
          <button
            onClick={() => setView('occp-observability')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'occp-observability'
                  ? 'bg-blue-900/50 border border-blue-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Activity size={18} className={view === 'occp-observability' ? 'text-blue-400' : ''} />
            <span className="text-sm font-medium">Metrics</span>
          </button>
          <button
            onClick={() => setView('federation')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'federation'
                  ? 'bg-purple-900/50 border border-purple-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Globe size={18} className={view === 'federation' ? 'text-purple-400' : ''} />
            <span className="text-sm font-medium">Federation</span>
          </button>
          <button
            onClick={() => setView('artifacts')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'artifacts'
                  ? 'bg-orange-900/50 border border-orange-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <FileText size={18} className={view === 'artifacts' ? 'text-orange-400' : ''} />
            <span className="text-sm font-medium">Artifacts</span>
          </button>
          <button
            onClick={() => setView('autonomy')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'autonomy'
                  ? 'bg-indigo-900/50 border border-indigo-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Brain size={18} className={view === 'autonomy' ? 'text-indigo-400' : ''} />
            <span className="text-sm font-medium">Autonomy</span>
          </button>
          <button
            onClick={() => setView('admin')}
            className={`
              w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
              ${
                view === 'admin'
                  ? 'bg-cyan-900/50 border border-cyan-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            <Shield size={18} className={view === 'admin' ? 'text-cyan-400' : ''} />
            <span className="text-sm font-medium">Admin</span>
          </button>
        </div>
      </div>

      {/* Mode Selector */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-gray-400 text-xs uppercase mb-3 font-bold">Thinking Mode</h3>
        <div className="space-y-2">
          {modes.map((m) => (
            <button
              key={m.id}
              onClick={() => setMode(m.id)}
              className={`
                w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
                ${
                  mode === m.id
                    ? 'bg-blue-900/50 border border-blue-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }
              `}
            >
              <m.icon size={18} className={mode === m.id ? m.color : ''} />
              <span className="text-sm font-medium">{m.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Filter Controls */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-gray-400 text-xs uppercase font-bold flex items-center gap-2">
            <Filter size={14} />
            Filters
          </h3>
          <span className="text-gray-500 text-xs">
            {filteredCount}/{totalCount}
          </span>
        </div>

        {/* Search */}
        <div className="mb-3">
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search goals..."
              value={searchQuery}
              onChange={(e) => handleSearch(e.target.value)}
              className="w-full bg-gray-700 text-white text-sm rounded-lg pl-10 pr-3 py-2 border border-gray-600 focus:border-blue-500 focus:outline-none"
            />
            {searchQuery && (
              <button
                onClick={() => handleSearch('')}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
              >
                <XCircle size={16} />
              </button>
            )}
          </div>
        </div>

        {/* Quick Status Filters */}
        <div className="grid grid-cols-2 gap-2">
          {quickFilters.map((filter) => (
            <button
              key={filter.key}
              onClick={() => toggleStatusFilter(filter.key)}
              className={`
                flex items-center gap-2 px-2 py-1.5 rounded text-xs transition-all
                ${filters[filter.key]
                  ? `${filter.color} bg-gray-700 border border-current`
                  : 'text-gray-500 hover:text-gray-300 bg-gray-800'
                }
              `}
            >
              <filter.icon size={14} />
              <span>{filter.label}</span>
            </button>
          ))}
        </div>

        {/* Quick Actions */}
        <div className="flex gap-2">
          <button
            onClick={() => setFilters({
              showOnlyRoots: !filters.showOnlyRoots,
              showDone: true,
              showActive: true,
              showPending: true,
              showBlocked: true
            })}
            className={`
              flex-1 text-xs px-2 py-1.5 rounded transition-all flex items-center justify-center gap-1
              ${filters.showOnlyRoots
                ? 'bg-purple-900/50 text-purple-400 border border-purple-500'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }
            `}
          >
            {filters.showOnlyRoots ? 'Show All' : 'Roots Only'}
          </button>
          <button
            onClick={() => setFilters({ showDone: false, showActive: true, showPending: true, showBlocked: false })}
            className="flex-1 text-xs bg-blue-900/30 text-blue-400 hover:bg-blue-900/50 px-2 py-1 rounded transition-all"
          >
            Hide Done
          </button>
          <button
            onClick={() => setFilters({
              showDone: true,
              showActive: true,
              showPending: true,
              showBlocked: true,
              showOnlyRoots: false
            })}
            className="flex-1 text-xs bg-gray-700 text-gray-300 hover:bg-gray-600 px-2 py-1 rounded transition-all"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Overlay Selector */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-gray-400 text-xs uppercase mb-3 font-bold">View Overlay</h3>
        <div className="space-y-2">
          {overlays.map((o) => (
            <button
              key={o.id}
              onClick={() => (o.id === 'none' ? clearOverlay() : setOverlay(o.id))}
              className={`
                w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all
                ${
                  overlay === o.id
                    ? 'bg-purple-900/50 border border-purple-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }
              `}
            >
              {o.icon && <o.icon size={18} className={overlay === o.id ? o.color : ''} />}
              <span className="text-sm font-medium">{o.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Emotional Layer Toggle */}
      {onToggleEmotionalLayer && (
        <div className="p-4 border-b border-gray-700">
          <button
            onClick={onToggleEmotionalLayer}
            className="w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all text-sm font-medium bg-pink-900/50 border border-pink-500 text-white hover:bg-pink-800/50"
          >
            <Heart size={18} className="text-pink-400" />
            <span>Emotional Layer</span>
          </button>
        </div>
      )}

      {/* Constraints */}
      <div className="p-4 border-b border-gray-700 flex-1">
        <h3 className="text-gray-400 text-xs uppercase mb-3 font-bold">Constraints</h3>

        {/* Ethics */}
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Shield size={16} className="text-green-400" />
            <span className="text-white text-sm">Ethics</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {constraints.ethics.length > 0 ? (
              constraints.ethics.map((ethic, i) => (
                <span
                  key={i}
                  className="text-xs bg-green-800 text-white px-2 py-1 rounded"
                >
                  {ethic}
                </span>
              ))
            ) : (
              <span className="text-gray-400 text-xs">No ethics constraints</span>
            )}
          </div>
        </div>

        {/* Budget */}
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign size={16} className="text-yellow-400" />
            <span className="text-white text-sm">Budget</span>
          </div>
          {constraints.budget ? (
            <div className="text-white text-sm">${constraints.budget.toFixed(2)}</div>
          ) : (
            <span className="text-gray-400 text-xs">No budget limit</span>
          )}
        </div>

        {/* Time Horizon */}
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Clock size={16} className="text-blue-400" />
            <span className="text-white text-sm">Time Horizon</span>
          </div>
          {constraints.timeHorizon ? (
            <div className="text-white text-sm">{constraints.timeHorizon}</div>
          ) : (
            <span className="text-gray-400 text-xs">No time limit</span>
          )}
        </div>
      </div>

      {/* Override Controls */}
      <div className="p-4">
        <h3 className="text-gray-400 text-xs uppercase mb-3 font-bold flex items-center gap-2">
          <Ban size={16} />
          Override
        </h3>

        <button
          onClick={() => setOverride(!override.enabled)}
          className={`
            w-full px-3 py-2 rounded-lg transition-all text-sm font-medium
            ${
              override.enabled
                ? 'bg-red-900/50 border border-red-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }
          `}
        >
          {override.enabled ? 'Override Active' : 'Enable Override'}
        </button>

        {override.enabled && override.decisionId && (
          <div className="mt-2 text-xs text-gray-400">
            Decision ID: {override.decisionId.slice(0, 8)}...
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlPanel;
