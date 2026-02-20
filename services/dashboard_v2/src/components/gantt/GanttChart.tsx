/**
 * Gantt Chart View
 *
 * Alternative visualization showing goals as timeline bars with hierarchy
 */

import React, { useMemo, useState } from 'react';
import { useGraphStore } from '../../store/graphStore';
import { useUIStore } from '../../store/uiStore';
import { Clock, CheckCircle, Zap, XCircle, Calendar, ChevronDown, ChevronRight, Filter } from 'lucide-react';

interface GanttBarProps {
  goal: any;
  level: number;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onGoalClick?: (goal: any) => void;
  dateRange: { min: Date; max: Date; totalDays: number };
}

const GanttBar: React.FC<GanttBarProps> = ({ goal, level, isExpanded, onToggleExpand, onGoalClick, dateRange }) => {
  const hasChildren = goal.hasChildren || (goal.childIds && goal.childIds.length > 0);

  // Safely parse dates with fallback to current time
  const parseDate = (dateStr: string | undefined | null): Date => {
    if (!dateStr) return new Date();
    const date = new Date(dateStr);
    return isNaN(date.getTime()) ? new Date() : date;
  };

  const startDate = parseDate(goal.createdAt);
  const endDate = goal.completedAt ? parseDate(goal.completedAt) : new Date();

  // Calculate position and width on timeline with protection against zero duration
  const totalDuration = Math.max(dateRange.max.getTime() - dateRange.min.getTime(), 1);
  const startOffset = totalDuration > 0
    ? ((startDate.getTime() - dateRange.min.getTime()) / totalDuration) * 100
    : 0;
  const barWidth = totalDuration > 0
    ? Math.max(((endDate.getTime() - startDate.getTime()) / totalDuration) * 100, 2)
    : 100;

  const getStatusIcon = () => {
    switch (goal.status) {
      case 'pending':
        return <Clock size={12} className="text-gray-400" />;
      case 'active':
        return <Zap size={12} className="text-blue-400" />;
      case 'done':
        return <CheckCircle size={12} className="text-green-400" />;
      case 'blocked':
        return <XCircle size={12} className="text-red-400" />;
      default:
        return null;
    }
  };

  const getGoalTypeColor = () => {
    switch (goal.goalType) {
      case 'achievable':
        return 'border-blue-500 bg-blue-500/20';
      case 'unachievable':
        return 'border-red-500 bg-red-500/20';
      case 'philosophical':
        return 'border-purple-500 bg-purple-500/20';
      default:
        return 'border-gray-500 bg-gray-500/20';
    }
  };

  const getGoalTypeBarColor = () => {
    switch (goal.goalType) {
      case 'achievable':
        return 'bg-blue-500';
      case 'unachievable':
        return 'bg-red-500';
      case 'philosophical':
        return 'bg-purple-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusBarColor = () => {
    if (goal.status === 'done') return 'bg-green-500';
    if (goal.status === 'active') return 'bg-blue-500';
    if (goal.status === 'blocked') return 'bg-red-500';
    return 'bg-gray-500';
  };

  return (
    <div className="border-b border-gray-700 hover:bg-gray-800/30 transition-colors">
      <div
        className="flex items-center cursor-pointer"
        onClick={() => onGoalClick?.(goal)}
      >
        {/* Left Panel: Goal Info */}
        <div className="flex-1 flex items-center gap-2 py-2 min-w-0">
          {/* Expand/Collapse */}
          <div className="w-6 flex-shrink-0 flex justify-center">
            {hasChildren && (
              <button
                onClick={onToggleExpand}
                className="text-gray-500 hover:text-white transition-colors p-0.5"
              >
                {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </button>
            )}
          </div>

          {/* Status Icon */}
          <div className="w-5 flex-shrink-0 flex justify-center">{getStatusIcon()}</div>

          {/* Goal Details */}
          <div
            className="flex-1 min-w-0"
            style={{ marginLeft: `${level * 20}px` }}
          >
            <div className="text-sm text-white font-medium truncate">{goal.intent}</div>
            <div className="text-xs text-gray-400 flex items-center gap-2">
              <span className={`px-1.5 py-0.5 rounded ${getGoalTypeColor()}`}>
                {goal.goalType}
              </span>
              <span>{Math.round(goal.progress * 100)}%</span>
            </div>
          </div>

          {/* Duration Badge */}
          <div className="text-xs text-gray-500 flex-shrink-0 px-2">
            {Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24))}d
          </div>
        </div>

        {/* Right Panel: Timeline Bar */}
        <div className="w-1/3 px-4 py-2 relative" style={{ minWidth: '250px' }}>
          <div className="relative h-full flex items-center">
            {/* Timeline Bar */}
            <div
              className={`h-6 rounded ${getGoalTypeBarColor()} relative overflow-hidden`}
              style={{
                marginLeft: `${startOffset}%`,
                width: `${barWidth}%`,
                minWidth: '4px',
              }}
            >
              {/* Progress Overlay */}
              <div
                className={`absolute inset-y-0 left-0 ${getStatusBarColor()} opacity-50`}
                style={{ width: `${goal.progress * 100}%` }}
              />
              {/* Progress Bar Border */}
              <div className="absolute inset-0 border border-white/20 rounded" />
            </div>

            {/* Hover Tooltip */}
            <div className="absolute inset-0 group">
              <div className="hidden group-hover:block absolute z-20 bg-gray-900 border border-gray-600 rounded px-2 py-1 text-xs whitespace-nowrap">
                {startDate.toLocaleDateString('ru-RU')} - {endDate.toLocaleDateString('ru-RU')}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const GanttChart: React.FC = () => {
  const { nodes, getFilteredNodes } = useGraphStore();
  const { dispatchEvent } = useUIStore();
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [showAllGoals, setShowAllGoals] = useState<boolean>(true); // По умолчанию показывать все

  // Date range filter
  const [filterStart, setFilterStart] = useState<string>('');
  const [filterEnd, setFilterEnd] = useState<string>('');

  const toggleExpand = (nodeId: string) => {
    setExpandedNodes((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  // Build hierarchical tree and calculate timeline
  const { timelineData, dateRange, goalsMap } = useMemo(() => {
    let filteredNodes = getFilteredNodes().filter((n: any) => n.type === 'goal');
    let goalNodes = filteredNodes as any[];

    if (goalNodes.length === 0) {
      return { timelineData: [], dateRange: { min: new Date(), max: new Date(), totalDays: 0 }, goalsMap: new Map() };
    }

    // Apply date filters with safe date parsing
    if (filterStart) {
      const startDate = new Date(filterStart);
      if (!isNaN(startDate.getTime())) {
        goalNodes = goalNodes.filter((g) => {
          const goalDate = g.createdAt ? new Date(g.createdAt) : new Date();
          return goalDate >= startDate;
        });
      }
    }
    if (filterEnd) {
      const endDate = new Date(filterEnd);
      if (!isNaN(endDate.getTime())) {
        goalNodes = goalNodes.filter((g) => {
          const completed = g.completedAt ? new Date(g.completedAt) : new Date();
          return completed <= endDate;
        });
      }
    }

    if (goalNodes.length === 0) {
      return { timelineData: [], dateRange: { min: new Date(), max: new Date(), totalDays: 0 }, goalsMap: new Map() };
    }

    // Build child relationships from parent_id
    const goalsWithChildren = goalNodes.map(goal => {
      const childIds = goalNodes
        .filter(g => g.parentId === goal.id)
        .map(g => g.id);

      return {
        ...goal,
        childIds,
        hasChildren: childIds.length > 0
      };
    });

    // Find date range
    const dates = goalsWithChildren.flatMap((g) => [
      new Date(g.createdAt),
      g.completedAt ? new Date(g.completedAt) : new Date(),
    ]);

    const minDate = new Date(Math.min(...dates.map(d => d.getTime())));
    const maxDate = new Date(Math.max(...dates.map(d => d.getTime())));

    // Add padding
    minDate.setDate(minDate.getDate() - 1);
    maxDate.setDate(maxDate.getDate() + 1);

    const totalDays = Math.ceil((maxDate.getTime() - minDate.getTime()) / (1000 * 60 * 60 * 24));

    // Build all goals map for hierarchy
    const map = new Map(goalsWithChildren.map(g => [g.id, g]));

    // Filter timeline data based on showAllGoals setting
    const rootGoals = goalsWithChildren.filter((g) => !g.parentId);
    const timelineDataToShow = showAllGoals ? rootGoals : goalsWithChildren.filter((g) => !g.parentId);

    return {
      timelineData: timelineDataToShow,
      dateRange: { min: minDate, max: maxDate, totalDays },
      goalsMap: map,
    };
  }, [nodes, getFilteredNodes, filterStart, filterEnd, showAllGoals]);

  // Handle goal click to show in Inspector
  const handleGoalClick = (goal: any) => {
    dispatchEvent({
      type: 'SELECT_NODE',
      nodeId: goal.id,
      nodeType: 'goal',
    });
  };

  // Recursively render goal hierarchy
  const renderGoalTree = (goal: any, level: number): React.ReactNode => {
    const isExpanded = expandedNodes.has(goal.id);
    const hasChildren = goal.hasChildren && goal.childIds && goal.childIds.length > 0;

    return (
      <div key={goal.id}>
        <GanttBar
          goal={goal}
          level={level}
          isExpanded={isExpanded}
          onToggleExpand={() => toggleExpand(goal.id)}
          onGoalClick={handleGoalClick}
          dateRange={dateRange}
        />
        {isExpanded && hasChildren && (
          <div>
            {goal.childIds.map((childId: string) => {
              const childGoal = goalsMap.get(childId);
              if (childGoal) {
                return renderGoalTree(childGoal, level + 1);
              }
              return null;
            })}
          </div>
        )}
      </div>
    );
  };

  if (timelineData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900">
        <div className="text-gray-400">No goals to display</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <Calendar size={18} className="text-blue-400" />
            <h2 className="text-lg font-semibold text-white">Timeline (Gantt) View</h2>
            <div className="text-sm text-gray-400">
              {dateRange.min.toLocaleDateString('ru-RU')} - {dateRange.max.toLocaleDateString('ru-RU')}
              <span className="ml-2 text-gray-500">({dateRange.totalDays} days)</span>
            </div>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center gap-2 bg-gray-700 rounded-lg px-3 py-1.5">
            <button
              onClick={() => setShowAllGoals(true)}
              className={`text-xs px-2 py-1 rounded transition-colors ${
                showAllGoals
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Все цели (иерархия)
            </button>
            <button
              onClick={() => setShowAllGoals(false)}
              className={`text-xs px-2 py-1 rounded transition-colors ${
                !showAllGoals
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Только корневые
            </button>
          </div>
        </div>

        {/* Stats and Filters Row */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-4 text-gray-400">
            <span>Всего целей: <span className="text-white font-semibold">{goalsMap.size}</span></span>
            <span>Корневых: <span className="text-white font-semibold">{timelineData.length}</span></span>
            {showAllGoals && (
              <span className="text-blue-400">💡 Разверните цели чтобы увидеть подцели</span>
            )}
          </div>

          {/* Date Filters */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <Filter size={14} className="text-gray-400" />
              <input
                type="date"
                value={filterStart}
                onChange={(e) => setFilterStart(e.target.value)}
                className="bg-gray-700 text-white text-xs rounded px-2 py-1 border border-gray-600 focus:border-blue-500 focus:outline-none"
                placeholder="Start date"
              />
              <span className="text-gray-500">→</span>
              <input
                type="date"
                value={filterEnd}
                onChange={(e) => setFilterEnd(e.target.value)}
                className="bg-gray-700 text-white text-xs rounded px-2 py-1 border border-gray-600 focus:border-blue-500 focus:outline-none"
                placeholder="End date"
              />
              {(filterStart || filterEnd) && (
                <button
                  onClick={() => {
                    setFilterStart('');
                    setFilterEnd('');
                  }}
                  className="text-xs text-blue-400 hover:text-blue-300"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-2 text-xs">
          <span className="text-gray-400">Goal Types:</span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-blue-500" />
            <span className="text-gray-300">Achievable</span>
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-purple-500" />
            <span className="text-gray-300">Philosophical</span>
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-red-500" />
            <span className="text-gray-300">Unachievable</span>
          </span>
          <span className="ml-4 text-gray-400">Status Overlay:</span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-green-500 opacity-50" />
            <span className="text-gray-300">Done</span>
          </span>
          <span className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-blue-500 opacity-50" />
            <span className="text-gray-300">Active</span>
          </span>
        </div>
      </div>

      {/* Timeline Header */}
      <div className="bg-gray-800/50 border-b border-gray-700 px-4 py-2">
        <div className="flex items-center">
          <div className="flex-1 text-xs text-gray-400 font-medium">
            Goal Hierarchy
          </div>
          <div className="w-1/3 text-xs text-gray-400 font-medium text-center px-4" style={{ minWidth: '250px' }}>
            Timeline
          </div>
        </div>
      </div>

      {/* Timeline Content */}
      <div className="flex-1 overflow-y-auto">
        {timelineData.map((goal) => renderGoalTree(goal, 0))}
      </div>
    </div>
  );
};

export default GanttChart;
