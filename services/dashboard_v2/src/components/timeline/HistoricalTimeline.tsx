/**
 * Historical Timeline
 *
 * Shows chronological history of system events and goal executions
 * From past to future with ability to navigate through time
 */

import React, { useMemo, useState } from 'react';
import { useGraphStore } from '../../store/graphStore';
import {
  Clock,
  CheckCircle,
  Zap,
  XCircle,
  ArrowRight,
  Calendar,
  ZoomIn,
  ZoomOut,
  Target,
  Filter,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { ru } from 'date-fns/locale';

interface TimelineEvent {
  id: string;
  timestamp: string;
  type: 'goal_created' | 'goal_started' | 'goal_completed' | 'goal_failed' | 'goal_blocked';
  goalId: string;
  goalIntent: string;
  goalType: 'achievable' | 'unachievable' | 'philosophical';
  description?: string;
}

const HistoricalTimeline: React.FC = () => {
  const { nodes } = useGraphStore();
  const [selectedEvent, setSelectedEvent] = useState<TimelineEvent | null>(null);
  const [filter, setFilter] = useState<'all' | 'achievable' | 'unachievable' | 'philosophical'>('all');
  const [zoom, setZoom] = useState(1); // 1 = normal, 2 = zoomed in

  // Build chronological timeline from goal data
  const { events, dateRange, stats } = useMemo(() => {
    const goals = Array.from(nodes.values()).filter((n: any) => n.type === 'goal') as any[];

    const allEvents: TimelineEvent[] = [];

    goals.forEach((goal) => {
      // Goal created
      allEvents.push({
        id: `${goal.id}-created`,
        timestamp: goal.createdAt,
        type: 'goal_created',
        goalId: goal.id,
        goalIntent: goal.intent,
        goalType: goal.goalType || 'achievable',
        description: 'Цель создана',
      });

      // Goal started
      if (goal.startedAt) {
        allEvents.push({
          id: `${goal.id}-started`,
          timestamp: goal.startedAt,
          type: 'goal_started',
          goalId: goal.id,
          goalIntent: goal.intent,
          goalType: goal.goalType || 'achievable',
          description: 'Выполнение начато',
        });
      }

      // Goal completed/failed/blocked
      if (goal.completedAt) {
        const type = goal.status === 'done'
          ? 'goal_completed'
          : goal.status === 'failed'
          ? 'goal_failed'
          : goal.status === 'blocked'
          ? 'goal_blocked'
          : 'goal_completed';

        const description = goal.status === 'done'
          ? 'Выполнено успешно'
          : goal.status === 'failed'
          ? 'Выполнение не удалось'
          : goal.status === 'blocked'
          ? 'Выполнение заблокировано'
          : 'Завершено';

        allEvents.push({
          id: `${goal.id}-completed`,
          timestamp: goal.completedAt,
          type: type as any,
          goalId: goal.id,
          goalIntent: goal.intent,
          goalType: goal.goalType || 'achievable',
          description,
        });
      }
    });

    // Sort by timestamp (oldest first)
    allEvents.sort((a, b) =>
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // Calculate date range
    if (allEvents.length === 0) {
      return { events: [], dateRange: { min: new Date(), max: new Date() }, stats: { total: 0, completed: 0, active: 0 } };
    }

    const timestamps = allEvents.map(e => new Date(e.timestamp).getTime());
    const minDate = new Date(Math.min(...timestamps));
    const maxDate = new Date(Math.max(...timestamps));

    // Calculate stats
    const total = goals.length;
    const completed = goals.filter((g: any) => g.status === 'done').length;
    const active = goals.filter((g: any) => g.status === 'active').length;

    return {
      events: allEvents,
      dateRange: { min: minDate, max: maxDate },
      stats: { total, completed, active },
    };
  }, [nodes]);

  // Filter events by goal type
  const filteredEvents = useMemo(() => {
    if (filter === 'all') return events;
    return events.filter(e => e.goalType === filter);
  }, [events, filter]);

  // Group events by date for better visualization
  const groupedEvents = useMemo(() => {
    const groups: Record<string, { events: TimelineEvent[]; displayDate: string; sortKey: string }> = {};

    filteredEvents.forEach(event => {
      const eventDate = new Date(event.timestamp);
      const dateKey = eventDate.toISOString().split('T')[0]; // YYYY-MM-DD for grouping
      const displayDate = eventDate.toLocaleDateString('ru-RU'); // For display

      if (!groups[dateKey]) {
        groups[dateKey] = {
          events: [],
          displayDate: displayDate,
          sortKey: dateKey
        };
      }
      groups[dateKey].events.push(event);
    });

    return groups;
  }, [filteredEvents]);

  const getEventIcon = (type: TimelineEvent['type']) => {
    switch (type) {
      case 'goal_created':
        return <Target size={16} className="text-gray-400" />;
      case 'goal_started':
        return <Zap size={16} className="text-blue-400" />;
      case 'goal_completed':
        return <CheckCircle size={16} className="text-green-400" />;
      case 'goal_failed':
        return <XCircle size={16} className="text-red-400" />;
      case 'goal_blocked':
        return <XCircle size={16} className="text-orange-400" />;
      default:
        return <Clock size={16} className="text-gray-400" />;
    }
  };

  const getEventColor = (type: TimelineEvent['type']) => {
    switch (type) {
      case 'goal_created':
        return 'border-l-gray-500';
      case 'goal_started':
        return 'border-l-blue-500';
      case 'goal_completed':
        return 'border-l-green-500';
      case 'goal_failed':
        return 'border-l-red-500';
      case 'goal_blocked':
        return 'border-l-orange-500';
      default:
        return 'border-l-gray-500';
    }
  };

  const getGoalTypeBadge = (goalType: string) => {
    switch (goalType) {
      case 'achievable':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/50';
      case 'unachievable':
        return 'bg-red-500/20 text-red-300 border-red-500/50';
      case 'philosophical':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/50';
      default:
        return 'bg-gray-500/20 text-gray-300 border-gray-500/50';
    }
  };

  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900">
        <div className="text-center">
          <Clock size={48} className="text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">История событий пуста</p>
          <p className="text-sm text-gray-500 mt-2">Создайте цели чтобы увидеть их историю</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gray-900 flex flex-col relative">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <Clock size={20} className="text-blue-400" />
            <h2 className="text-xl font-semibold text-white">История системы</h2>
          </div>

          {/* Stats */}
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="text-gray-300">Всего: {stats.total}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-400" />
              <span className="text-gray-300">Выполнено: {stats.completed}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
              <span className="text-gray-300">Активных: {stats.active}</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Filter */}
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-gray-400" />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as any)}
              className="bg-gray-700 text-white text-sm rounded px-3 py-1.5 border border-gray-600 focus:border-blue-500 focus:outline-none"
            >
              <option value="all">Все типы</option>
              <option value="achievable">Achievable</option>
              <option value="unachievable">Unachievable</option>
              <option value="philosophical">Philosophical</option>
            </select>
          </div>

          {/* Date range */}
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Calendar size={14} />
            <span>{dateRange.min.toLocaleDateString('ru-RU')}</span>
            <ArrowRight size={14} />
            <span>{dateRange.max.toLocaleDateString('ru-RU')}</span>
          </div>

          {/* Zoom */}
          <div className="flex items-center gap-1 ml-auto">
            <button
              onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Уменьшить"
            >
              <ZoomOut size={16} />
            </button>
            <span className="text-xs text-gray-400 w-12 text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={() => setZoom(Math.min(3, zoom + 0.25))}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Увеличить"
            >
              <ZoomIn size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Timeline Content */}
      <div className="flex-1 overflow-y-auto p-6 pb-32">
        {/* Central timeline line */}
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-500 via-purple-500 to-green-500" />

          {/* Events */}
          <div className="space-y-4">
            {Object.entries(groupedEvents)
              .sort(([, a], [, b]) => a.sortKey.localeCompare(b.sortKey))
              .map(([dateKey, groupData]) => (
              <div key={dateKey}>
                {/* Date header */}
                <div className="flex items-center gap-4 mb-3">
                  <div className="w-16 flex-shrink-0 text-center">
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-gray-800 border-2 border-blue-500 text-white font-semibold text-sm z-10 relative">
                      {new Date(dateKey).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' })}
                    </div>
                  </div>
                  <div className="flex-1 text-sm font-semibold text-gray-300">
                    {groupData.displayDate}
                    <span className="ml-2 text-xs text-gray-500 font-normal">
                      ({groupData.events.length} {groupData.events.length === 1 ? 'событие' : groupData.events.length < 5 ? 'события' : 'событий'})
                    </span>
                  </div>
                </div>

                {/* Events for this date */}
                <div className="space-y-2 ml-16">
                  {groupData.events.map((event) => (
                    <div
                      key={event.id}
                      onClick={() => setSelectedEvent(event)}
                      className={`
                        relative bg-gray-800 rounded-lg p-4 border-l-4 ${getEventColor(event.type)}
                        hover:bg-gray-750 cursor-pointer transition-all
                        ${selectedEvent?.id === event.id ? 'ring-2 ring-blue-500' : ''}
                      `}
                      style={{
                        transform: `scale(${zoom})`,
                        transformOrigin: 'left center',
                      }}
                    >
                      {/* Timeline dot */}
                      <div className="absolute -left-8 top-1/2 transform -translate-y-1/2">
                        <div className={`w-3 h-3 rounded-full ${
                          event.type === 'goal_created' ? 'bg-gray-400' :
                          event.type === 'goal_started' ? 'bg-blue-400' :
                          event.type === 'goal_completed' ? 'bg-green-400' :
                          event.type === 'goal_failed' ? 'bg-red-400' :
                          'bg-orange-400'
                        }`} />
                      </div>

                      <div className="flex items-start gap-3">
                        {/* Icon */}
                        <div className="mt-0.5">{getEventIcon(event.type)}</div>

                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-semibold text-white">
                              {event.goalIntent}
                            </span>
                            <span className={`px-2 py-0.5 rounded text-xs border ${getGoalTypeBadge(event.goalType)}`}>
                              {event.goalType}
                            </span>
                          </div>

                          <div className="flex items-center gap-2 text-xs text-gray-400">
                            <Clock size={12} />
                            <span>{new Date(event.timestamp).toLocaleTimeString('ru-RU')}</span>
                            <span>•</span>
                            <span>{event.description}</span>
                            <span className="ml-auto text-gray-500">
                              {formatDistanceToNow(new Date(event.timestamp), { locale: ru, addSuffix: true })}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Selected Event Detail Panel - Fixed at bottom */}
      {selectedEvent && (
        <div className="absolute bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 shadow-2xl z-20">
          <div className="p-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  {getEventIcon(selectedEvent.type)}
                  <h3 className="text-lg font-semibold text-white">{selectedEvent.goalIntent}</h3>
                  <span className={`px-2 py-1 rounded text-xs border ${getGoalTypeBadge(selectedEvent.goalType)}`}>
                    {selectedEvent.goalType}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-400">
                  <span className="flex items-center gap-1">
                    <Calendar size={14} />
                    {new Date(selectedEvent.timestamp).toLocaleString('ru-RU')}
                  </span>
                  <span>{selectedEvent.description}</span>
                </div>
                {/* Raw Event Data */}
                <div className="mt-3 p-3 bg-gray-900 rounded border border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">Сырые данные события:</div>
                  <pre className="text-xs text-gray-300 overflow-x-auto">
                    {JSON.stringify(selectedEvent, null, 2)}
                  </pre>
                </div>
              </div>
              <button
                onClick={() => setSelectedEvent(null)}
                className="text-gray-400 hover:text-white transition-colors p-2"
                title="Закрыть"
              >
                ✕
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoricalTimeline;
