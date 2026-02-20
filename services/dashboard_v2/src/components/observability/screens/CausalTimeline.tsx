/**
 * Screen 4: Causal Timeline (CORE SCREEN)
 *
 * Question: How did events actually unfold?
 * Frequency: During incident investigation
 *
 * Most powerful tool for observation
 */

import { useEffect, useState } from 'react';
import { Clock, ChevronRight } from 'lucide-react';
import { apiClient } from '../../../api/client';

interface TimelineEvent {
  id: string;
  type: 'forecast' | 'context' | 'outcome' | 'error' | 'alert' | 'candidate';
  timestamp: string;
  data: any;
}

export function CausalTimeline() {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);

  useEffect(() => {
    loadTimeline();
  }, []);

  const loadTimeline = async () => {
    setLoading(true);
    try {
      // Load goals from backend to build timeline
      const goals = await apiClient.fetchV1Goals();

      // Transform goals to timeline events
      const timelineEvents: TimelineEvent[] = goals
        .slice(0, 50)
        .map((goal: any) => {
          // Determine event type based on goal status
          let eventType: TimelineEvent['type'] = 'forecast';
          if (goal.status === 'done' || goal.status === 'completed') {
            eventType = 'outcome';
          } else if (goal.status === 'failed' || goal.status === 'blocked') {
            eventType = 'error';
          } else if (goal.status === 'in_progress') {
            eventType = 'context';
          }

          return {
            id: goal.id,
            type: eventType,
            timestamp: goal.created_at || goal.updated_at || new Date().toISOString(),
            data: {
              node_id: goal.id,
              event_type: goal.status,
              node_type: 'goal',
              title: goal.title,
              status: goal.status,
              goal_type: goal.goal_type,
              progress: goal.progress
            }
          };
        });

      // Sort by timestamp (newest first)
      timelineEvents.sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );

      setEvents(timelineEvents);
    } catch (error) {
      console.error('Failed to load timeline:', error);
      setEvents([]);
    } finally {
      setLoading(false);
    }
  };

  const getEventTypeLabel = (type: string) => {
    switch (type) {
      case 'forecast': return 'Прогноз Выдан';
      case 'context': return 'Снимок Контекста';
      case 'outcome': return 'Результат Наблюдён';
      case 'error': return 'Метка Ошибки';
      case 'alert': return 'Тревога Активирована';
      case 'candidate': return 'Кандидат Интервенции';
      default: return type;
    }
  };

  const getEventTypeColor = (type: string) => {
    switch (type) {
      case 'forecast': return 'border-blue-900/50 bg-blue-900/10';
      case 'context': return 'border-gray-700 bg-gray-900/50';
      case 'outcome': return 'border-green-900/50 bg-green-900/10';
      case 'error': return 'border-red-900/50 bg-red-900/10';
      case 'alert': return 'border-yellow-900/50 bg-yellow-900/10';
      case 'candidate': return 'border-purple-900/50 bg-purple-900/10';
      default: return 'border-gray-800 bg-gray-900';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="text-blue-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Каузальная Временная Шкала Событий</h3>
        </div>

        <p className="text-sm text-gray-500">
          Последовательный вид того, как на самом деле развивались события
        </p>

        <div className="mt-4 p-3 bg-blue-900/10 border border-blue-900/50 rounded text-xs text-blue-400">
          <div className="font-mono mb-1">ОСНОВНОЙ ЭКРАН:</div>
          Это самый мощный инструмент наблюдения.<br />
          Показывает: Прогноз → Контекст → Результат → Метка Ошибки → Тревога → Кандидат → (Нет Действия)
        </div>
      </div>

      {/* Timeline */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-gray-500">Загрузка временной шкалы...</div>
        </div>
      ) : (
        <div className="space-y-2">
          {events.map((event, index) => (
            <div key={event.id} className="relative">
              {/* Timeline line */}
              {index < events.length - 1 && (
                <div className="absolute left-4 top-8 bottom-0 w-px bg-gray-800" />
              )}

              {/* Event card */}
              <div className="flex gap-4">
                {/* Timeline dot */}
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-900 border-2 border-gray-700 flex items-center justify-center z-10">
                  <ChevronRight size={14} className="text-gray-500" />
                </div>

                {/* Event content */}
                <div
                  className={`flex-1 border rounded-lg p-4 cursor-pointer transition-colors ${getEventTypeColor(event.type)}`}
                  onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-mono text-gray-400 uppercase">
                          {getEventTypeLabel(event.type)}
                        </span>
                        <span className="text-xs text-gray-600">
                          {new Date(event.timestamp).toLocaleString()}
                        </span>
                      </div>

                      {event.data?.title && (
                        <div className="text-sm text-gray-300 truncate">
                          {event.data.title}
                        </div>
                      )}

                      {event.data?.status && (
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs text-gray-500">
                            Status: {event.data.status}
                          </span>
                          {event.data?.progress !== undefined && (
                            <span className="text-xs text-gray-600">
                              ({Math.round(event.data.progress * 100)}%)
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Expanded details */}
                  {expandedEvent === event.id && (
                    <div className="mt-4 pt-4 border-t border-gray-800">
                      <details className="text-xs">
                        <summary className="cursor-pointer text-gray-500 hover:text-gray-400 mb-2">
                          Сырые данные события
                        </summary>
                        <pre className="bg-gray-950 p-3 rounded overflow-x-auto text-gray-600">
                          {JSON.stringify(event.data, null, 2)}
                        </pre>
                      </details>

                      <div className="mt-3 space-y-2">
                        <div className="text-xs text-gray-600">
                          <span className="font-mono">УВЕРЕННОСТЬ:</span> -
                        </div>
                        <div className="text-xs text-gray-600">
                          <span className="font-mono">ПОДАВЛЕННЫЕ АЛЬТЕРНАТИВЫ:</span> Не отслеживается
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Usage guidance */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h4 className="text-sm font-semibold text-gray-300 mb-3">Как использовать эту временную шкалу</h4>

        <div className="space-y-2 text-xs text-gray-500">
          <div>
            <span className="font-mono text-gray-400">1.</span> Нажмите на событие, чтобы раскрыть детали
          </div>
          <div>
            <span className="font-mono text-gray-400">2.</span> Ищите разрывы между прогнозом и результатом
          </div>
          <div>
            <span className="font-mono text-gray-400">3.</span> Проверьте, были ли тревоги активированы до ошибок
          </div>
          <div>
            <span className="font-mono text-gray-400">4.</span> Проверьте, были ли кандидаты зарегистрированы, но не выполнены
          </div>
          <div>
            <span className="font-mono text-gray-400">5.</span> Идентифицируйте систематические паттерны в нескольких цепочках
          </div>
        </div>
      </div>
    </div>
  );
}
