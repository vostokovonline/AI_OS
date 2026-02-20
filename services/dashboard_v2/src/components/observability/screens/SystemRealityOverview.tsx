/**
 * Screen 1: System Reality Overview
 *
 * Question: What does the system know about itself now?
 * Frequency: Daily / Weekly
 */

import { useEffect, useState } from 'react';
import { Activity, TrendingUp, BarChart3 } from 'lucide-react';
import { apiClient } from '../../../api/client';

export function SystemRealityOverview() {
  const [timeRange, setTimeRange] = useState<'7d' | '30d'>('7d');
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    loadData();
  }, [timeRange]);

  const loadData = async () => {
    try {
      const resp = await apiClient.get('/goals/stats');
      setStats(resp);
    } catch (error) {
      console.error('Failed to load system reality:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Time Range Selector (Read-Only) */}
      <div className="flex items-center gap-4">
        <div className="text-sm text-gray-400">Период:</div>
        <div className="flex gap-2">
          <button
            onClick={() => setTimeRange('7d')}
            className={`px-3 py-1 rounded text-sm border transition-colors ${
              timeRange === '7d'
                ? 'bg-blue-900/20 text-blue-400 border-blue-900/50'
                : 'bg-gray-900 text-gray-500 border-gray-800'
            }`}
          >
            7 дней
          </button>
          <button
            onClick={() => setTimeRange('30d')}
            className={`px-3 py-1 rounded text-sm border transition-colors ${
              timeRange === '30d'
                ? 'bg-blue-900/20 text-blue-400 border-blue-900/50'
                : 'bg-gray-900 text-gray-500 border-gray-800'
            }`}
          >
            30 дней
          </button>
        </div>
      </div>

      {/* Activity Window */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="text-blue-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Окно Активности</h3>
        </div>

        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-950 border border-gray-800 rounded p-4">
            <div className="text-xs text-gray-500 mb-1">Всего Целей</div>
            <div className="text-2xl font-mono text-gray-200">{stats?.total || '-'}</div>
            <div className="text-xs text-gray-600 mt-2">Все время</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-4">
            <div className="text-xs text-gray-500 mb-1">Завершено</div>
            <div className="text-2xl font-mono text-gray-200">{stats?.by_status?.done || '-'}</div>
            <div className="text-xs text-gray-600 mt-2">Статус: done</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-4">
            <div className="text-xs text-gray-500 mb-1">Выполняется</div>
            <div className="text-2xl font-mono text-gray-200">{stats?.by_status?.in_progress || '-'}</div>
            <div className="text-xs text-gray-600 mt-2">В процессе</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-4">
            <div className="text-xs text-gray-500 mb-1">Ожидают</div>
            <div className="text-2xl font-mono text-gray-200">{stats?.by_status?.pending || '-'}</div>
            <div className="text-xs text-gray-600 mt-2">В очереди</div>
          </div>
        </div>
      </div>

      {/* Prediction Reality Gap */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="text-purple-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Разрыв Уверенности — Точности</h3>
        </div>

        <div className="bg-gray-950 border border-gray-800 rounded p-6">
          <div className="text-center text-gray-500 text-sm">
            <div className="mb-2">Нет данных о прогнозах</div>
            <div className="text-xs text-gray-600">
              Эта метрика отслеживает: <br />
              - Средняя уверенность (mean) <br />
              - Реализованная точность (post-hoc) <br />
              - Разрыв (Δ) с графиком
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-gray-950/50 border border-gray-800 rounded text-xs text-gray-500">
          <div className="font-mono mb-1">ИНТЕРПРЕТАЦИЯ:</div>
          Никакой интерпретации. Показывает только факт расхождения.
        </div>
      </div>

      {/* Model Contribution Split */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="text-green-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Распределение Источников Решений</h3>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left py-2 px-3 text-gray-400 font-medium">Источник</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Количество</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Совпадение %</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Воздержание %</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-800/50">
                <td className="py-2 px-3 text-gray-300">LLM</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-2 px-3 text-gray-300">Правила</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
              </tr>
              <tr>
                <td className="py-2 px-3 text-gray-300">Гибрид</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
                <td className="py-2 px-3 text-right text-gray-500">-</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Drift Signals */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-100 mb-4">Сигналы Дрейфа</h3>

        <div className="space-y-3">
          <div className="bg-gray-950 border border-gray-800 rounded p-3">
            <div className="text-xs text-gray-500 mb-1">Оценка Дрейфа Признаков</div>
            <div className="text-sm text-gray-600">Не отслеживается</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-3">
            <div className="text-xs text-gray-500 mb-1">Сдвиг Распределения Эмоций</div>
            <div className="text-sm text-gray-600">Не отслеживается</div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-3">
            <div className="text-xs text-gray-500 mb-1">Скорость Новизны Контекста</div>
            <div className="text-sm text-gray-600">Не отслеживается</div>
          </div>
        </div>
      </div>
    </div>
  );
}
