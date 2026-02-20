/**
 * Screen 5: Самооценка Честности
 *
 * Вопрос: Насколько система честна с собой?
 * Частота: Редко, стратегически
 */

import { useEffect, useState } from 'react';
import { Shield, Eye } from 'lucide-react';
import { apiClient } from '../../../api/client';

interface ConfidenceBin {
  range: string;
  reported: number;
  actual: number;
  gap: number;
}

export function SelfEvalIntegrity() {
  const [confidenceBins, setConfidenceBins] = useState<ConfidenceBin[]>([]);
  const [loading, setLoading] = useState(true);
  const [avgProgress, setAvgProgress] = useState(0);
  const [avgCompletion, setAvgCompletion] = useState(0);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const goals = await apiClient.fetchV1Goals();

      // Initialize bins
      const bins: ConfidenceBin[] = [
        { range: '0.0-0.2', reported: 0, actual: 0, gap: 0 },
        { range: '0.2-0.4', reported: 0, actual: 0, gap: 0 },
        { range: '0.4-0.6', reported: 0, actual: 0, gap: 0 },
        { range: '0.6-0.8', reported: 0, actual: 0, gap: 0 },
        { range: '0.8-1.0', reported: 0, actual: 0, gap: 0 }
      ];

      let totalProgress = 0;
      let completedCount = 0;
      let totalCount = 0;

      goals.forEach((goal: any) => {
        const progress = goal.progress || 0;
        const isDone = goal.status === 'done' || goal.status === 'completed';

        // Find the right bin
        const binIndex = Math.min(Math.floor(progress * 5), 4);
        bins[binIndex].reported++;

        // Count actual completions
        if (isDone) {
          bins[binIndex].actual++;
          completedCount++;
        }

        totalProgress += progress;
        totalCount++;
      });

      // Calculate gaps
      bins.forEach(bin => {
        if (bin.reported > 0) {
          bin.gap = bin.reported - bin.actual;
        }
      });

      setConfidenceBins(bins);
      setAvgProgress(totalCount > 0 ? totalProgress / totalCount : 0);
      setAvgCompletion(totalCount > 0 ? (completedCount / totalCount) * 100 : 0);
    } catch (error) {
      console.error('Не удалось загрузить данные самооценки честности:', error);
    } finally {
      setLoading(false);
    }
  };

  const getIntegrityScore = () => {
    if (confidenceBins.length === 0) return 0;

    let totalGap = 0;
    let totalReported = 0;

    confidenceBins.forEach(bin => {
      totalGap += Math.abs(bin.gap);
      totalReported += bin.reported;
    });

    if (totalReported === 0) return 100;
    return Math.max(0, 100 - (totalGap / totalReported * 100));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-2">
          <Shield className="text-purple-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Честность Самооценки</h3>
        </div>

        <p className="text-sm text-gray-500">
          Измеряет, насколько уверенность системы соответствует реальной производительности
        </p>

        <div className="mt-4 p-3 bg-purple-900/10 border border-purple-900/50 rounded text-xs text-purple-400">
          <div className="font-mono mb-1">СТРАТЕГИЧЕСКОЕ ИСПОЛЬЗОВАНИЕ:</div>
          Используйте редко для выявления систематической самоуверенности или дрейфа калибровки
        </div>
      </div>

      {/* Integrity Score */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h4 className="text-sm font-semibold text-gray-300 mb-4">Общий Показатель Честности</h4>

        <div className="flex items-center gap-6">
          <div className="flex-1">
            <div className="flex justify-between text-xs mb-2">
              <span className="text-gray-500">Интегритет</span>
              <span className={`font-mono ${loading ? 'text-gray-500' : getIntegrityScore() > 80 ? 'text-green-400' : getIntegrityScore() > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                {loading ? '...' : getIntegrityScore().toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all ${
                  !loading && getIntegrityScore() > 80 ? 'bg-green-500' :
                  !loading && getIntegrityScore() > 50 ? 'bg-yellow-500' :
                  'bg-red-500'
                }`}
                style={{ width: loading ? '0%' : `${getIntegrityScore()}%` }}
              />
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs text-gray-500">Средний прогресс</div>
            <div className="text-lg font-mono text-gray-200">
              {loading ? '...' : `${(avgProgress * 100).toFixed(1)}%`}
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs text-gray-500">Реальное выполнение</div>
            <div className="text-lg font-mono text-gray-200">
              {loading ? '...' : `${avgCompletion.toFixed(1)}%`}
            </div>
          </div>
        </div>
      </div>

      {/* Self-Eval vs Outcome */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h4 className="text-sm font-semibold text-gray-300 mb-4">Заявленная Уверенность против Реального Выполнения</h4>

        {loading ? (
          <div className="text-center py-8 text-gray-500">Загрузка данных...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-2 px-3 text-gray-400 font-medium">Диапазон Прогресса</th>
                  <th className="text-right py-2 px-3 text-gray-400 font-medium">Заявлено</th>
                  <th className="text-right py-2 px-3 text-gray-400 font-medium">Выполнено</th>
                  <th className="text-right py-2 px-3 text-gray-400 font-medium">Разрыв</th>
                  <th className="text-right py-2 px-3 text-gray-400 font-medium">Честность</th>
                </tr>
              </thead>
              <tbody>
                {confidenceBins.map(bin => {
                  const integrity = bin.reported > 0 ? (bin.actual / bin.reported * 100) : 100;

                  return (
                    <tr key={bin.range} className="border-b border-gray-800/50">
                      <td className="py-2 px-3 text-gray-300 font-mono">{bin.range}</td>
                      <td className="py-2 px-3 text-right text-gray-400">{bin.reported}</td>
                      <td className="py-2 px-3 text-right">
                        <span className={bin.actual > 0 ? "text-green-400" : "text-gray-500"}>
                          {bin.actual}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right">
                        <span className={`font-mono ${bin.gap > 0 ? "text-red-400" : "text-green-400"}`}>
                          {bin.gap > 0 ? `+${bin.gap}` : bin.gap}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-right">
                        <span className={`font-mono text-xs ${integrity > 80 ? 'text-green-400' : integrity > 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                          {integrity.toFixed(0)}%
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Insights */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Eye className="text-blue-400" size={16} />
          <h4 className="text-sm font-semibold text-gray-300">Инсайты Честности</h4>
        </div>

        <div className="space-y-3">
          {loading ? (
            <div className="text-center py-4 text-gray-500">Анализ...</div>
          ) : (
            <>
              {getIntegrityScore() > 90 ? (
                <div className="p-3 bg-green-900/10 border border-green-900/50 rounded">
                  <div className="text-xs text-green-400">
                    ✓ Отличная калибровка: Прогнозы системы хорошо соответствуют действительности
                  </div>
                </div>
              ) : getIntegrityScore() > 70 ? (
                <div className="p-3 bg-yellow-900/10 border border-yellow-900/50 rounded">
                  <div className="text-xs text-yellow-400">
                    ⚠ Умеренная калибровка: Наблюдается некоторый разрыв между прогнозами и результатами
                  </div>
                </div>
              ) : (
                <div className="p-3 bg-red-900/10 border border-red-900/50 rounded">
                  <div className="text-xs text-red-400">
                    ✗ Poor калибровка: Систематически завышенные ожидания. Рекомендуется пересмотр оценок.
                  </div>
                </div>
              )}

              <div className="p-3 bg-blue-900/10 border border-blue-900/50 rounded">
                <div className="text-xs text-blue-400">
                  <div className="font-mono mb-1">МЕТОДИКА:</div>
                  Честность = (Выполненные / Заявленные) × 100% для каждого диапазона прогресса
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
