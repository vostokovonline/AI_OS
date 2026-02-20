/**
 * Screen 2: Error Surface
 *
 * Question: Where are errors non-random?
 * Frequency: 1-2 times per week
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, Grid3x3 } from 'lucide-react';
import { apiClient } from '../../../api/client';

interface ErrorMatrix {
  [key: string]: {
    [key: string]: number;
  };
}

export function ErrorSurface() {
  const [errorMatrix, setErrorMatrix] = useState<ErrorMatrix>({});
  const [totalErrors, setTotalErrors] = useState(0);
  const [errorByGoalType, setErrorByGoalType] = useState<{[key: string]: number}>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      // Load goals and analyze errors
      const goals = await apiClient.fetchV1Goals();

      // Build error matrix: error_status × goal_type
      const matrix: ErrorMatrix = {};
      const byType: {[key: string]: number} = {};
      let errorCount = 0;

      goals.forEach((goal: any) => {
        const goalType = goal.goal_type || 'unknown';
        const status = goal.status;

        // Count failed/blocked as errors
        if (status === 'failed' || status === 'blocked') {
          errorCount++;

          // Count by goal type
          if (!byType[goalType]) {
            byType[goalType] = 0;
          }
          byType[goalType]++;

          // Build matrix
          if (!matrix[status]) {
            matrix[status] = {};
          }
          if (!matrix[status][goalType]) {
            matrix[status][goalType] = 0;
          }
          matrix[status][goalType]++;
        }
      });

      setErrorMatrix(matrix);
      setTotalErrors(errorCount);
      setErrorByGoalType(byType);
    } catch (error) {
      console.error('Failed to load error surface:', error);
    } finally {
      setLoading(false);
    }
  };

  const errorStatuses = ['failed', 'blocked'];
  const goalTypes = ['achievable', 'continuous', 'directional', 'exploratory', 'unknown'];

  const getDominantError = () => {
    let max = 0;
    let dominant = 'Нет данных';
    Object.entries(errorByGoalType).forEach(([type, count]) => {
      if (count > max) {
        max = count;
        dominant = type;
      }
    });
    return dominant;
  };

  return (
    <div className="space-y-6">
      {/* Error Taxonomy Matrix */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <Grid3x3 className="text-red-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Матрица Ошибок по Типам Целей</h3>
        </div>

        <div className="text-xs text-gray-500 mb-4">
          Распределение ошибок: Статус × Тип Цели (количество)
        </div>

        {loading ? (
          <div className="text-center py-8 text-gray-500">Загрузка данных...</div>
        ) : (
          <>
            {/* Summary */}
            <div className="mb-4 p-3 bg-red-900/10 border border-red-900/50 rounded">
              <div className="text-sm text-red-400">
                Всего обнаружено ошибок: <span className="font-mono font-bold">{totalErrors}</span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left py-2 px-3 text-gray-400 font-medium">Статус Ошибки</th>
                    {goalTypes.map(type => (
                      <th key={type} className="text-right py-2 px-3 text-gray-400 font-medium capitalize">
                        {type}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {errorStatuses.map(errorStatus => (
                    <tr key={errorStatus} className="border-b border-gray-800/50">
                      <td className="py-2 px-3 text-gray-300 text-xs font-mono capitalize">
                        {errorStatus === 'failed' ? '❌ Failed' : '🚫 Blocked'}
                      </td>
                      {goalTypes.map(goalType => {
                        const count = errorMatrix[errorStatus]?.[goalType] || 0;
                        return (
                          <td key={goalType} className="py-2 px-3 text-right">
                            {count > 0 ? (
                              <span className="text-red-400 font-mono">{count}</span>
                            ) : (
                              <span className="text-gray-700">-</span>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      {/* Error × Goal Type Analysis */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-100 mb-4">Ошибки × Типы Целей</h3>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left py-2 px-3 text-gray-400 font-medium">Тип Цели</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Количество Ошибок</th>
                <th className="text-center py-2 px-3 text-gray-400 font-medium">Тренд</th>
              </tr>
            </thead>
            <tbody>
              {goalTypes.map(goalType => {
                const count = errorByGoalType[goalType] || 0;
                const percentage = totalErrors > 0 ? (count / totalErrors * 100).toFixed(1) : '0.0';

                return (
                  <tr key={goalType} className="border-b border-gray-800/50">
                    <td className="py-2 px-3 text-gray-300 capitalize">{goalType}</td>
                    <td className="py-2 px-3 text-right">
                      <span className={count > 0 ? "text-red-400 font-mono" : "text-gray-600"}>
                        {count}
                      </span>
                      <span className="text-xs text-gray-600 ml-2">({percentage}%)</span>
                    </td>
                    <td className="py-2 px-3 text-center">
                      {count === 0 ? (
                        <span className="text-gray-600">-</span>
                      ) : count > 5 ? (
                        <span className="text-red-400 text-xs">⬆︎ Высокий</span>
                      ) : (
                        <span className="text-yellow-400 text-xs">◐ Средний</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Repeatability Index */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="text-yellow-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Анализ Повторяемости</h3>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-950 border border-gray-800 rounded p-4">
            <div className="text-xs text-gray-500 mb-1">Доминирующий тип с ошибками</div>
            <div className="text-lg font-mono text-gray-200 capitalize">
              {loading ? '...' : getDominantError()}
            </div>
          </div>

          <div className="bg-gray-950 border border-gray-800 rounded p-4">
            <div className="text-xs text-gray-500 mb-1">Всего типов с ошибками</div>
            <div className="text-lg font-mono text-gray-200">
              {loading ? '...' : Object.keys(errorByGoalType).filter(k => errorByGoalType[k] > 0).length}
            </div>
          </div>
        </div>

        <div className="mt-4 p-3 bg-yellow-900/10 border border-yellow-900/50 rounded text-xs text-yellow-400">
          <div className="font-mono mb-1">ВАЖНЫЙ ИНСАЙТ:</div>
          {totalErrors === 0 ? (
            <span className="text-green-400">✓ Систематические ошибки не обнаружены</span>
          ) : (
            <>
              Обнаружены ошибки в <span className="font-mono">{totalErrors}</span> целях.
              Рекомендуется анализ доминирующего типа: <span className="font-mono capitalize">{getDominantError()}</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
