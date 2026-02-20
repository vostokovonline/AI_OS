/**
 * Observability Console - Read-Only System Truth Interface
 *
 * Global UI Invariants:
 * - NO buttons: apply, approve, tune, retry
 * - NO overall scores: health, intelligence, success
 * - NO color judgments: good/bad - only tension, gaps, growth
 * - All aggregates drill-down to raw events
 * - Each screen answers ONE question
 */

import { useState } from 'react';
import { Activity, AlertTriangle, Eye, Clock, Shield } from 'lucide-react';

import { SystemRealityOverview } from './screens/SystemRealityOverview';
import { ErrorSurface } from './screens/ErrorSurface';
import { AlertIRLMonitor } from './screens/AlertIRLMonitor';
import { CausalTimeline } from './screens/CausalTimeline';
import { SelfEvalIntegrity } from './screens/SelfEvalIntegrity';

type Screen = 'reality' | 'errors' | 'alerts' | 'timeline' | 'integrity';

interface ScreenConfig {
  id: Screen;
  title: string;
  description: string;
  icon: React.ReactNode;
  question: string;
}

const SCREENS: ScreenConfig[] = [
  {
    id: 'reality',
    title: 'Реальность Системы',
    description: 'Что система знает о себе сейчас',
    icon: <Activity size={16} />,
    question: 'Что система о себе знает сейчас?'
  },
  {
    id: 'errors',
    title: 'Поверхность Ошибок',
    description: 'Где ошибки неслучайны',
    icon: <AlertTriangle size={16} />,
    question: 'Где ошибки неслучайны?'
  },
  {
    id: 'alerts',
    title: 'Тревоги и IRL',
    description: 'Где система просит внимания',
    icon: <Eye size={16} />,
    question: 'Где система сама просит быть замеченной?'
  },
  {
    id: 'timeline',
    title: 'Каузальная Временная Шкала',
    description: 'Как разворачивались события',
    icon: <Clock size={16} />,
    question: 'Как на самом деле развивались события?'
  },
  {
    id: 'integrity',
    title: 'Самооценка Честности',
    description: 'Насколько система честна с собой',
    icon: <Shield size={16} />,
    question: 'Насколько система честна с собой?'
  }
];

export function ObservabilityConsole() {
  const [activeScreen, setActiveScreen] = useState<Screen>('timeline');

  const currentScreen = SCREENS.find(s => s.id === activeScreen);

  return (
    <div className="h-full flex bg-gray-950">
      {/* Sidebar Navigation */}
      <div className="w-72 bg-gray-900 border-r border-gray-800 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-800">
          <div className="flex items-center gap-2 mb-2">
            <Eye className="text-blue-400" size={20} />
            <h1 className="text-lg font-semibold text-gray-100">Наблюдаемость</h1>
          </div>
          <p className="text-xs text-gray-500">Интерфейс наблюдения за системой</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {SCREENS.map(screen => (
            <button
              key={screen.id}
              onClick={() => setActiveScreen(screen.id)}
              className={`
                w-full text-left px-3 py-2.5 rounded-lg transition-colors
                flex items-center gap-3 group
                ${activeScreen === screen.id
                  ? 'bg-blue-900/20 text-blue-400 border border-blue-900/50'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200 border border-transparent'
                }
              `}
            >
              <div className={activeScreen === screen.id ? 'text-blue-400' : 'text-gray-500 group-hover:text-gray-400'}>
                {screen.icon}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{screen.title}</div>
                <div className="text-xs text-gray-500 truncate">{screen.description}</div>
              </div>
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-3 border-t border-gray-800">
          <div className="text-xs text-gray-600">
            <div className="font-mono">ТОЛЬКО ЧТЕНИЕ</div>
            <div className="font-mono">БЕЗ УПРАВЛЕНИЯ</div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Screen Header */}
        <div className="bg-gray-900 border-b border-gray-800 p-4">
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-100 flex items-center gap-2">
                {currentScreen?.icon}
                {currentScreen?.title}
              </h2>
              <p className="text-sm text-gray-500 mt-1">{currentScreen?.question}</p>
            </div>
          </div>
        </div>

        {/* Screen Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeScreen === 'reality' && <SystemRealityOverview />}
          {activeScreen === 'errors' && <ErrorSurface />}
          {activeScreen === 'alerts' && <AlertIRLMonitor />}
          {activeScreen === 'timeline' && <CausalTimeline />}
          {activeScreen === 'integrity' && <SelfEvalIntegrity />}
        </div>
      </div>
    </div>
  );
}
