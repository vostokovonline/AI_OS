/**
 * v2 UI - Main App Component
 *
 * Root component for AI-OS v2 Dashboard
 * Orchestrates all panels and manages global state
 */

import React, { useEffect, useState } from 'react';
import { useUIStore } from './store/uiStore';
import { apiClient } from './api/client';
import ControlPanel from './components/controls/ControlPanel';
import GraphCanvas from './components/canvas/GraphCanvas';
import HistoricalTimeline from './components/timeline/HistoricalTimeline';
import DependencyTree from './components/tree/DependencyTree';
import InspectorPanel from './components/inspector/InspectorPanel';
// import TimelineStrip from './components/timeline/TimelineStrip'; // DISABLED: functionality in Gantt
import ToastContainer from './components/toast/ToastContainer';
import ExecutionLog from './components/execution/ExecutionLog';
import EmotionalDashboard from './components/emotional/EmotionalDashboard';
import { ObservabilityConsole } from './components/observability/ObservabilityConsole';
import { QuestionsScreen } from "./components/questions/QuestionsScreen";
import { DecompositionScreen } from "./components/decomposition/DecompositionScreen";
import { Skills } from "./pages/Skills";
import { Deployments } from "./pages/Deployments";
import { Observability } from "./pages/Observability";
import { Federation } from "./pages/Federation";
import { Artifacts } from "./pages/Artifacts";
import Autonomy from "./pages/Autonomy";
import Admin from "./pages/Admin";
import Decision from "./pages/Decision";
import VField from "./pages/VField";
import LLMAnalytics from "./pages/LLMAnalytics";
import SystemHealth from "./pages/SystemHealth";
import Performance from "./pages/Performance";
import UnifiedChat from "./pages/UnifiedChat";
import { LLMControlCenter } from "./pages/LLMControlCenter";
import ControlCenter from "./pages/ControlCenter";
import Evolution from "./pages/Evolution";
import Goals from "./pages/Goals";
import PlanMemory from "./pages/PlanMemory";
import TraceTimeline from "./pages/TraceTimeline";
import Capabilities from "./pages/Capabilities";
// import QuestionsViewTest from './components/questions/QuestionsViewTest';
import { useExecutionLogStore } from './store/executionLogStore';
import { X } from 'lucide-react';

const App: React.FC = () => {
  const { handleSystemEvent, view } = useUIStore();
  const logs = useExecutionLogStore((state) => state.logs);
  const [showEmotionalLayer, setShowEmotionalLayer] = useState(false);

  // Subscribe to server-sent events for real-time updates
  useEffect(() => {
    const eventSource = apiClient.subscribeToUpdates((event) => {
      handleSystemEvent(event);
    });

    return () => {
      eventSource.close();
    };
  }, [handleSystemEvent]);

  return (
    <div className="h-screen w-screen bg-gray-900 flex flex-col overflow-hidden">
      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Control Panel - ALWAYS VISIBLE */}
        <div className="flex flex-col">
          <ControlPanel onToggleEmotionalLayer={() => setShowEmotionalLayer(!showEmotionalLayer)} />
          <div className="flex-1 p-4 overflow-hidden">
            <ExecutionLog logs={logs} />
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 relative">
          {/* OCCP Pages - Fixed backgrounds for dark theme */}
          <div className={view === 'skills' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Skills />
          </div>
          <div className={view === 'deployments' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Deployments />
          </div>
          <div className={view === 'occp-observability' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Observability />
          </div>
          <div className={view === 'federation' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Federation />
          </div>
          <div className={view === 'artifacts' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Artifacts />
          </div>
          <div className={view === 'autonomy' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Autonomy />
          </div>
          <div className={view === 'admin' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Admin />
          </div>
          <div className={view === 'decision' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Decision />
          </div>
          <div className={view === 'vfield' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <VField />
          </div>

          {/* New Analytics Pages - Fixed backgrounds */}
          <div className={view === 'llm-analytics' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <LLMAnalytics />
          </div>
          <div className={view === 'system-health' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <SystemHealth />
          </div>
          <div className={view === 'performance' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Performance />
          </div>
          <div className={view === 'unified-chat' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <UnifiedChat />
          </div>
          <div className={view === 'llm-control' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <LLMControlCenter />
          </div>

          {/* Control Center - Metrics Dashboard */}
          <div className={view === 'control-center' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <ControlCenter />
          </div>

          {/* Evolution Dashboard - Self-Evolving OS */}
          <div className={view === 'evolution' ? 'absolute inset-0 bg-gray-100 overflow-auto' : 'hidden'}>
            <Evolution />
          </div>

          {/* Goals Page - Goal Hierarchy & Execution Tracking */}
          <div className={view === 'goals' ? 'absolute inset-0 bg-gray-900 overflow-hidden' : 'hidden'}>
            <Goals />
          </div>

          {/* Plan Memory - Hierarchical MAB */}
          <div className={view === 'plan-memory' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <PlanMemory />
          </div>

          {/* Trace Timeline - Execution Traces */}
          <div className={view === 'trace-timeline' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <TraceTimeline />
          </div>

          {/* Capabilities - UCB1 Selector */}
          <div className={view === 'capabilities' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <Capabilities />
          </div>
          <div className={view === 'vfield' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <VField />
          </div>

          {/* Observability Console */}
          <div className={view === 'observability' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <ObservabilityConsole />
          </div>

          {/* Questions Screen */}
          <div className={view === 'questions' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <QuestionsScreen />
          </div>

          {/* Decomposition Screen */}
          <div className={view === 'decomposition' ? 'absolute inset-0 bg-gray-900 overflow-auto' : 'hidden'}>
            <DecompositionScreen />
          </div>

          {/* Graph/Gantt/Tree Views */}
          <div className={view !== 'observability' && view !== 'questions' && view !== 'decomposition' && view !== 'skills' && view !== 'deployments' && view !== 'occp-observability' && view !== 'artifacts' && view !== 'autonomy' && view !== 'admin' && view !== 'decision' && view !== 'vfield' && view !== 'llm-analytics' && view !== 'system-health' && view !== 'performance' && view !== 'unified-chat' && view !== 'llm-control' && view !== 'evolution' && view !== 'goals' && view !== 'control-center' && view !== 'plan-memory' && view !== 'trace-timeline' && view !== 'capabilities' ? 'absolute inset-0 flex' : 'hidden'}>
            {/* Center Canvas */}
            <div className="flex-1 flex flex-col">
              {view === 'graph' && <GraphCanvas />}
              {view === 'gantt' && <HistoricalTimeline />}
              {view === 'tree' && <DependencyTree />}
            </div>

            {/* Right Inspector Panel or Emotional Layer */}
            {showEmotionalLayer ? (
              <div className="w-96 bg-gray-900 border-l border-gray-700 flex flex-col">
                <div className="p-4 border-b border-gray-700 flex items-center justify-between">
                  <h2 className="text-white font-bold text-lg">Emotional Layer</h2>
                  <button
                    onClick={() => setShowEmotionalLayer(false)}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <X size={20} />
                  </button>
                </div>
                <div className="flex-1 overflow-hidden">
                  <EmotionalDashboard />
                </div>
              </div>
            ) : (
              <InspectorPanel />
            )}
          </div>
        </div>
      </div>

      {/* Bottom Timeline - DISABLED: functionality available in Gantt view */}
      {/* {view !== 'observability' && <TimelineStrip />} */}

      {/* Toast Notifications */}
      <ToastContainer />
    </div>
  );
};

export default App;
