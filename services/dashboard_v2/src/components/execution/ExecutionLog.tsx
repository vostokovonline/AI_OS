/**
 * Execution Log Panel
 *
 * Shows real-time execution events and history
 */

import React from 'react';
import { Clock, CheckCircle, XCircle, Loader, Play } from 'lucide-react';

export interface LogEntry {
  id: string;
  timestamp: Date;
  type: 'start' | 'progress' | 'complete' | 'error';
  nodeName: string;
  nodeId: string;
  message: string;
  duration?: number;
}

interface ExecutionLogProps {
  logs: LogEntry[];
  maxSize?: number;
}

const ExecutionLog: React.FC<ExecutionLogProps> = ({ logs, maxSize = 50 }) => {
  const displayLogs = logs.slice(-maxSize).reverse();

  const getIcon = (type: LogEntry['type']) => {
    switch (type) {
      case 'start':
        return <Play size={14} className="text-blue-400" />;
      case 'progress':
        return <Loader size={14} className="text-yellow-400 spinner" />;
      case 'complete':
        return <CheckCircle size={14} className="text-green-400" />;
      case 'error':
        return <XCircle size={14} className="text-red-400" />;
    }
  };

  const getBgClass = (type: LogEntry['type']) => {
    switch (type) {
      case 'start':
        return 'bg-blue-900/20 border-l-blue-500';
      case 'progress':
        return 'bg-yellow-900/20 border-l-yellow-500';
      case 'complete':
        return 'bg-green-900/20 border-l-green-500';
      case 'error':
        return 'bg-red-900/20 border-l-red-500';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-white font-bold text-sm flex items-center gap-2">
          <Clock size={16} className="text-gray-400" />
          Execution Log
        </h3>
        <span className="text-gray-400 text-xs">{displayLogs.length} events</span>
      </div>

      {/* Log entries */}
      <div className="space-y-2 max-h-64 overflow-y-auto">
        {displayLogs.length === 0 ? (
          <div className="text-gray-500 text-sm text-center py-4">
            No execution events yet
          </div>
        ) : (
          displayLogs.map((log) => (
            <div
              key={log.id}
              className={`
                px-3 py-2 rounded border-l-2
                ${getBgClass(log.type)}
              `}
            >
              <div className="flex items-start gap-2">
                <div className="flex-shrink-0 mt-0.5">{getIcon(log.type)}</div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-white text-sm font-medium truncate">
                      {log.nodeName}
                    </span>
                    <span className="text-gray-400 text-xs flex-shrink-0">
                      {log.timestamp.toLocaleTimeString()}
                    </span>
                  </div>

                  <p className="text-gray-300 text-xs mt-1">{log.message}</p>

                  {log.duration && (
                    <span className="text-gray-400 text-xs">
                      Duration: {log.duration}ms
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ExecutionLog;
