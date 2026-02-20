/**
 * Execution Log Store
 *
 * Manages execution event history
 */

import { create } from 'zustand';
import { LogEntry } from '../components/execution/ExecutionLog';

interface ExecutionLogStore {
  logs: LogEntry[];
  add: (entry: Omit<LogEntry, 'id' | 'timestamp'>) => void;
  clear: () => void;
}

export const useExecutionLogStore = create<ExecutionLogStore>((set) => ({
  logs: [],

  add: (entry) => {
    const id = Math.random().toString(36).substring(7);
    const newEntry: LogEntry = {
      ...entry,
      id,
      timestamp: new Date(),
    };

    set((state) => ({
      logs: [...state.logs, newEntry],
    }));
  },

  clear: () => {
    set({ logs: [] });
  },
}));
