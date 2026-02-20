/**
 * Toast Notification Store
 *
 * Manages toast notifications
 */

import { create } from 'zustand';
import { Toast } from '../components/toast/ToastContainer';

interface ToastStore {
  toasts: Toast[];
  add: (toast: Omit<Toast, 'id' | 'timestamp'>) => void;
  remove: (id: string) => void;
  clear: () => void;
}

export const useToastStore = create<ToastStore>((set) => ({
  toasts: [],

  add: (toast) => {
    const id = Math.random().toString(36).substring(7);
    const newToast: Toast = {
      ...toast,
      id,
      timestamp: new Date(),
      duration: toast.duration ?? 5000,
    };

    set((state) => ({
      toasts: [...state.toasts, newToast],
    }));

    return id;
  },

  remove: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },

  clear: () => {
    set({ toasts: [] });
  },
}));
