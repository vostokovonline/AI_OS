/**
 * Toast Notifications
 *
 * Real-time notifications for system events
 */

import React, { useEffect } from 'react';
import { X, CheckCircle, AlertCircle, Loader, Info } from 'lucide-react';
import { useToastStore } from '../../store/toastStore';

export interface Toast {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'executing';
  title: string;
  message?: string;
  duration?: number;
  timestamp: Date;
}

const ToastIcon: React.FC<{ type: Toast['type'] }> = ({ type }) => {
  switch (type) {
    case 'success':
      return <CheckCircle size={20} className="text-green-400" />;
    case 'error':
      return <AlertCircle size={20} className="text-red-400" />;
    case 'warning':
      return <AlertCircle size={20} className="text-yellow-400" />;
    case 'executing':
      return <Loader size={20} className="text-blue-400 spinner" />;
    default:
      return <Info size={20} className="text-blue-400" />;
  }
};

const ToastItem: React.FC<{
  toast: Toast;
  onRemove: (id: string) => void;
}> = ({ toast, onRemove }) => {
  useEffect(() => {
    if (toast.duration && toast.duration > 0) {
      const timer = setTimeout(() => onRemove(toast.id), toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.id, toast.duration, onRemove]);

  const getBackgroundClass = () => {
    switch (toast.type) {
      case 'success':
        return 'bg-green-900/90 border-green-500';
      case 'error':
        return 'bg-red-900/90 border-red-500';
      case 'warning':
        return 'bg-yellow-900/90 border-yellow-500';
      case 'executing':
        return 'bg-blue-900/90 border-blue-500';
      default:
        return 'bg-gray-800/90 border-gray-600';
    }
  };

  return (
    <div
      className={`
        flex items-start gap-3 px-4 py-3 rounded-lg border-l-4 shadow-lg
        ${getBackgroundClass()}
        slide-in max-w-md
      `}
    >
      <div className="flex-shrink-0 mt-0.5">
        <ToastIcon type={toast.type} />
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <h4 className="text-white font-medium text-sm">{toast.title}</h4>
          <button
            onClick={() => onRemove(toast.id)}
            className="flex-shrink-0 text-gray-400 hover:text-white transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {toast.message && (
          <p className="text-gray-300 text-sm mt-1">{toast.message}</p>
        )}

        <div className="text-xs text-gray-400 mt-1">
          {toast.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

const ToastContainer: React.FC = () => {
  const { toasts, remove } = useToastStore();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItem toast={toast} onRemove={remove} />
        </div>
      ))}
    </div>
  );
};

export default ToastContainer;
