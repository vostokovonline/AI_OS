/**
 * Floating Label for Nodes
 *
 * Shows status and progress above nodes during execution
 */

import React, { useEffect, useState } from 'react';
import { Loader, CheckCircle, XCircle, Clock } from 'lucide-react';

interface FloatingLabelProps {
  status: 'pending' | 'active' | 'done' | 'blocked' | 'failed';
  title?: string;
  progress?: number;
  duration?: number;
  visible?: boolean;
}

const FloatingLabel: React.FC<FloatingLabelProps> = ({
  status,
  title,
  progress,
  duration,
  visible = true,
}) => {
  const [show, setShow] = useState(visible);

  useEffect(() => {
    setShow(visible);
  }, [visible]);

  if (!show || status === 'pending') return null;

  const getLabelContent = () => {
    switch (status) {
      case 'active':
        return {
          icon: <Loader size={14} className="text-blue-400 spinner" />,
          text: title || 'Executing...',
          bgColor: 'bg-blue-900/95',
          borderColor: 'border-blue-500',
          textColor: 'text-white',
        };
      case 'done':
        return {
          icon: <CheckCircle size={14} className="text-green-400" />,
          text: title || 'Completed',
          bgColor: 'bg-green-900/95',
          borderColor: 'border-green-500',
          textColor: 'text-white',
        };
      case 'blocked':
        return {
          icon: <XCircle size={14} className="text-red-400" />,
          text: title || 'Blocked',
          bgColor: 'bg-red-900/95',
          borderColor: 'border-red-500',
          textColor: 'text-white',
        };
      case 'failed':
        return {
          icon: <XCircle size={14} className="text-red-400" />,
          text: title || 'Failed',
          bgColor: 'bg-red-900/95',
          borderColor: 'border-red-500',
          textColor: 'text-white',
        };
      default:
        return null;
    }
  };

  const content = getLabelContent();
  if (!content) return null;

  return (
    <div
      className={`
        absolute -top-16 left-1/2 transform -translate-x-1/2
        px-3 py-2 rounded-lg border-2 shadow-lg whitespace-nowrap
        ${content.bgColor} ${content.borderColor}
        fade-in z-50 min-w-[150px] pointer-events-none
      `}
    >
      <div className="flex items-center gap-2">
        {content.icon}
        <span className={`text-xs font-medium ${content.textColor}`}>
          {content.text}
        </span>
      </div>

      {/* Progress bar for active status */}
      {status === 'active' && progress !== undefined && (
        <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
          <div
            className="bg-blue-500 h-1 rounded-full transition-all duration-300 progress-animated"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      )}

      {/* Duration for done status */}
      {status === 'done' && duration && (
        <div className="flex items-center gap-1 mt-1 text-gray-300 text-xs">
          <Clock size={10} />
          <span>{duration}ms</span>
        </div>
      )}

      {/* Arrow pointing down */}
      <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-8 border-l-transparent border-r-8 border-r-transparent border-t-8" />
    </div>
  );
};

export default FloatingLabel;
