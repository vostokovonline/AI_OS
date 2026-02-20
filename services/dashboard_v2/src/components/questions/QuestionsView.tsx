/**
 * v2 UI - Questions View
 *
 * Displays pending questions from the AI-OS system
 * Allows users to answer, use default, or dismiss questions
 */

import React, { useEffect, useState } from 'react';
import { useQuestionsStore, Question } from '../../store/questionsStore';
import {
  MessageCircle,
  AlertCircle,
  AlertTriangle,
  Clock,
  Send,
  RefreshCw,
  Filter,
  Trash2,
  SkipForward,
} from 'lucide-react';

const QuestionsView: React.FC = () => {
  const {
    questions,
    stats,
    loading,
    error,
    selectedGoalId,
    fetchQuestions,
    fetchStats,
    answerQuestion,
    dismissQuestion,
    setSelectedGoalId,
  } = useQuestionsStore();

  const [answerInputs, setAnswerInputs] = useState<Record<string, string>>({});
  const [answeringQuestions, setAnsweringQuestions] = useState<Record<string, boolean>>({});
  const [showGoalFilter, setShowGoalFilter] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    if (!isInitialized) {
      fetchQuestions().catch(console.error);
      fetchStats().catch(console.error);
      setIsInitialized(true);
    }
  }, [isInitialized]);

  const handleAnswer = async (question: Question) => {
    const answer = answerInputs[question.artifact_id];
    if (!answer || !answer.trim()) {
      return;
    }

    setAnsweringQuestions((prev) => ({ ...prev, [question.artifact_id]: true }));

    try {
      await answerQuestion(question.artifact_id, answer);
      setAnswerInputs((prev) => ({ ...prev, [question.artifact_id]: '' }));
    } finally {
      setAnsweringQuestions((prev) => ({ ...prev, [question.artifact_id]: false }));
    }
  };

  const handleUseDefault = async (question: Question) => {
    if (!question.default_answer) return;

    setAnsweringQuestions((prev) => ({ ...prev, [question.artifact_id]: true }));

    try {
      await answerQuestion(question.artifact_id, question.default_answer, true);
    } finally {
      setAnsweringQuestions((prev) => ({ ...prev, [question.artifact_id]: false }));
    }
  };

  const handleDismiss = async (question: Question) => {
    setAnsweringQuestions((prev) => ({ ...prev, [question.artifact_id]: true }));

    try {
      await dismissQuestion(question.artifact_id);
    } finally {
      setAnsweringQuestions((prev) => ({ ...prev, [question.artifact_id]: false }));
    }
  };

  const getPriorityInfo = (priority: string) => {
    switch (priority) {
      case 'critical':
        return {
          color: 'red',
          bgColor: 'bg-red-900/20',
          borderColor: 'border-red-500',
          textColor: 'text-red-400',
          icon: AlertCircle,
        };
      case 'high':
        return {
          color: 'orange',
          bgColor: 'bg-orange-900/20',
          borderColor: 'border-orange-500',
          textColor: 'text-orange-400',
          icon: AlertTriangle,
        };
      case 'low':
        return {
          color: 'gray',
          bgColor: 'bg-gray-700/50',
          borderColor: 'border-gray-500',
          textColor: 'text-gray-400',
          icon: MessageCircle,
        };
      default:
        return {
          color: 'green',
          bgColor: 'bg-green-900/20',
          borderColor: 'border-green-500',
          textColor: 'text-green-400',
          icon: MessageCircle,
        };
    }
  };

  const getTimeUntilTimeout = (timeoutAt: string) => {
    const now = new Date();
    const timeout = new Date(timeoutAt);
    const diff = timeout.getTime() - now.getTime();

    if (diff <= 0) return 'Expired';

    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;

    if (hours > 0) {
      return `${hours}h ${remainingMinutes}m`;
    }
    return `${minutes}m`;
  };

  return (
    <div className="h-full w-full bg-gray-900 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-white font-bold text-xl flex items-center gap-2">
            <MessageCircle size={24} className="text-blue-400" />
            Questions from System
          </h2>
          <button
            onClick={() => {
              fetchQuestions();
              fetchStats();
            }}
            disabled={loading}
            className="text-gray-400 hover:text-white transition-colors disabled:opacity-50"
          >
            <RefreshCw size={20} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>

        {/* Statistics */}
        {stats && (
          <div className="grid grid-cols-4 gap-3 mb-4">
            <div className="bg-gray-800 p-3 rounded border border-gray-700">
              <div className="text-2xl font-bold text-white">{stats.pending_count}</div>
              <div className="text-xs text-gray-400">Pending Questions</div>
            </div>
            <div className="bg-red-900/20 p-3 rounded border border-red-900/50">
              <div className="text-2xl font-bold text-red-400 flex items-center gap-1">
                <AlertCircle size={16} />
                {stats.priority_breakdown.critical}
              </div>
              <div className="text-xs text-gray-400">Critical</div>
            </div>
            <div className="bg-orange-900/20 p-3 rounded border border-orange-900/50">
              <div className="text-2xl font-bold text-orange-400 flex items-center gap-1">
                <AlertTriangle size={16} />
                {stats.priority_breakdown.high}
              </div>
              <div className="text-xs text-gray-400">High</div>
            </div>
            <div className="bg-green-900/20 p-3 rounded border border-green-900/50">
              <div className="text-2xl font-bold text-green-400 flex items-center gap-1">
                <MessageCircle size={16} />
                {stats.priority_breakdown.normal}
              </div>
              <div className="text-xs text-gray-400">Normal</div>
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setShowGoalFilter(!showGoalFilter)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded text-sm transition-colors ${
              selectedGoalId ? 'bg-blue-900/50 text-blue-400' : 'bg-gray-700 text-gray-300'
            }`}
          >
            <Filter size={16} />
            Filter by Goal
          </button>

          {showGoalFilter && (
            <input
              type="text"
              placeholder="Enter Goal ID..."
              value={selectedGoalId || ''}
              onChange={(e) => setSelectedGoalId(e.target.value || null)}
              className="bg-gray-700 text-white px-3 py-1.5 rounded text-sm border border-gray-600 focus:border-blue-500 focus:outline-none"
            />
          )}

          {selectedGoalId && (
            <button
              onClick={() => setSelectedGoalId(null)}
              className="text-gray-400 hover:text-white text-sm"
            >
              Clear filter
            </button>
          )}

          <div className="ml-auto text-gray-400 text-sm">
            {questions.length} question{questions.length !== 1 ? 's' : ''}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mt-4 p-3 bg-red-900/20 border border-red-500 rounded text-red-400 text-sm flex items-center gap-2">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Questions List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {questions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <MessageCircle size={48} className="mb-3 opacity-50" />
            <p className="text-lg">No pending questions</p>
            <p className="text-sm mt-1">Questions from the system will appear here</p>
          </div>
        ) : (
          questions.map((question) => {
            const priorityInfo = getPriorityInfo(question.priority);
            const PriorityIcon = priorityInfo.icon;
            const isAnswering = answeringQuestions[question.artifact_id];
            const answer = answerInputs[question.artifact_id] || '';

            return (
              <div
                key={question.artifact_id}
                className={`bg-gray-800 rounded-lg border-l-4 ${priorityInfo.borderColor} overflow-hidden`}
              >
                {/* Question Header */}
                <div className={`p-3 ${priorityInfo.bgColor} border-b border-gray-700`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <PriorityIcon size={18} className={priorityInfo.textColor} />
                      <span className={`text-xs font-semibold uppercase ${priorityInfo.textColor}`}>
                        {question.priority}
                      </span>
                      <span className="text-gray-500 text-xs">•</span>
                      <span className="text-gray-400 text-xs font-mono">
                        {question.goal_id.slice(0, 8)}...
                      </span>
                    </div>
                    <div className="flex items-center gap-1 text-gray-400 text-xs">
                      <Clock size={14} />
                      <span>{getTimeUntilTimeout(question.timeout_at)}</span>
                    </div>
                  </div>
                </div>

                {/* Question Content */}
                <div className="p-4 space-y-3">
                  {/* Question */}
                  <div>
                    <h3 className="text-white font-medium text-sm mb-1">Question:</h3>
                    <p className="text-gray-300 text-sm">{question.question}</p>
                  </div>

                  {/* Context */}
                  {question.context && (
                    <div>
                      <h3 className="text-gray-400 text-xs uppercase mb-1">Context:</h3>
                      <p className="text-gray-400 text-xs">{question.context}</p>
                    </div>
                  )}

                  {/* Options (if provided) */}
                  {question.options && question.options.length > 0 && (
                    <div>
                      <h3 className="text-gray-400 text-xs uppercase mb-2">Options:</h3>
                      <div className="grid grid-cols-2 gap-2">
                        {question.options.map((option, idx) => (
                          <button
                            key={idx}
                            onClick={() =>
                              setAnswerInputs((prev) => ({
                                ...prev,
                                [question.artifact_id]: option,
                              }))
                            }
                            className="bg-gray-700 hover:bg-gray-600 text-gray-300 text-xs px-3 py-2 rounded transition-colors text-left"
                          >
                            {option}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Timeout Info */}
                  <div className="bg-gray-700/50 p-2 rounded text-xs">
                    <div className="flex items-center justify-between text-gray-400">
                      <span>Timeout action:</span>
                      <span className="text-gray-300">{question.timeout_action}</span>
                    </div>
                    {question.default_answer && (
                      <div className="flex items-center justify-between text-gray-400 mt-1">
                        <span>Default answer:</span>
                        <span className="text-gray-300 truncate ml-2">
                          {question.default_answer}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Answer Input */}
                  <div>
                    <label className="text-gray-400 text-xs uppercase mb-2 block">Your Answer:</label>
                    <textarea
                      value={answer}
                      onChange={(e) =>
                        setAnswerInputs((prev) => ({
                          ...prev,
                          [question.artifact_id]: e.target.value,
                        }))
                      }
                      placeholder="Type your answer here..."
                      className="w-full bg-gray-700 text-white text-sm p-3 rounded border border-gray-600 focus:border-blue-500 focus:outline-none resize-none"
                      rows={3}
                      disabled={isAnswering}
                    />
                  </div>

                  {/* Action Buttons */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleAnswer(question)}
                      disabled={!answer.trim() || isAnswering}
                      className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:text-gray-400 text-white text-sm px-4 py-2 rounded transition-colors"
                    >
                      {isAnswering ? (
                        <>
                          <RefreshCw size={16} className="animate-spin" />
                          Sending...
                        </>
                      ) : (
                        <>
                          <Send size={16} />
                          Submit Answer
                        </>
                      )}
                    </button>

                    {question.default_answer && (
                      <button
                        onClick={() => handleUseDefault(question)}
                        disabled={isAnswering}
                        className="flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white text-sm px-3 py-2 rounded transition-colors"
                        title="Use default answer"
                      >
                        <SkipForward size={16} />
                        Use Default
                      </button>
                    )}

                    <button
                      onClick={() => handleDismiss(question)}
                      disabled={isAnswering}
                      className="flex items-center gap-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white text-sm px-3 py-2 rounded transition-colors"
                      title="Dismiss question"
                    >
                      <Trash2 size={16} />
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default QuestionsView;
