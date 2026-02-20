/**
 * Decomposition Screen - Human-in-the-Loop Goal Decomposition
 *
 * Features:
 * - Interactive AI Avatar (color = emotion, pulse = activity)
 * - Real-time question/answer flow
 * - Session history
 */

import { useState, useEffect } from 'react';
import { Brain, MessageCircle, Send, Clock, CheckCircle, Target } from 'lucide-react';
import { apiClient } from '../../api/client';

interface Goal {
  id: string;
  title: string;
  goal_type: string;
  status: string;
  created_at: string;
}

interface DecompositionSession {
  id: string;
  goal_id: string;
  status: 'awaiting_user' | 'in_progress' | 'completed' | 'aborted';
  initiated_by: string;
  created_at: string;
  updated_at: string;
}

interface DecompositionQuestion {
  id: string;
  question_text: string;
  question_index: number;
  question_type: string;
  asked_by: string;
  created_at: string;
  answer?: {
    id: string;
    answer_text: string;
    answered_by: string;
    created_at: string;
  };
}

// AI Avatar emotions mapped to colors
type Emotion = 'neutral' | 'thinking' | 'curious' | 'waiting' | 'processing' | 'satisfied';

const EMOTION_COLORS: Record<Emotion, string> = {
  neutral: 'bg-blue-500',
  thinking: 'bg-purple-500',
  curious: 'bg-yellow-500',
  waiting: 'bg-gray-400',
  processing: 'bg-green-500',
  satisfied: 'bg-emerald-400'
};

const EMOTION_GLOW: Record<Emotion, string> = {
  neutral: 'shadow-blue-500/50',
  thinking: 'shadow-purple-500/50',
  curious: 'shadow-yellow-500/50',
  waiting: 'shadow-gray-400/50',
  processing: 'shadow-green-500/50',
  satisfied: 'shadow-emerald-400/50'
};

interface AIAvatarProps {
  emotion: Emotion;
  isActive: boolean;
  size?: 'sm' | 'md' | 'lg';
}

function AIAvatar({ emotion, isActive, size = 'lg' }: AIAvatarProps) {
  const sizeClasses = {
    sm: 'w-12 h-12',
    md: 'w-20 h-20',
    lg: 'w-32 h-32'
  };

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Avatar with pulse animation */}
      <div className="relative">
        {/* Pulse effect when active */}
        {isActive && (
          <div className={`absolute inset-0 rounded-full ${EMOTION_COLORS[emotion]} animate-ping opacity-20`} />
        )}

        {/* Main avatar */}
        <div
          className={`
            ${sizeClasses[size]}
            rounded-full
            ${EMOTION_COLORS[emotion]}
            ${EMOTION_GLOW[emotion]}
            shadow-lg
            flex items-center justify-center
            transition-all duration-500
            ${isActive ? 'scale-110' : 'scale-100'}
          `}
        >
          <Brain className="text-white" size={size === 'sm' ? 24 : size === 'md' ? 36 : 48} />
        </div>

        {/* Activity indicator ring */}
        {isActive && (
          <div className={`
            absolute inset-0 rounded-full border-2
            ${EMOTION_COLORS[emotion]}
            animate-pulse
          `} style={{ animationDuration: '2s' }} />
        )}
      </div>

      {/* Emotion label */}
      <div className="text-xs text-gray-400 font-mono uppercase">
        {emotion}
      </div>
    </div>
  );
}

export function DecompositionScreen() {
  const [goals, setGoals] = useState<Goal[]>([]);
  const [selectedGoal, setSelectedGoal] = useState<Goal | null>(null);
  const [currentSession, setCurrentSession] = useState<DecompositionSession | null>(null);
  const [questions, setQuestions] = useState<DecompositionQuestion[]>([]);
  const [avatarEmotion, setAvatarEmotion] = useState<Emotion>('neutral');
  const [isAvatarActive, setIsAvatarActive] = useState(false);
  const [userAnswer, setUserAnswer] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [currentQuestionId, setCurrentQuestionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingGoals, setLoadingGoals] = useState(true);
  const [decomposing, setDecomposing] = useState(false);

  // Load goals on mount
  useEffect(() => {
    loadGoals();
  }, []);

  // Load active session when goal is selected
  useEffect(() => {
    if (selectedGoal) {
      loadSessionsForGoal(selectedGoal.id);
    }
  }, [selectedGoal]);

  // Update avatar emotion based on session state
  useEffect(() => {
    if (!currentSession) {
      setAvatarEmotion('neutral');
      setIsAvatarActive(false);
      return;
    }

    switch (currentSession.status) {
      case 'awaiting_user':
        setAvatarEmotion('waiting');
        setIsAvatarActive(true);
        break;
      case 'in_progress':
        setAvatarEmotion('processing');
        setIsAvatarActive(true);
        break;
      case 'completed':
        setAvatarEmotion('satisfied');
        setIsAvatarActive(false);
        break;
      default:
        setAvatarEmotion('neutral');
        setIsAvatarActive(false);
    }
  }, [currentSession]);

  const loadGoals = async () => {
    try {
      setLoadingGoals(true);
      const resp = await apiClient.getPendingGoals();
      const pendingGoals = resp.goals || [];

      // Filter only pending goals that can be decomposed
      const decomposableGoals = pendingGoals.filter((g: Goal) =>
        g.status === 'pending' && g.goal_type !== 'atomic'
      );

      setGoals(decomposableGoals);

      // Auto-select first goal if available
      if (decomposableGoals.length > 0) {
        setSelectedGoal(decomposableGoals[0]);
      }
    } catch (error) {
      console.error('Failed to load goals:', error);
    } finally {
      setLoadingGoals(false);
    }
  };

  const loadSessionsForGoal = async (goalId: string) => {
    try {
      const resp = await apiClient.getActiveDecompositionSession(goalId);

      if (resp.has_active_session && resp.session_id) {
        // Load the full session data
        const sessionData = await apiClient.getDecompositionSession(resp.session_id);
        setCurrentSession(sessionData.session);
        setQuestions(sessionData.questions);

        // Find first unanswered question
        const unanswered = sessionData.questions.find((q: DecompositionQuestion) => !q.answer);
        if (unanswered) {
          setCurrentQuestionId(unanswered.id);
        }

        setAvatarEmotion('waiting');
        setIsAvatarActive(true);
      } else {
        // No active session, clear state
        setCurrentSession(null);
        setQuestions([]);
        setCurrentQuestionId(null);
        setAvatarEmotion('neutral');
        setIsAvatarActive(false);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const startDecomposition = async () => {
    if (!selectedGoal || loading) return;

    setLoading(true);
    setAvatarEmotion('thinking');
    setIsAvatarActive(true);

    try {
      const goalId = selectedGoal.id;

      // Check if there's already an active session
      const checkResp = await apiClient.getActiveDecompositionSession(goalId);

      if (checkResp.has_active_session && checkResp.session_id) {
        // Load existing session
        const sessionData = await apiClient.getDecompositionSession(checkResp.session_id);
        setCurrentSession(sessionData.session);
        setQuestions(sessionData.questions);

        // Find first unanswered question
        const unanswered = sessionData.questions.find((q: DecompositionQuestion) => !q.answer);
        if (unanswered) {
          setCurrentQuestionId(unanswered.id);
        }

        setAvatarEmotion('waiting');
      } else {
        // Start new decomposition - ask first question based on goal type
        let firstQuestion = '';
        let questionType = 'criteria';

        if (selectedGoal.goal_type === 'philosophical') {
          firstQuestion = 'Как ты поймёшь, что эта цель начала реализовываться?\n\nОпиши наблюдаемый признак, событие или результат.';
          questionType = 'criteria';
        } else if (selectedGoal.goal_type === 'achievable') {
          firstQuestion = 'Что уже есть на старте? Какие ресурсы/знания доступны?';
          questionType = 'exploration';
        } else {
          firstQuestion = 'С чего начнём? Опиши первый шаг.';
          questionType = 'first_step';
        }

        const askResp = await apiClient.askDecomposition({
          goal_id: goalId,
          question_text: firstQuestion,
          question_type: questionType,
          initiated_by: 'human'
        });

        // Load the created session
        const sessionData = await apiClient.getDecompositionSession(askResp.session_id);
        setCurrentSession(sessionData.session);
        setQuestions(sessionData.questions);

        if (sessionData.questions.length > 0) {
          setCurrentQuestionId(sessionData.questions[0].id);
        }

        setAvatarEmotion('waiting');
      }

      setIsAvatarActive(true);
    } catch (error) {
      console.error('Failed to start decomposition:', error);
      setAvatarEmotion('neutral');
      setIsAvatarActive(false);
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async () => {
    if (!userAnswer.trim() || !currentQuestionId || submitting) return;

    setSubmitting(true);
    setAvatarEmotion('processing');

    try {
      // Submit answer
      await apiClient.submitDecompositionAnswer({
        question_id: currentQuestionId,
        answer_text: userAnswer,
        answered_by: 'human'
      });

      setUserAnswer('');

      // Reload session to get updated questions
      if (currentSession) {
        const sessionData = await apiClient.getDecompositionSession(currentSession.id);
        setCurrentSession(sessionData.session);
        setQuestions(sessionData.questions);

        // Find next unanswered question
        const nextUnanswered = sessionData.questions.find((q: DecompositionQuestion) => !q.answer);
        if (nextUnanswered) {
          setCurrentQuestionId(nextUnanswered.id);
          setAvatarEmotion('waiting');
        } else {
          // All questions answered
          setAvatarEmotion('satisfied');
          setIsAvatarActive(false);
        }
      }
    } catch (error) {
      console.error('Failed to submit answer:', error);
      setAvatarEmotion('waiting');
    } finally {
      setSubmitting(false);
    }
  };

  const decomposeFromAnswers = async () => {
    if (!currentSession || decomposing) return;

    setDecomposing(true);
    setAvatarEmotion('thinking');
    setIsAvatarActive(true);

    try {
      const result = await apiClient.decomposeFromAnswers(currentSession.id);

      // Перезагружаем цели чтобы увидеть новые подцели
      await loadGoals();

      setAvatarEmotion('satisfied');
      setIsAvatarActive(false);

      // Показываем уведомление
      alert(`Декомпозиция запущена!\n\n${result.message}`);
    } catch (error) {
      console.error('Failed to decompose:', error);
      setAvatarEmotion('waiting');
      alert('Ошибка при запуске декомпозиции: ' + (error as any).message);
    } finally {
      setDecomposing(false);
    }
  };

  return (
    <div className="h-full flex bg-gray-950">
      {/* Left Panel - Avatar + Active Session */}
      <div className="w-1/3 border-r border-gray-800 p-8 flex flex-col">
        {/* Goal Selector */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-3">
            <Target className="text-purple-400" size={18} />
            <h3 className="text-sm font-semibold text-gray-300">Выберите цель</h3>
          </div>

          {loadingGoals ? (
            <div className="text-sm text-gray-500 italic">Загрузка целей...</div>
          ) : goals.length === 0 ? (
            <div className="text-sm text-gray-500 italic">Нет доступных целей</div>
          ) : (
            <select
              value={selectedGoal?.id || ''}
              onChange={(e) => {
                const goal = goals.find(g => g.id === e.target.value);
                if (goal) setSelectedGoal(goal);
              }}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 focus:border-purple-500 focus:outline-none"
            >
              {goals.map((goal) => (
                <option key={goal.id} value={goal.id}>
                  {goal.title}
                </option>
              ))}
            </select>
          )}

          {selectedGoal && (
            <div className="mt-2 text-xs text-gray-500">
              Тип: {selectedGoal.goal_type}
            </div>
          )}
        </div>

        {/* AI Avatar */}
        <div className="flex-1 flex flex-col items-center justify-center">
          <AIAvatar
            emotion={avatarEmotion}
            isActive={isAvatarActive}
            size="lg"
          />

          {currentSession && (
            <div className="mt-8 text-center">
              <div className="text-sm text-gray-400 mb-2">Статус сессии</div>
              <div className="text-lg text-gray-200 font-mono">
                {currentSession.status}
              </div>
            </div>
          )}
        </div>

        {/* Action */}
        <div className="border-t border-gray-800 pt-6">
          <button
            onClick={() => startDecomposition()}
            disabled={loading || !selectedGoal}
            className="w-full bg-blue-900/20 text-blue-400 border border-blue-900/50 hover:bg-blue-900/30 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-4 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Clock className="animate-spin" size={16} />
                Загрузка...
              </>
            ) : (
              <>
                <Brain size={16} />
                Начать декомпозицию
              </>
            )}
          </button>
        </div>
      </div>

      {/* Right Panel - Questions & Answers */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-800">
          <div className="flex items-center gap-2 mb-2">
            <MessageCircle className="text-blue-400" size={20} />
            <h2 className="text-lg font-semibold text-gray-100">Декомпозиция цели</h2>
          </div>
          <p className="text-sm text-gray-500">
            Диалог с системой для разбиения цели на подцели
          </p>
        </div>

        {/* Questions List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {questions.length === 0 ? (
            <div className="text-center py-12">
              <Brain className="text-gray-600 mx-auto mb-3" size={48} />
              <div className="text-gray-400 mb-2">Нет активных вопросов</div>
              <div className="text-sm text-gray-600">
                Используйте /decompose в Telegram или нажмите "Начать декомпозицию"
              </div>
            </div>
          ) : (
            questions.map((q) => (
              <div
                key={q.id}
                className="bg-gray-900 border border-gray-800 rounded-lg p-6"
              >
                {/* Question */}
                <div className="flex items-start gap-3 mb-4">
                  <div className="bg-purple-900/20 rounded-full p-2">
                    <Brain className="text-purple-400" size={16} />
                  </div>
                  <div className="flex-1">
                    <div className="text-sm text-gray-500 mb-1">
                      Вопрос {q.question_index}
                    </div>
                    <div className="text-gray-200">{q.question_text}</div>
                  </div>
                </div>

                {/* Answer */}
                {q.answer ? (
                  <div className="flex items-start gap-3 pl-11">
                    <div className="bg-green-900/20 rounded-full p-2">
                      <CheckCircle className="text-green-400" size={16} />
                    </div>
                    <div className="flex-1">
                      <div className="text-sm text-gray-500 mb-1">
                        Ваш ответ
                      </div>
                      <div className="text-gray-300">{q.answer.answer_text}</div>
                    </div>
                  </div>
                ) : q.id === currentQuestionId ? (
                  <div className="pl-11">
                    <textarea
                      value={userAnswer}
                      onChange={(e) => setUserAnswer(e.target.value)}
                      placeholder="Ваш ответ..."
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-sm text-gray-200 resize-none focus:border-blue-500 focus:outline-none"
                      rows={3}
                    />
                    <button
                      onClick={submitAnswer}
                      disabled={submitting || !userAnswer.trim()}
                      className="mt-2 flex items-center gap-2 bg-blue-900/20 text-blue-400 border border-blue-900/50 hover:bg-blue-900/30 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                    >
                      {submitting ? (
                        <>
                          <Clock className="animate-spin" size={16} />
                          Отправка...
                        </>
                      ) : (
                        <>
                          <Send size={16} />
                          Отправить
                        </>
                      )}
                    </button>
                  </div>
                ) : (
                  <div className="pl-11 text-sm text-gray-500 italic">
                    Ожидает ответа...
                  </div>
                )}
              </div>
            ))
          )}

          {/* Completed message */}
          {currentSession?.status === 'completed' && questions.length > 0 && questions.every(q => q.answer) && (
            <div className="mt-6 bg-emerald-900/20 border border-emerald-700 rounded-lg p-6">
              <div className="flex items-start gap-3">
                <CheckCircle className="text-emerald-400" size={24} />
                <div className="flex-1">
                  <div className="text-emerald-300 font-semibold mb-2">Декомпозиция завершена!</div>
                  <div className="text-gray-300 text-sm mb-4">
                    Все вопросы отвечены. Сессия сохранена со статусом "completed".
                  </div>

                  {/* Decompose Action Button */}
                  <button
                    onClick={decomposeFromAnswers}
                    disabled={decomposing}
                    className="w-full bg-purple-900/30 text-purple-300 border border-purple-700/50 hover:bg-purple-900/50 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-4 py-3 text-sm font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    {decomposing ? (
                      <>
                        <Clock className="animate-spin" size={16} />
                        Запуск декомпозиции цели...
                      </>
                    ) : (
                      <>
                        <Brain size={16} />
                        Запустить декомпозицию на основе ответов
                      </>
                    )}
                  </button>

                  <div className="text-xs text-gray-500 mt-2 italic">
                    Это создаст подцели на основе ваших ответов
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
