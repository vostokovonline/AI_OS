/**
 * Screen: Questions from System
 *
 * Минимальный интерфейс для ask_user
 * Показывает вопросы системы и позволяет ответить
 */

import { useEffect, useState } from 'react';
import { MessageCircle, CheckCircle, Clock, Send } from 'lucide-react';

interface SystemQuestion {
  question_id: string;
  subject_type: string;
  subject_id: string;
  question: string;
  options?: string[];
  status: 'pending' | 'answered';
  answer?: string;
  free_text?: string;
  created_at: string;
}

export function QuestionsScreen() {
  const [questions, setQuestions] = useState<SystemQuestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [answering, setAnswering] = useState<string | null>(null);
  const [selectedAnswer, setSelectedAnswer] = useState<string>('');
  const [freeText, setFreeText] = useState<string>('');

  useEffect(() => {
    loadQuestions();
  }, []);

  const loadQuestions = async () => {
    setLoading(true);
    try {
      // TODO: Загрузить реальные вопросы из API
      // const resp = await apiClient.get('/questions/pending');
      // setQuestions(resp.questions || []);
      
      // Временно пусто - вопросов пока нет
      setQuestions([]);
    } catch (error) {
      console.error('Failed to load questions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAnswer = async (questionId: string) => {
    if (!selectedAnswer && !freeText.trim()) {
      return; // Нельзя отправить пустой ответ
    }

    setAnswering(questionId);

    try {
      // TODO: Отправить ответ в API
      // await apiClient.post(`/questions/${questionId}/answer`, {
      //   answer: selectedAnswer,
      //   free_text: freeText
      // });

      // Удалить отвеченный вопрос из списка
      setQuestions(prev => prev.filter(q => q.question_id !== questionId));
      
      // Сбросить форму
      setSelectedAnswer('');
      setFreeText('');
      setAnswering(null);
    } catch (error) {
      console.error('Failed to submit answer:', error);
      setAnswering(null);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">Загрузка вопросов...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-2">
          <MessageCircle className="text-blue-400" size={20} />
          <h3 className="text-lg font-semibold text-gray-100">Вопросы Системы</h3>
        </div>

        <p className="text-sm text-gray-500">
          Система задаёт вопросы для принятия решений
        </p>

        <div className="mt-4 p-3 bg-blue-900/10 border border-blue-900/50 rounded text-xs text-blue-400">
          <div className="font-mono mb-1">HUMAN-IN-THE-LOOP:</div>
          Система НЕ принимает решения autonomously.<br />
          Она спрашивает оператора и ждёт явного ответа.
        </div>
      </div>

      {/* Questions List */}
      {questions.length === 0 ? (
        <div className="bg-gray-900 border border-gray-800 rounded-lg p-12 text-center">
          <CheckCircle className="text-green-400 mx-auto mb-3" size={48} />
          <div className="text-gray-400 mb-2">Нет активных вопросов</div>
          <div className="text-sm text-gray-600">
            Система не требует внимания оператора
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {questions.map((question) => (
            <div
              key={question.question_id}
              className="bg-gray-900 border border-gray-800 rounded-lg p-6"
            >
              {/* Question Header */}
              <div className="flex items-start justify-between gap-4 mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-mono text-gray-500 uppercase">
                      {question.subject_type}
                    </span>
                    <span className="text-xs text-gray-600">
                      {new Date(question.created_at).toLocaleString()}
                    </span>
                  </div>
                  <h4 className="text-base text-gray-200">{question.question}</h4>
                </div>
                {question.status === 'answered' && (
                  <CheckCircle className="text-green-400" size={20} />
                )}
                {question.status === 'pending' && (
                  <Clock className="text-yellow-400" size={20} />
                )}
              </div>

              {/* Answer Form */}
              {question.status === 'pending' && answering === question.question_id && (
                <div className="mt-4 pt-4 border-t border-gray-800 space-y-4">
                  {/* Options */}
                  {question.options && question.options.length > 0 && (
                    <div>
                      <label className="text-sm text-gray-400 mb-2 block">Выберите вариант:</label>
                      <div className="space-y-2">
                        {question.options.map((option) => (
                          <label key={option} className="flex items-center gap-3 cursor-pointer">
                            <input
                              type="radio"
                              name={`question-${question.question_id}`}
                              value={option}
                              checked={selectedAnswer === option}
                              onChange={(e) => setSelectedAnswer(e.target.value)}
                              className="w-4 h-4 text-blue-500"
                            />
                            <span className="text-sm text-gray-300">{option}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Free Text */}
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">
                      Комментарий (опционально):
                    </label>
                    <textarea
                      value={freeText}
                      onChange={(e) => setFreeText(e.target.value)}
                      placeholder="Опишите детали..."
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-sm text-gray-200 resize-none focus:border-blue-500 focus:outline-none"
                      rows={3}
                    />
                  </div>

                  {/* Submit Button */}
                  <button
                    onClick={() => handleAnswer(question.question_id)}
                    disabled={answering !== null || (!selectedAnswer && !freeText.trim())}
                    className="flex items-center justify-center gap-2 w-full bg-blue-900/20 text-blue-400 border border-blue-900/50 hover:bg-blue-900/30 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                  >
                    {answering === question.question_id ? (
                      <>
                        <Clock className="animate-spin" size={16} />
                        Отправка...
                      </>
                    ) : (
                      <>
                        <Send size={16} />
                        Отправить ответ
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Already Answered */}
              {question.status === 'answered' && (
                <div className="mt-4 pt-4 border-t border-gray-800">
                  <div className="text-sm text-gray-400 mb-1">Ваш ответ:</div>
                  <div className="text-sm text-gray-300">{question.answer}</div>
                  {question.free_text && (
                    <div className="mt-2 text-xs text-gray-500">{question.free_text}</div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Info */}
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h4 className="text-sm font-semibold text-gray-300 mb-3">Как это работает</h4>
        <div className="space-y-2 text-xs text-gray-500">
          <div>
            <span className="font-mono text-gray-400">1.</span> Система создаёт вопрос через ask_user
          </div>
          <div>
            <span className="font-mono text-gray-400">2.</span> Вопрос появляется на этом экране
          </div>
          <div>
            <span className="font-mono text-gray-400">3.</span> Вы выбираете вариант и/или пишете комментарий
          </div>
          <div>
            <span className="font-mono text-gray-400">4.</span> Система использует ответ для действий
          </div>
        </div>
      </div>
    </div>
  );
}
