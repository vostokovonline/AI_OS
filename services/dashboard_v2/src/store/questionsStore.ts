/**
 * v2 UI - Questions Store
 *
 * Zustand store for managing user questions
 */

import { create } from 'zustand';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export interface Question {
  question_id: string;
  artifact_id: string;
  goal_id: string;
  question: string;
  context: string;
  options?: string[];
  priority: 'critical' | 'high' | 'normal' | 'low';
  asked_at: string;
  timeout_at: string;
  timeout_action: 'continue_with_default' | 'wait_longer' | 'fail_goal';
  default_answer?: string;
  status: 'pending' | 'answered' | 'timeout';
}

export interface QuestionStats {
  pending_count: number;
  priority_breakdown: {
    critical: number;
    high: number;
    normal: number;
    low: number;
  };
}

interface QuestionsStore {
  questions: Question[];
  stats: QuestionStats | null;
  loading: boolean;
  error: string | null;
  selectedGoalId: string | null;

  // Actions
  setQuestions: (questions: Question[]) => void;
  setStats: (stats: QuestionStats) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setSelectedGoalId: (goalId: string | null) => void;
  fetchQuestions: () => Promise<void>;
  fetchStats: () => Promise<void>;
  answerQuestion: (questionId: string, answer: string, useDefault?: boolean) => Promise<void>;
  dismissQuestion: (questionId: string) => Promise<void>;
}

export const useQuestionsStore = create<QuestionsStore>((set, get) => ({
  questions: [],
  stats: null,
  loading: false,
  error: null,
  selectedGoalId: null,

  setQuestions: (questions) => set({ questions }),
  setStats: (stats) => set({ stats }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
  setSelectedGoalId: (goalId) => set({ selectedGoalId: goalId }),

  fetchQuestions: async () => {
    set({ loading: true, error: null });
    try {
      const { selectedGoalId } = get();
      const params = selectedGoalId ? `?goal_id=${selectedGoalId}` : '';
      const response = await fetch(`${API_BASE_URL}/questions/pending${params}`);
      const data = await response.json();

      if (data.status === 'ok') {
        set({ questions: data.questions || [] });
      } else {
        set({ error: data.error || 'Failed to fetch questions' });
      }
    } catch (error) {
      set({ error: (error as Error).message });
    } finally {
      set({ loading: false });
    }
  },

  fetchStats: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/questions/stats`);
      const data = await response.json();

      if (data.status === 'ok') {
        set({ stats: data.stats });
      }
    } catch (error) {
      console.error('Failed to fetch question stats:', error);
    }
  },

  answerQuestion: async (questionId, answer, useDefault = false) => {
    set({ loading: true, error: null });
    try {
      const response = await fetch(`${API_BASE_URL}/questions/${questionId}/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer, use_default: useDefault }),
      });
      const data = await response.json();

      if (data.status === 'ok') {
        // Remove the answered question from the list
        set((state) => ({
          questions: state.questions.filter((q) => q.artifact_id !== questionId),
        }));
      } else {
        set({ error: data.error || 'Failed to submit answer' });
      }
    } catch (error) {
      set({ error: (error as Error).message });
    } finally {
      set({ loading: false });
    }
  },

  dismissQuestion: async (questionId) => {
    set({ loading: true, error: null });
    try {
      const response = await fetch(`${API_BASE_URL}/questions/${questionId}/dismiss`, {
        method: 'POST',
      });
      const data = await response.json();

      if (data.status === 'ok') {
        // Remove the dismissed question from the list
        set((state) => ({
          questions: state.questions.filter((q) => q.artifact_id !== questionId),
        }));
      } else {
        set({ error: data.error || 'Failed to dismiss question' });
      }
    } catch (error) {
      set({ error: (error as Error).message });
    } finally {
      set({ loading: false });
    }
  },
}));
