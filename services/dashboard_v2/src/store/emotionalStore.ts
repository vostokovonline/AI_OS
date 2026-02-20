/**
 * v2 UI - Emotional Layer Store
 *
 * Manages emotional state for users
 */

import { create } from 'zustand';
import { EmotionalState, EmotionalInfluence } from '../types';
import { apiClient } from '../api/client';

interface EmotionalStore {
  // Current emotional state by user
  states: Map<string, EmotionalState>;

  // History by user
  history: Map<string, EmotionalState[]>;

  // Current influence by user
  influences: Map<string, EmotionalInfluence>;

  // Loading state
  loading: boolean;
  error: string | null;

  // Actions
  fetchCurrentState: (userId: string) => Promise<void>;
  fetchHistory: (userId: string, limit?: number) => Promise<void>;
  setState: (userId: string, state: EmotionalState) => void;
  setInfluence: (userId: string, influence: EmotionalInfluence) => void;
  clear: () => void;
}

const TEST_USER_ID = '00000000-0000-0000-0000-000000000001';

export const useEmotionalStore = create<EmotionalStore>((set) => ({
  states: new Map(),
  history: new Map(),
  influences: new Map(),
  loading: false,
  error: null,

  fetchCurrentState: async (userId: string) => {
    set({ loading: true, error: null });
    try {
      const data = await apiClient.getCurrentEmotionalState(userId);
      const emotionalState: EmotionalState = {
        arousal: data.arousal ?? 0.5,
        valence: data.valence ?? 0.0,
        focus: data.focus ?? 0.5,
        confidence: data.confidence ?? 0.5,
        timestamp: data.timestamp || data.created_at,
        source: data.source,
      };

      set((storeState) => ({
        states: new Map(storeState.states).set(userId, emotionalState),
        loading: false,
      }));
    } catch (error: any) {
      console.error('[EmotionalStore] Failed to fetch current state:', error);
      set({
        error: error.message || 'Failed to fetch emotional state',
        loading: false,
      });
    }
  },

  fetchHistory: async (userId: string, limit: number = 100) => {
    set({ loading: true, error: null });
    try {
      const data = await apiClient.getEmotionalHistory(userId, limit);

      // Check if data is an array
      if (!Array.isArray(data)) {
        console.warn('[EmotionalStore] API returned non-array data:', data);
        set((state) => ({
          history: new Map(state.history).set(userId, []),
          loading: false,
        }));
        return;
      }

      const history: EmotionalState[] = data.map((item: any) => ({
        arousal: item.arousal,
        valence: item.valence,
        focus: item.focus,
        confidence: item.confidence,
        timestamp: item.timestamp || item.created_at,
        source: item.source,
      }));

      set((state) => ({
        history: new Map(state.history).set(userId, history),
        loading: false,
      }));
    } catch (error: any) {
      console.error('[EmotionalStore] Failed to fetch history:', error);
      set({
        error: error.message || 'Failed to fetch emotional history',
        loading: false,
      });
    }
  },

  setState: (userId: string, state: EmotionalState) => {
    set((s) => ({
      states: new Map(s.states).set(userId, state),
    }));
  },

  setInfluence: (userId: string, influence: EmotionalInfluence) => {
    set((s) => ({
      influences: new Map(s.influences).set(userId, influence),
    }));
  },

  clear: () => {
    set({
      states: new Map(),
      history: new Map(),
      influences: new Map(),
      error: null,
    });
  },
}));

// Helper hooks
export const useCurrentEmotionalState = (userId: string = TEST_USER_ID) => {
  return useEmotionalStore((state) => state.states.get(userId));
};

export const useEmotionalHistory = (userId: string = TEST_USER_ID) => {
  return useEmotionalStore((state) => state.history.get(userId) || []);
};

export const useEmotionalInfluence = (userId: string = TEST_USER_ID) => {
  return useEmotionalStore((state) => state.influences.get(userId));
};
