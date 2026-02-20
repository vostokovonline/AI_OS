/**
 * Mock data for testing Dashboard v2
 */

import { Node, GraphEdge } from '../types';

// Mock goals for testing
export const mockGoals: Node[] = [
  {
    id: 'root-1',
    type: 'goal',
    intent: 'Создать AI-продукт',
    goalType: 'achievable',
    status: 'active',
    progress: 0.3,
    feasibility: 0.8,
    conflictScore: 0.1,
    uncertainty: 0.2,
    parentId: undefined,
    childIds: ['goal-1', 'goal-2', 'goal-3'],
    createdAt: new Date().toISOString(),
    constraints: {},
  },
  {
    id: 'goal-1',
    type: 'goal',
    intent: 'Изучить конкурентов',
    goalType: 'achievable',
    status: 'blocked',
    progress: 0.0,
    feasibility: 0.5,
    conflictScore: 0.3,
    uncertainty: 0.4,
    parentId: 'root-1',
    childIds: [],
    createdAt: new Date().toISOString(),
    constraints: {},
  },
  {
    id: 'goal-2',
    type: 'goal',
    intent: 'Создать MVP',
    goalType: 'achievable',
    status: 'active',
    progress: 0.33,
    feasibility: 0.7,
    conflictScore: 0.2,
    uncertainty: 0.3,
    parentId: 'root-1',
    childIds: ['goal-2-1', 'goal-2-2'],
    createdAt: new Date().toISOString(),
    constraints: {},
  },
  {
    id: 'goal-2-1',
    type: 'goal',
    intent: 'Написать код',
    goalType: 'achievable',
    status: 'done',
    progress: 1.0,
    feasibility: 0.9,
    conflictScore: 0.0,
    uncertainty: 0.1,
    parentId: 'goal-2',
    childIds: [],
    completedAt: new Date().toISOString(),
    createdAt: new Date(Date.now() - 86400000).toISOString(),
    constraints: {},
  },
  {
    id: 'goal-2-2',
    type: 'goal',
    intent: 'Создать интерфейс',
    goalType: 'achievable',
    status: 'active',
    progress: 0.33,
    feasibility: 0.8,
    conflictScore: 0.1,
    uncertainty: 0.2,
    parentId: 'goal-2',
    childIds: [],
    createdAt: new Date().toISOString(),
    constraints: {},
  },
  {
    id: 'goal-3',
    type: 'goal',
    intent: 'Найти клиентов',
    goalType: 'achievable',
    status: 'pending',
    progress: 0.0,
    feasibility: 0.6,
    conflictScore: 0.2,
    uncertainty: 0.5,
    parentId: 'root-1',
    childIds: [],
    createdAt: new Date().toISOString(),
    constraints: {},
  },
  {
    id: 'root-2',
    type: 'goal',
    intent: 'Развитие AI_OS',
    goalType: 'philosophical',
    status: 'active',
    progress: 0.87,
    feasibility: 0.7,
    conflictScore: 0.1,
    uncertainty: 0.3,
    parentId: undefined,
    childIds: ['goal-4'],
    createdAt: new Date(Date.now() - 172800000).toISOString(),
    constraints: {},
  },
  {
    id: 'goal-4',
    type: 'goal',
    intent: 'Улучшить навыки',
    goalType: 'achievable',
    status: 'active',
    progress: 0.5,
    feasibility: 0.8,
    conflictScore: 0.0,
    uncertainty: 0.2,
    parentId: 'root-2',
    childIds: [],
    createdAt: new Date().toISOString(),
    constraints: {},
  },
];

// Mock edges
export const mockEdges: GraphEdge[] = [
  {
    id: 'edge-1',
    source: 'root-1',
    target: 'goal-1',
    type: 'dependency',
    label: 'subgoal',
    strength: 1.0,
  },
  {
    id: 'edge-2',
    source: 'root-1',
    target: 'goal-2',
    type: 'dependency',
    label: 'subgoal',
    strength: 1.0,
  },
  {
    id: 'edge-3',
    source: 'root-1',
    target: 'goal-3',
    type: 'dependency',
    label: 'subgoal',
    strength: 1.0,
  },
  {
    id: 'edge-4',
    source: 'goal-2',
    target: 'goal-2-1',
    type: 'dependency',
    label: 'subgoal',
    strength: 1.0,
  },
  {
    id: 'edge-5',
    source: 'goal-2',
    target: 'goal-2-2',
    type: 'dependency',
    label: 'subgoal',
    strength: 1.0,
  },
  {
    id: 'edge-6',
    source: 'root-2',
    target: 'goal-4',
    type: 'causal',
    label: 'supports',
    strength: 0.8,
  },
];

// Helper function to create goals
function createGoal(
  id: string,
  intent: string,
  goalType: 'achievable' | 'unachievable' | 'philosophical',
  status: 'active' | 'blocked' | 'done' | 'pending',
  progress: number,
  parentId?: string,
  childIds: string[] = []
): any {
  return {
    id,
    type: 'goal',
    intent,
    goalType,
    status,
    progress,
    feasibility: 0.5 + Math.random() * 0.4,
    conflictScore: Math.random() * 0.3,
    uncertainty: Math.random() * 0.4,
    parentId,
    childIds,
    createdAt: new Date().toISOString(),
    constraints: {},
    ...(status === 'done' && { completedAt: new Date().toISOString() }),
  };
}

// Extended test data with realistic goal names for testing large graphs
export const simpleMockData = (() => {
  const nodes: any[] = [];
  const edges: any[] = [];

  // Root 1: Big project with meaningful subgoals
  nodes.push(createGoal('root-1', 'Создать AI-продукт', 'achievable', 'active', 0.25, undefined,
    ['g1-research', 'g1-mvp', 'g1-marketing', 'g1-sales', 'g1-support']));

  // Root 1 - Level 1 (Meaningful directions)
  nodes.push(createGoal('g1-research', 'Исследование рынка', 'achievable', 'done', 1.0, 'root-1',
    ['g1-analyze', 'g1-competitors']));
  nodes.push(createGoal('g1-mvp', 'Разработка MVP', 'achievable', 'active', 0.4, 'root-1',
    ['g1-backend', 'g1-frontend', 'g1-testing']));
  nodes.push(createGoal('g1-marketing', 'Маркетинг и реклама', 'achievable', 'pending', 0.0, 'root-1',
    ['g1-sm', 'g1-content']));
  nodes.push(createGoal('g1-sales', 'Продажи', 'achievable', 'pending', 0.0, 'root-1',
    ['g1-funnel']));
  nodes.push(createGoal('g1-support', 'Поддержка пользователей', 'achievable', 'pending', 0.0, 'root-1', []));

  // Root 1 - Level 2 (Meaningful tasks)
  nodes.push(createGoal('g1-analyze', 'Анализ целевой аудитории', 'achievable', 'done', 1.0, 'g1-research', []));
  nodes.push(createGoal('g1-competitors', 'Анализ конкурентов', 'achievable', 'active', 0.6, 'g1-research', []));

  nodes.push(createGoal('g1-backend', 'Backend разработка', 'achievable', 'active', 0.5, 'g1-mvp',
    ['g1-api', 'g1-db']));
  nodes.push(createGoal('g1-frontend', 'Frontend разработка', 'achievable', 'active', 0.3, 'g1-mvp',
    ['g1-ui', 'g1-integration']));
  nodes.push(createGoal('g1-testing', 'QA тестирование', 'achievable', 'pending', 0.0, 'g1-mvp', []));

  nodes.push(createGoal('g1-sm', 'SMM маркетинг', 'achievable', 'pending', 0.0, 'g1-marketing', []));
  nodes.push(createGoal('g1-content', 'Создание контента', 'achievable', 'pending', 0.0, 'g1-marketing', []));

  nodes.push(createGoal('g1-funnel', 'Воронка продаж', 'achievable', 'pending', 0.0, 'g1-sales', []));

  // Root 1 - Level 3 (Subtasks)
  nodes.push(createGoal('g1-api', 'REST API разработка', 'achievable', 'active', 0.6, 'g1-backend', []));
  nodes.push(createGoal('g1-db', 'База данных', 'achievable', 'done', 1.0, 'g1-backend', []));

  nodes.push(createGoal('g1-ui', 'UI компоненты', 'achievable', 'active', 0.4, 'g1-frontend', []));
  nodes.push(createGoal('g1-integration', 'Интеграция с API', 'achievable', 'pending', 0.0, 'g1-frontend', []));

  // Root 2: AI_OS development with meaningful structure
  nodes.push(createGoal('root-2', 'Развитие AI_OS', 'philosophical', 'active', 0.6, undefined,
    ['g2-skills', 'g2-optimization', 'g2-testing']));

  nodes.push(createGoal('g2-skills', 'Улучшение системы навыков', 'achievable', 'done', 1.0, 'root-2',
    ['g2-codegen', 'g2-registry']));
  nodes.push(createGoal('g2-optimization', 'Оптимизация производительности', 'achievable', 'active', 0.3, 'root-2',
    ['g2-cache', 'g2-db-opt']));
  nodes.push(createGoal('g2-testing', 'Автотестирование', 'achievable', 'pending', 0.0, 'root-2',
    ['g2-unit', 'g2-integration']));

  // Root 2 - Level 2
  nodes.push(createGoal('g2-codegen', 'Генерация кода', 'achievable', 'done', 1.0, 'g2-skills', []));
  nodes.push(createGoal('g2-registry', 'Реестр навыков', 'achievable', 'done', 1.0, 'g2-skills', []));

  nodes.push(createGoal('g2-cache', 'Кеширование', 'achievable', 'active', 0.5, 'g2-optimization', []));
  nodes.push(createGoal('g2-db-opt', 'Оптимизация БД', 'achievable', 'pending', 0.0, 'g2-optimization', []));

  nodes.push(createGoal('g2-unit', 'Unit тесты', 'achievable', 'pending', 0.0, 'g2-testing', []));
  nodes.push(createGoal('g2-integration', 'Integration тесты', 'achievable', 'pending', 0.0, 'g2-testing', []));

  // Root 3: Learning goals
  nodes.push(createGoal('root-3', 'Обучение', 'achievable', 'active', 0.6, undefined,
    ['g3-ml', 'g3-dev']));

  nodes.push(createGoal('g3-ml', 'Machine Learning', 'achievable', 'done', 1.0, 'root-3',
    ['g3-nn', 'g3-dl']));
  nodes.push(createGoal('g3-dev', 'Разработка', 'achievable', 'active', 0.4, 'root-3', []));

  nodes.push(createGoal('g3-nn', 'Нейронные сети', 'achievable', 'done', 1.0, 'g3-ml', []));
  nodes.push(createGoal('g3-dl', 'Deep Learning', 'achievable', 'active', 0.3, 'g3-ml', []));

  // Create edges from parent-child relationships
  nodes.forEach(node => {
    if (node.childIds && node.childIds.length > 0) {
      node.childIds.forEach((childId: string) => {
        edges.push({
          id: `edge-${node.id}-${childId}`,
          source: node.id,
          target: childId,
          type: 'dependency' as const,
          label: 'subgoal',
          strength: 1.0,
        });
      });
    }
  });

  // Add some cross-links
  edges.push({
    id: 'edge-cross-1',
    source: 'g2-1',
    target: 'g1-2',
    type: 'causal' as const,
    label: 'supports',
    strength: 0.7,
  });

  edges.push({
    id: 'edge-cross-2',
    source: 'g3-1',
    target: 'g2-1-1',
    type: 'reinforcement' as const,
    label: 'enables',
    strength: 0.8,
  });

  return {
    nodes: nodes as Node[],
    edges: edges as GraphEdge[],
  };
})();
