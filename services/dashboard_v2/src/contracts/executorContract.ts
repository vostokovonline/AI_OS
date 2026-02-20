/**
 * v2 UI - Goal Executor Contract
 *
 * JSON Schemas for UI ↔ Goal Executor communication
 * This is the contract that enables UI to mechanically integrate with Core System
 */

// ============================================================================
// COMMAND CONTRACT (UI → Goal Executor)
// ============================================================================

/**
 * Commands that UI can send to Goal Executor
 */
export interface ExecutorCommand {
  commandId: string;
  timestamp: string;
  command: ExecutorCommandType;
  payload: any;
  timeout?: number;
}

export type ExecutorCommandType =
  | 'DECOMPOSE_GOAL'
  | 'EXECUTE_GOAL'
  | 'SIMULATE_PATH'
  | 'OVERRIDE_DECISION'
  | 'UPDATE_CONSTRAINTS'
  | 'QUERY_STATUS'
  | 'CANCEL_EXECUTION';

// JSON Schemas for validation

export const EXECUTOR_COMMAND_SCHEMAS = {
  DECOMPOSE_GOAL: {
    type: 'object',
    required: ['commandId', 'timestamp', 'command', 'payload'],
    properties: {
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      command: { type: 'string', enum: ['DECOMPOSE_GOAL'] },
      payload: {
        type: 'object',
        required: ['goalId'],
        properties: {
          goalId: { type: 'string' },
          maxDepth: { type: 'number', minimum: 1, maximum: 5 },
          preserveConstraints: { type: 'boolean' },
        },
      },
      timeout: { type: 'number', minimum: 1000, maximum: 300000 },
    },
  },

  EXECUTE_GOAL: {
    type: 'object',
    required: ['commandId', 'timestamp', 'command', 'payload'],
    properties: {
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      command: { type: 'string', enum: ['EXECUTE_GOAL'] },
      payload: {
        type: 'object',
        required: ['goalId'],
        properties: {
          goalId: { type: 'string' },
          priority: { type: 'number', minimum: 1, maximum: 10 },
          resources: {
            type: 'object',
            properties: {
              maxCost: { type: 'number' },
              maxDuration: { type: 'number' },
            },
          },
        },
      },
      timeout: { type: 'number', minimum: 5000, maximum: 600000 },
    },
  },

  SIMULATE_PATH: {
    type: 'object',
    required: ['commandId', 'timestamp', 'command', 'payload'],
    properties: {
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      command: { type: 'string', enum: ['SIMULATE_PATH'] },
      payload: {
        type: 'object',
        required: ['nodeId'],
        properties: {
          nodeId: { type: 'string' },
          alternatives: { type: 'boolean' },
          depth: { type: 'number', minimum: 1, maximum: 10 },
        },
      },
      timeout: { type: 'number', minimum: 1000, maximum: 60000 },
    },
  },

  OVERRIDE_DECISION: {
    type: 'object',
    required: ['commandId', 'timestamp', 'command', 'payload'],
    properties: {
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      command: { type: 'string', enum: ['OVERRIDE_DECISION'] },
      payload: {
        type: 'object',
        required: ['decisionId', 'reason'],
        properties: {
          decisionId: { type: 'string' },
          reason: { type: 'string' },
          forceExecution: { type: 'boolean' },
        },
      },
      timeout: { type: 'number', minimum: 1000, maximum: 30000 },
    },
  },

  UPDATE_CONSTRAINTS: {
    type: 'object',
    required: ['commandId', 'timestamp', 'command', 'payload'],
    properties: {
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      command: { type: 'string', enum: ['UPDATE_CONSTRAINTS'] },
      payload: {
        type: 'object',
        required: ['constraintType'],
        properties: {
          constraintType: { type: 'string', enum: ['ethics', 'budget', 'time'] },
          value: {},
        },
      },
      timeout: { type: 'number', minimum: 100, maximum: 5000 },
    },
  },

  QUERY_STATUS: {
    type: 'object',
    required: ['commandId', 'timestamp', 'command', 'payload'],
    properties: {
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      command: { type: 'string', enum: ['QUERY_STATUS'] },
      payload: {
        type: 'object',
        properties: {
          goalId: { type: 'string' },
        },
      },
      timeout: { type: 'number', minimum: 100, maximum: 5000 },
    },
  },

  CANCEL_EXECUTION: {
    type: 'object',
    required: ['commandId', 'timestamp', 'command', 'payload'],
    properties: {
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      command: { type: 'string', enum: ['CANCEL_EXECUTION'] },
      payload: {
        type: 'object',
        required: ['executionId'],
        properties: {
          executionId: { type: 'string' },
          reason: { type: 'string' },
        },
      },
      timeout: { type: 'number', minimum: 100, maximum: 5000 },
    },
  },
};

// ============================================================================
// RESPONSE CONTRACT (Goal Executor → UI)
// ============================================================================

/**
 * Responses from Goal Executor to UI
 */
export interface ExecutorResponse {
  responseId: string;
  commandId: string;
  timestamp: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'rejected';
  result?: any;
  error?: ExecutorError;
}

export interface ExecutorError {
  code: string;
  message: string;
  details?: any;
  retryable: boolean;
}

export const EXECUTOR_RESPONSE_SCHEMAS = {
  DECOMPOSE_RESULT: {
    type: 'object',
    required: ['responseId', 'commandId', 'timestamp', 'status'],
    properties: {
      responseId: { type: 'string', format: 'uuid' },
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      status: {
        type: 'string',
        enum: ['pending', 'processing', 'completed', 'failed', 'rejected'],
      },
      result: {
        type: 'object',
        properties: {
          subgoals: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'string' },
                intent: { type: 'string' },
                goalType: { type: 'string', enum: ['achievable', 'unachievable', 'philosophical'] },
                parentId: { type: 'string' },
                progress: { type: 'number', minimum: 0, maximum: 1 },
              },
            },
          },
          conflicts: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                id: { type: 'string' },
                severity: { type: 'number', minimum: 0, maximum: 1 },
                description: { type: 'string' },
              },
            },
          },
        },
      },
      error: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          message: { type: 'string' },
          details: {},
          retryable: { type: 'boolean' },
        },
      },
    },
  },

  EXECUTION_RESULT: {
    type: 'object',
    required: ['responseId', 'commandId', 'timestamp', 'status'],
    properties: {
      responseId: { type: 'string', format: 'uuid' },
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      status: {
        type: 'string',
        enum: ['pending', 'processing', 'completed', 'failed', 'rejected'],
      },
      result: {
        type: 'object',
        properties: {
          executionId: { type: 'string' },
          goalId: { type: 'string' },
          finalStatus: { type: 'string', enum: ['done', 'failed', 'blocked'] },
          progress: { type: 'number', minimum: 0, maximum: 1 },
          duration: { type: 'number' },
          output: {},
          executionLog: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                timestamp: { type: 'string', format: 'date-time' },
                phase: { type: 'string' },
                message: { type: 'string' },
              },
            },
          },
        },
      },
      error: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          message: { type: 'string' },
          details: {},
          retryable: { type: 'boolean' },
        },
      },
    },
  },

  SIMULATION_RESULT: {
    type: 'object',
    required: ['responseId', 'commandId', 'timestamp', 'status'],
    properties: {
      responseId: { type: 'string', format: 'uuid' },
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      status: {
        type: 'string',
        enum: ['pending', 'processing', 'completed', 'failed', 'rejected'],
      },
      result: {
        type: 'object',
        properties: {
          simulationId: { type: 'string' },
          path: {
            type: 'array',
            items: { type: 'string' },
          },
          score: { type: 'number', minimum: 0, maximum: 1 },
          duration: { type: 'number' },
          success: { type: 'boolean' },
          alternatives: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                path: { type: 'array', items: { type: 'string' } },
                score: { type: 'number' },
              },
            },
          },
        },
      },
      error: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          message: { type: 'string' },
          details: {},
          retryable: { type: 'boolean' },
        },
      },
    },
  },

  STATUS_QUERY: {
    type: 'object',
    required: ['responseId', 'commandId', 'timestamp', 'status'],
    properties: {
      responseId: { type: 'string', format: 'uuid' },
      commandId: { type: 'string', format: 'uuid' },
      timestamp: { type: 'string', format: 'date-time' },
      status: {
        type: 'string',
        enum: ['pending', 'processing', 'completed', 'failed', 'rejected'],
      },
      result: {
        type: 'object',
        properties: {
          goalId: { type: 'string' },
          status: { type: 'string' },
          progress: { type: 'number' },
          activeSubgoals: { type: 'array', items: { type: 'string' } },
          recentActivity: {
            type: 'array',
            items: { type: 'object' },
          },
        },
      },
      error: {
        type: 'object',
        properties: {
          code: { type: 'string' },
          message: { type: 'string' },
          details: {},
          retryable: { type: 'boolean' },
        },
      },
    },
  },
};

// ============================================================================
// COMMAND HELPERS
// ============================================================================

/**
 * Helper functions to create properly formatted commands
 */
export const ExecutorCommands = {
  decomposeGoal: (
    goalId: string,
    options?: { maxDepth?: number; preserveConstraints?: boolean }
  ): ExecutorCommand => ({
    commandId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    command: 'DECOMPOSE_GOAL',
    payload: {
      goalId,
      maxDepth: options?.maxDepth || 3,
      preserveConstraints: options?.preserveConstraints ?? true,
    },
    timeout: 60000,
  }),

  executeGoal: (
    goalId: string,
    options?: { priority?: number; maxCost?: number; maxDuration?: number }
  ): ExecutorCommand => ({
    commandId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    command: 'EXECUTE_GOAL',
    payload: {
      goalId,
      priority: options?.priority || 5,
      resources: {
        maxCost: options?.maxCost,
        maxDuration: options?.maxDuration,
      },
    },
    timeout: 600000, // 10 minutes
  }),

  simulatePath: (
    nodeId: string,
    options?: { alternatives?: boolean; depth?: number }
  ): ExecutorCommand => ({
    commandId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    command: 'SIMULATE_PATH',
    payload: {
      nodeId,
      alternatives: options?.alternatives ?? false,
      depth: options?.depth || 3,
    },
    timeout: 30000,
  }),

  overrideDecision: (
    decisionId: string,
    reason: string,
    forceExecution?: boolean
  ): ExecutorCommand => ({
    commandId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    command: 'OVERRIDE_DECISION',
    payload: {
      decisionId,
      reason,
      forceExecution: forceExecution ?? false,
    },
    timeout: 10000,
  }),

  updateConstraints: (
    constraintType: 'ethics' | 'budget' | 'time',
    value: any
  ): ExecutorCommand => ({
    commandId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    command: 'UPDATE_CONSTRAINTS',
    payload: {
      constraintType,
      value,
    },
    timeout: 5000,
  }),

  queryStatus: (goalId?: string): ExecutorCommand => ({
    commandId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    command: 'QUERY_STATUS',
    payload: {
      goalId,
    },
    timeout: 5000,
  }),

  cancelExecution: (executionId: string, reason: string): ExecutorCommand => ({
    commandId: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    command: 'CANCEL_EXECUTION',
    payload: {
      executionId,
      reason,
    },
    timeout: 5000,
  }),
};

// ============================================================================
// VALIDATION HELPERS
// ============================================================================

/**
 * Validate command against schema
 */
export function validateCommand(
  command: ExecutorCommand,
  _schema: any
): { valid: boolean; errors: string[] } {
  // Simple validation - in production use ajv or similar
  const errors: string[] = [];

  if (!command.commandId) errors.push('Missing commandId');
  if (!command.timestamp) errors.push('Missing timestamp');
  if (!command.payload) errors.push('Missing payload');

  // Check required fields based on command type
  if (command.command === 'DECOMPOSE_GOAL' && !command.payload.goalId) {
    errors.push('DECOMPOSE_GOAL requires goalId');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate response against schema
 */
export function validateResponse(
  response: ExecutorResponse,
  _schema: any
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!response.responseId) errors.push('Missing responseId');
  if (!response.commandId) errors.push('Missing commandId');
  if (!response.timestamp) errors.push('Missing timestamp');
  if (!response.status) errors.push('Missing status');

  if (response.status === 'failed' || response.status === 'rejected') {
    if (!response.error) errors.push('Failed/rejected responses must include error');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

// ============================================================================
// ASYNC API CLIENT (for actual communication)
// ============================================================================

/**
 * Send command to Goal Executor and get response
 */
export async function sendCommand(
  command: ExecutorCommand,
  endpoint: string = '/api/executor/command'
): Promise<ExecutorResponse> {
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(command),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return data as ExecutorResponse;
  } catch (error) {
    // Return error response
    return {
      responseId: crypto.randomUUID(),
      commandId: command.commandId,
      timestamp: new Date().toISOString(),
      status: 'failed',
      error: {
        code: 'NETWORK_ERROR',
        message: error instanceof Error ? error.message : 'Unknown error',
        retryable: true,
      },
    };
  }
}
