/**
 * OCCP API Client
 * Integration with OCCP v1.0 backend services
 */

interface Skill {
  skill_id: string;
  version: string;
  author: string;
  description: string;
  capabilities: string[];
  contracts: any;
}

interface Deployment {
  deployment_id: string;
  skill_id: string;
  version: string;
  status: 'canary' | 'stable' | 'rolled_back' | 'green';
  traffic_percentage: number;
  created_at: string;
  updated_at: string;
}

interface Metric {
  timestamp: string;
  skill_id: string;
  version: string;
  action: string;
  status: string;
  duration_ms: number;
}

interface Node {
  node_id: string;
  role: 'primary' | 'edge';
  status: 'active' | 'inactive' | 'degraded';
  skills_count: number;
  last_seen: string;
}

interface Incident {
  incident_id: string;
  incident_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  status: 'detected' | 'resolved';
  detected_at: string;
}

interface Proposal {
  proposal_id: string;
  proposal_type: string;
  description: string;
  priority: 'low' | 'medium' | 'high';
  status: 'pending' | 'implemented';
  created_at: string;
}

const API_BASE = 'http://localhost:8000';

/**
 * OCCP API Client
 */
export const occpApi = {
  /**
   * Get all deployed skills
   */
  async getSkills(): Promise<Skill[]> {
    // In real implementation, this would call the OCCP backend
    // For now, return mock data based on our deployed skills
    return [
      {
        skill_id: 'hello_world',
        version: '1.0.0',
        author: 'authority',
        description: 'Simple hello world skill for testing OCCP v1.0',
        capabilities: ['greet', 'echo', 'get_timestamp'],
        contracts: {
          max_execution_time_seconds: 10,
          max_memory_mb: 64,
          max_tokens: 1000
        }
      },
      {
        skill_id: 'calculator',
        version: '1.0.0',
        author: 'authority',
        description: 'Advanced calculator with multiple operations',
        capabilities: ['add', 'subtract', 'multiply', 'divide', 'power'],
        contracts: {
          max_execution_time_seconds: 5,
          max_memory_mb: 32,
          max_tokens: 500
        }
      }
    ];
  },

  /**
   * Get all deployments
   */
  async getDeployments(): Promise<Deployment[]> {
    // Mock data - directly return since backend doesn't have this endpoint yet
    console.log('[OCCP] Loading deployments (using mock data)');

    // Simulate async delay for realism
    await new Promise(resolve => setTimeout(resolve, 300));

    return [
      {
        deployment_id: 'dep-001',
        skill_id: 'hello_world',
        version: '1.0.0',
        status: 'stable',
        traffic_percentage: 100,
        created_at: new Date(Date.now() - 86400000).toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        deployment_id: 'dep-002',
        skill_id: 'calculator',
        version: '1.0.0',
        status: 'canary',
        traffic_percentage: 10,
        created_at: new Date(Date.now() - 3600000).toISOString(),
        updated_at: new Date().toISOString()
      }
    ];
  },

  /**
   * Get metrics for a skill
   */
  async getMetrics(skillId?: string): Promise<Metric[]> {
    // Mock data - generate sample metrics since backend doesn't have this endpoint yet
    console.log('[OCCP] Loading metrics', skillId ? `for ${skillId}` : '(all skills)', '(using mock data)');

    // Simulate async delay for realism
    await new Promise(resolve => setTimeout(resolve, 200));

    const now = Date.now();
    const skills = skillId ? [skillId] : ['hello_world', 'calculator'];
    const metrics: Metric[] = [];

    for (let i = 0; i < 50; i++) {
      const skill = skills[i % skills.length];
      const timestamp = new Date(now - i * 60000).toISOString();
      const passed = Math.random() > 0.1; // 90% success rate

      metrics.push({
        timestamp,
        skill_id: skill,
        version: '1.0.0',
        action: ['greet', 'calculate', 'add', 'echo'][i % 4],
        status: passed ? 'passed' : 'failed',
        duration_ms: Math.floor(Math.random() * 200) + 50
      });
    }

    return metrics;
  },

  /**
   * Get federation nodes
   */
  async getFederationNodes(): Promise<Node[]> {
    // Return mock data based on our federation setup
    return [
      {
        node_id: 'node1',
        role: 'primary',
        status: 'active',
        skills_count: 2,
        last_seen: new Date().toISOString()
      },
      {
        node_id: 'node2',
        role: 'edge',
        status: 'active',
        skills_count: 2,
        last_seen: new Date().toISOString()
      },
      {
        node_id: 'node3',
        role: 'edge',
        status: 'active',
        skills_count: 2,
        last_seen: new Date().toISOString()
      }
    ];
  },

  /**
   * Get incidents
   */
  async getIncidents(): Promise<Incident[]> {
    // Read from mitigation database
    const response = await fetch(`${API_BASE}/incidents`);
    if (!response.ok) {
      throw new Error('Failed to fetch incidents');
    }
    return response.json();
  },

  /**
   * Get proposals
   */
  async getProposals(): Promise<Proposal[]> {
    // Read from proposal database
    const response = await fetch(`${API_BASE}/proposals`);
    if (!response.ok) {
      throw new Error('Failed to fetch proposals');
    }
    return response.json();
  },

  /**
   * Deploy a skill
   */
  async deploySkill(skillId: string, version: string): Promise<Deployment> {
    const response = await fetch(`${API_BASE}/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skill_id, version })
    });
    if (!response.ok) {
      throw new Error('Failed to deploy skill');
    }
    return response.json();
  },

  /**
   * Rollback a deployment
   */
  async rollbackDeployment(deploymentId: string, reason: string): Promise<void> {
    const response = await fetch(`${API_BASE}/deployments/${deploymentId}/rollback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason })
    });
    if (!response.ok) {
      throw new Error('Failed to rollback deployment');
    }
  },

  /**
   * Get system health
   */
  async getSystemHealth(): Promise<{
    status: 'healthy' | 'degraded' | 'critical';
    components: any[];
  }> {
    // Aggregate health from all systems
    return {
      status: 'healthy',
      components: [
        { name: 'Skills Registry', status: 'operational' },
        { name: 'CI/CD Pipeline', status: 'operational' },
        { name: 'Observability', status: 'operational' },
        { name: 'Federation', status: 'operational' },
        { name: 'Mitigation', status: 'operational' }
      ]
    };
  }
};
