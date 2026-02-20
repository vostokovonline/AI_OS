# AI-OS v2 Dashboard

**Operational Thinking Interface for AI-OS**

## Overview

The v2 Dashboard is a complete redesign of the AI-OS interface, following the principle that "UI is a projection of the system's internal state, not a collection of screens."

## Key Principles

1. **Graph-Based Visualization** - Goals, agents, skills, and memories as interconnected nodes
2. **Context-Aware Inspectors** - Click any node to see detailed context
3. **Layered Views** - Heatmaps, conflict overlays, memory traces
4. **Operational Focus** - Mode switching (explore/exploit/reflect)
5. **Causality Timeline** - Trace decisions and branches

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  Control Layer (Modes / Constraints / Overrides)               │
│  [ Explore | Exploit | Reflect ]   ⛔ Ethics ⏱ Time 💾 Memory   │
├───────────────┬─────────────────────────────────┬─────────────┤
│               │                                 │             │
│   Left Rail   │        MAIN CANVAS               │ Inspector  │
│   Controls    │   (Goal / Agent Graph)           │   Panel    │
│               │                                 │             │
├───────────────┴─────────────────────────────────┴─────────────┤
│   Timeline / Causality Strip                                   │
└───────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **ReactFlow** - Graph visualization
- **Zustand** - State management
- **Axios** - API client
- **TailwindCSS** - Styling
- **Lucide React** - Icons

## Project Structure

```
src/
├── api/
│   └── client.ts           # API client for backend communication
├── components/
│   ├── canvas/
│   │   ├── GraphCanvas.tsx
│   │   └── nodes/          # Custom ReactFlow nodes
│   ├── controls/
│   │   └── ControlPanel.tsx
│   ├── inspector/
│   │   └── InspectorPanel.tsx
│   └── timeline/
│       └── TimelineStrip.tsx
├── store/
│   ├── uiStore.ts          # UI state machine
│   └── graphStore.ts       # Graph state management
├── types/
│   └── index.ts            # Type definitions
├── App.tsx                 # Root component
├── main.tsx                # Entry point
└── index.css               # Global styles
```

## Development

### Prerequisites

- Node.js 18+
- npm or yarn
- AI-OS backend running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:3000`

### WSL2 Access (Windows)

If you're running on WSL2 and want to access from Windows browser:

#### Option 1: Direct IP Access (Recommended)

```bash
# Get WSL2 IP address
npm run wsl-ip

# Or run the script directly
bash ./get-wsl-ip.sh
```

Then open in Windows browser:
- `http://<WSL2_IP>:3000` for Dashboard
- `http://<WSL2_IP>:8000` for Backend API

#### Option 2: Port Forwarding (Persistent)

1. **Run PowerShell as Administrator** in Windows
2. Navigate to the dashboard directory: `cd \\wsl$\Ubuntu\home\onor\ai_os_final\services\dashboard_v2`
3. Run the setup script: `.\setup-windows-access.ps1`

This will set up port forwarding so you can use `http://localhost:3000` from Windows.

#### Option 3: Manual Port Forwarding

If the script doesn't work, run these commands in Administrator PowerShell:

```powershell
# Get WSL2 IP
wsl hostname -I

# Replace <WSL_IP> with the actual IP address
netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=<WSL_IP>
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=<WSL_IP>

# View rules
netsh interface portproxy show all

# To remove later:
netsh interface portproxy delete v4tov4 listenport=3000 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0
```

#### Windows Firewall

If you still can't access, allow ports through Windows Firewall:

```powershell
netsh advfirewall firewall add rule name="WSL2 Dashboard" dir=in action=allow protocol=TCP localport=3000
netsh advfirewall firewall add rule name="WSL2 Backend" dir=in action=allow protocol=TCP localport=8000
```

### Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Features

### 1. Graph Canvas
- Interactive visualization of goals, agents, and skills
- Custom node types with rich visual encoding
- Zoom and pan support
- Mini-map for navigation

### 2. Inspector Panel
- Context-aware detailed view
- Goal metrics (feasibility, conflict, uncertainty)
- Agent performance (confidence, success rate, cost)
- Skill capabilities and usage statistics
- Memory traces and TTL

### 3. Control Panel
- Mode switching (Explore / Exploit / Reflect)
- View overlays (Heatmap / Conflicts / Memory Traces)
- Constraint management (ethics, budget, time)
- Manual override controls

### 4. Timeline Strip
- Decision history and causality
- Branch visualization
- Time-travel to past states
- Knowledge state snapshots

## UI State Machine

The dashboard follows a strict state machine model:

```typescript
UIState {
  mode: 'explore' | 'exploit' | 'reflect'
  focus: { nodeId, nodeType }
  overlay: 'none' | 'heatmap' | 'conflicts' | 'memory_traces'
  timelineCursor: timestamp | null
  constraints: { ethics[], budget, timeHorizon }
  override: { enabled, decisionId }
  graph: { zoom, center, collapsedLevels }
}
```

## Event Model

### UI → System Events

```typescript
- SELECT_NODE(nodeId)
- CHANGE_MODE(mode)
- APPLY_OVERLAY(overlay)
- TIMELINE_JUMP(timestamp)
- REQUEST_DECOMPOSE(goalId)
- REQUEST_SIMULATION(nodeId)
- OVERRIDE_DECISION(decisionId)
- CONSTRAINT_UPDATE(type, value)
```

### System → UI Events

```typescript
- GRAPH_UPDATED(diff)
- GOAL_STATUS_CHANGED(goalId, status, progress)
- CONFLICT_DETECTED(nodeA, nodeB, severity)
- SIMULATION_RESULT(path, score)
- EXECUTION_PROGRESS(nodeId, progress)
- ERROR(reason, context)
```

## API Integration

The dashboard communicates with the AI-OS backend via:

- REST API for queries and mutations
- Server-Sent Events (SSE) for real-time updates
- WebSocket for live execution progress

## Design Philosophy

Unlike traditional dashboards, v2 is designed as:

1. **Operational** - Focus on what the system is thinking, not just displaying data
2. **Explainable** - Every decision can be traced through the timeline
3. **Debuggable** - Inspect any node to understand the full context
4. **Scalable** - Graph-based design scales with system complexity

## Future Enhancements

- [ ] Simulation mode with "what-if" scenarios
- [ ] Multi-view comparison
- [ ] Advanced filtering and search
- [ ] Custom node layouts (hierarchical, force-directed)
- [ ] Export/import graph states
- [ ] Real-time collaboration
- [ ] Mobile responsive design

## License

MIT
