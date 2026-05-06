/**
 * FlowHamster Port Configuration
 *
 * Unified port and connection configuration for frontend-backend communication.
 * This file mirrors backend/config/ports.json for TypeScript usage.
 */

export const PORTS = {
  backend: {
    api: 8000,
    description: 'FastAPI server port',
  },
  frontend: {
    dev: 5173,
    description: 'Vite development server',
  },
} as const;

export const CORS = {
  allowedOrigins: [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://192.168.102.231:5173',
  ],
} as const;

export const WEBSOCKET = {
  paths: {
    code: '/ws/code',
    sshLogs: '/api/ssh/ws/logs',
  },
  heartbeatIntervalSeconds: 30,
} as const;

export const API = {
  baseUrl: `http://localhost:${PORTS.backend.api}`,
  wsUrl: `ws://localhost:${PORTS.backend.api}`,
  endpoints: {
    generate: '/api/generate',
    execute: '/api/execute',
    executeForward: '/api/execute/forward',
    executeGradients: '/api/execute/gradients',
    export: '/api/export',
    exportNotebook: '/api/export-notebook',
    templates: '/api/templates',
    workflows: '/api/workflows',
    runs: '/api/runs',
    ssh: {
      connect: '/api/ssh/connect',
      disconnect: '/api/ssh/disconnect',
      execute: '/api/ssh/execute',
      submitTraining: '/api/ssh/submit-training',
      trainingStatus: '/api/ssh/training-status',
      trainingLogs: '/api/ssh/training-logs',
      stopTraining: '/api/ssh/stop-training',
      connections: '/api/ssh/connections',
    },
  },
} as const;

export type Ports = typeof PORTS;
export type CorsConfig = typeof CORS;
export type WebSocketConfig = typeof WEBSOCKET;
export type ApiConfig = typeof API;
