/**
 * FlowHamster API Client
 */
import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { getExecutableGraph } from './graphStructure'
import { API_BASE_URL } from './runtimeConfig'
import { WorkflowTrainingConfig } from '../schema/workflowDocument'

export interface GenerateOptions {
  include_trainer?: boolean
  optimizer?: string
  loss_function?: string
  lr?: number
  epochs?: number
  evaluation_nodes?: boolean   // 是否包含评测节点代码
}

export async function generateCode(
  nodes: FlowHamsterNode[],
  edges: FlowHamsterEdge[],
  options?: GenerateOptions,
  trainingConfig?: WorkflowTrainingConfig
): Promise<{ success: boolean; code: string; warnings: string[] }> {
  const executableGraph = getExecutableGraph(nodes, edges)
  const graph = {
    nodes: executableGraph.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data })),
    edges: executableGraph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      sourceHandle: e.sourceHandle ?? null,
      targetHandle: e.targetHandle ?? null,
      data: e.data,
    })),
  }
  const res = await fetch(`${API_BASE_URL}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ graph, options: options || {}, training_config: trainingConfig || null }),
  })
  return res.json()
}

export async function executeCode(
  code: string,
  device: string = 'cpu'
): Promise<{ success: boolean; output: string; error?: string }> {
  const res = await fetch(`${API_BASE_URL}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, target_device: device }),
  })
  return res.json()
}

export async function exportCode(
  code: string,
  format: 'py' | 'ipynb',
  filename: string
): Promise<{ success: boolean; content: string; filename: string; mime_type: string }> {
  const res = await fetch(`${API_BASE_URL}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, format, filename }),
  })
  return res.json()
}

/**
 * 导出为 Jupyter Notebook（推荐方式）。
 * 前端传 graph JSON，后端负责代码生成 + cell 划分。
 */
export async function exportNotebook(
  graph: { nodes: any[]; edges: any[] },
  filename: string = 'flowhamster_model',
  trainingConfig?: WorkflowTrainingConfig,
  dataGraph?: { nodes: any[]; edges: any[] },
  bindings?: Array<{ sourceGraph: string; sourceKey: string; target: string; targetKey: string }>
): Promise<{ success: boolean; content: string; filename: string; mime_type: string }> {
  const executableGraph = getExecutableGraph(graph.nodes as FlowHamsterNode[], graph.edges as FlowHamsterEdge[])
  const res = await fetch(`${API_BASE_URL}/export-notebook`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      graph: executableGraph,
      filename,
      training_config: trainingConfig || null,
      data_graph: dataGraph || null,
      bindings: bindings || null,
    }),
  })
  const data = await res.json()
  if (data.success) {
    downloadFile(data.content, data.filename, data.mime_type)
  }
  return data
}

export function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = atob(content)
  const bytes = new Uint8Array(blob.length)
  for (let i = 0; i < blob.length; i++) bytes[i] = blob.charCodeAt(i)
  const url = URL.createObjectURL(new Blob([bytes], { type: mimeType }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
