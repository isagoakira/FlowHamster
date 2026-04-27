import { describe, expect, it } from 'vitest'
import { FlowHamsterEdge, FlowHamsterNode } from '../types/graph'
import { LOCAL_TEMPLATES } from '../hooks/useTemplateStore'
import { generateLocalCode } from './codeGenerator'

function getTemplateGraph(templateId: string): { nodes: FlowHamsterNode[]; edges: FlowHamsterEdge[] } {
  const template = LOCAL_TEMPLATES.find((item) => item.id === templateId)
  if (!template) throw new Error(`Missing template: ${templateId}`)

  return {
    nodes: template.graph.nodes.map((node) => ({
      id: node.id,
      type: `${node.data.nodeType}Node`,
      position: node.data.position ?? { x: 0, y: 0 },
      data: {
        nodeType: node.data.nodeType as any,
        label: node.data.label,
        params: node.data.params as Record<string, number | string | boolean>,
      },
    })) as FlowHamsterNode[],
    edges: template.graph.edges.map((edge, index) => ({
      id: `e_${index}`,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle ?? 'result',
      targetHandle: edge.targetHandle ?? 'a',
    })) as FlowHamsterEdge[],
  }
}

describe('codeGenerator', () => {
  it('emits ViT transformer blocks as readable classes and instances', () => {
    const { nodes, edges } = getTemplateGraph('vit')
    const { code } = generateLocalCode(nodes, edges)

    expect(code).toContain('class FlowHamsterTransformerEncoderBlock(nn.Module):')
    expect(code).toContain('class FlowHamsterTransformerEncoder(nn.Module):')
    expect(code).toContain('self.x_transformerencoder_1 = FlowHamsterTransformerEncoder(')
    expect(code).not.toContain('sa_q_proj')
    expect(code).not.toContain('getattr(self, f"')
  })

  it('registers parameter nodes in __init__ instead of creating them in forward', () => {
    const { nodes, edges } = getTemplateGraph('vit')
    const { code } = generateLocalCode(nodes, edges)

    expect(code).toContain('self.x_parameter_1 = nn.Parameter(torch.zeros(1,1,768))')
    expect(code).toContain('x_parameter_1 = self.x_parameter_1')
    expect(code).not.toMatch(/^\s*x_parameter_1 = nn\.Parameter\(/m)
    expect(code).toContain('x_add_1 = x_concat_1 + x_parameter_2')
  })
})
