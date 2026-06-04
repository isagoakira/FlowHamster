import { beforeEach, describe, expect, it } from 'vitest'
import { FlowHamsterEdge, FlowHamsterNode } from '../types/graph'
import { LOCAL_TEMPLATES } from '../hooks/useTemplateStore'
import { generateLocalCode } from './codeGenerator'
import { WorkflowTrainingConfig } from '../schema/workflowDocument'
import { registerCustomClass } from './customCompositeRegistry'

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
  beforeEach(() => {
    localStorage.removeItem('flowhamster_custom_composites')
  })

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

  it('emits all class definitions required by nested custom packages', () => {
    const innerClass = registerCustomClass(
      'InnerBlock',
      'custom',
      'other',
      'pkg',
      'inner',
      [{ id: 'inner_relu', type: 'relu', label: 'InnerReLU', params: {} }],
      [],
      'inner_relu'
    )
    const outerClass = registerCustomClass(
      'OuterBlock',
      'custom',
      'other',
      'pkg',
      'outer',
      [
        {
          id: 'inner_instance',
          type: 'custom',
          label: 'inner_0',
          params: {},
          customClassId: innerClass.name,
          data: {
            nodeType: 'custom' as any,
            label: 'inner_0',
            params: {},
            isComposite: true,
            isCustomComposite: true,
            customClassId: innerClass.name,
            customClassRegistryId: innerClass.id,
            isExpanded: false,
            internalStructure: innerClass.internalStructure,
            internalEdges: innerClass.internalEdges,
            outputVar: innerClass.outputVar,
            inputs: [],
            outputs: [],
            childNodeIds: ['inner_relu'],
            internalEdgeIds: [],
          },
        },
        { id: 'outer_relu', type: 'relu', label: 'OuterReLU', params: {} },
      ],
      [{ id: 'inner-to-outer', from: 'inner_instance', to: 'outer_relu' }],
      'outer_relu'
    )

    const nodes: FlowHamsterNode[] = [
      {
        id: 'input',
        type: 'inputNode',
        position: { x: 0, y: 0 },
        data: { nodeType: 'input', label: 'Input', params: {} },
      },
      {
        id: 'outer_node',
        type: 'customNode',
        position: { x: 120, y: 0 },
        data: {
          nodeType: 'custom',
          label: 'outer_0',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: outerClass.name,
          customClassRegistryId: outerClass.id,
          isExpanded: false,
          internalStructure: outerClass.internalStructure,
          internalEdges: outerClass.internalEdges,
          outputVar: outerClass.outputVar,
          inputs: [],
          outputs: [],
          childNodeIds: ['inner_instance', 'outer_relu'],
          internalEdgeIds: ['inner-to-outer'],
        },
      },
      {
        id: 'output',
        type: 'outputNode',
        position: { x: 260, y: 0 },
        data: { nodeType: 'output', label: 'Output', params: {} },
      },
    ]
    const edges: FlowHamsterEdge[] = [
      { id: 'input-to-outer', source: 'input', target: 'outer_node', sourceHandle: 'result', targetHandle: 'input_0' },
      { id: 'outer-to-output', source: 'outer_node', target: 'output', sourceHandle: 'output_0', targetHandle: 'input' },
    ]

    const { code } = generateLocalCode(nodes, edges)

    expect(code).toContain('class InnerBlock(nn.Module):')
    expect(code).toContain('class OuterBlock(nn.Module):')
    expect(code).toContain('self.inner_instance = InnerBlock()')
    expect(code).toContain('self.x_outer_0 = OuterBlock()')
  })

  it('emits inline class definition for custom composite nodes not in localStorage', () => {
    // Clear localStorage to simulate the bug scenario
    localStorage.removeItem('flowhamster_custom_composites')

    const nodes: FlowHamsterNode[] = [
      {
        id: 'input',
        type: 'inputNode',
        position: { x: 0, y: 0 },
        data: { nodeType: 'input', label: 'Input', params: {} },
      },
      {
        id: 'resblock_1',
        type: 'customNode',
        position: { x: 120, y: 0 },
        data: {
          nodeType: 'custom',
          label: 'ResBlock',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'ResBlock',
          isExpanded: false,
          internalStructure: [
            { id: 'conv1', type: 'conv2d', label: 'Conv1', params: { in_channels: 64, out_channels: 64, kernel_size: 3, padding: 1 } },
            { id: 'relu1', type: 'relu', label: 'ReLU1', params: {} },
          ],
          internalEdges: [{ from: 'conv1', to: 'relu1' }],
          outputVar: 'relu1',
          inputs: [],
          outputs: [],
          childNodeIds: ['conv1', 'relu1'],
          internalEdgeIds: ['conv1-relu1'],
        },
      },
      {
        id: 'output',
        type: 'outputNode',
        position: { x: 260, y: 0 },
        data: { nodeType: 'output', label: 'Output', params: {} },
      },
    ]
    const edges: FlowHamsterEdge[] = [
      { id: 'input-to-resblock', source: 'input', target: 'resblock_1', sourceHandle: 'result', targetHandle: 'input_0' },
      { id: 'resblock-to-output', source: 'resblock_1', target: 'output', sourceHandle: 'output_0', targetHandle: 'input' },
    ]

    const { code } = generateLocalCode(nodes, edges)

    // The class definition must be present even without localStorage
    expect(code).toContain('class ResBlock(nn.Module):')
    expect(code).toContain('self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)')
    expect(code).toContain('self.relu1 = nn.ReLU()')
    expect(code).toContain('self.x_ResBlock = ResBlock()')
    expect(code).toContain('x_relu1 = self.relu1(x_conv1)')
  })

  describe('composite loss handling', () => {
    it('should handle empty components array with fallback', () => {
      // This tests BUG-003 fix: composite loss with empty components should use fallback
      const nodes: FlowHamsterNode[] = [
        {
          id: 'input',
          type: 'inputNode',
          position: { x: 0, y: 0 },
          data: { nodeType: 'input', label: 'Input', params: {} },
        },
        {
          id: 'linear',
          type: 'linearNode',
          position: { x: 100, y: 0 },
          data: { nodeType: 'linear', label: 'Linear', params: { in_features: 10, out_features: 10 } },
        },
      ]
      const edges: FlowHamsterEdge[] = [
        { id: 'e1', source: 'input', target: 'linear', sourceHandle: null, targetHandle: null },
      ]

      // Create a mock training config with empty components but fallback
      const mockConfig: WorkflowTrainingConfig = {
        taskType: 'classification',
        loss: {
          type: 'composite',
          enabled: true,
          params: {
            components: [],  // Empty components - BUG-003 scenario
            fallbackLoss: 'cross_entropy',
          },
        },
        optimizer: {
          type: 'adam',
          enabled: true,
          params: { lr: 0.001 },
        },
        scheduler: {
          type: 'step',
          enabled: false,
          params: {},
        },
        metrics: [],
        runtime: {
          device: 'cuda',
          epochs: 10,
          batchSize: 32,
          amp: false,
          gradClip: null,
          numWorkers: 4,
        },
        checkpoint: {
          enabled: false,
          saveTopK: 1,
          monitor: 'val_loss',
          mode: 'min',
          earlyStopPatience: null,
        },
      }

      // The generateLocalCode function takes nodes, edges, features?, trainingConfig?, workflowOptions?
      const { code } = generateLocalCode(nodes, edges, undefined, mockConfig)

      // Should fall back to cross_entropy when components is empty
      expect(code).toContain('nn.CrossEntropyLoss()')
      // Should NOT try to build composite loss expression with empty components
      expect(code).not.toContain('loss_fn_0')
    })

    it('should handle multiple weighted loss components', () => {
      const nodes: FlowHamsterNode[] = [
        {
          id: 'input',
          type: 'inputNode',
          position: { x: 0, y: 0 },
          data: { nodeType: 'input', label: 'Input', params: {} },
        },
        {
          id: 'linear',
          type: 'linearNode',
          position: { x: 100, y: 0 },
          data: { nodeType: 'linear', label: 'Linear', params: { in_features: 10, out_features: 10 } },
        },
      ]
      const edges: FlowHamsterEdge[] = [
        { id: 'e1', source: 'input', target: 'linear', sourceHandle: null, targetHandle: null },
      ]

      const mockConfig: WorkflowTrainingConfig = {
        taskType: 'segmentation',
        loss: {
          type: 'composite',
          enabled: true,
          params: {
            components: [
              { type: 'dice', weight: 0.3 },
              { type: 'cross_entropy', weight: 0.7 },
            ],
          },
        },
        optimizer: {
          type: 'adam',
          enabled: true,
          params: { lr: 0.001 },
        },
        scheduler: {
          type: 'step',
          enabled: false,
          params: {},
        },
        metrics: [],
        runtime: {
          device: 'cuda',
          epochs: 10,
          batchSize: 32,
          amp: false,
          gradClip: null,
          numWorkers: 4,
        },
        checkpoint: {
          enabled: false,
          saveTopK: 1,
          monitor: 'val_loss',
          mode: 'min',
          earlyStopPatience: null,
        },
      }

      const { code } = generateLocalCode(nodes, edges, undefined, mockConfig)

      // Should have weighted sum expression
      expect(code).toContain('0.3')
      expect(code).toContain('0.7')
      // Should emit both loss functions
      expect(code).toContain('loss_fn_0')
      expect(code).toContain('loss_fn_1')
    })

    it('should handle custom loss code in composite', () => {
      const nodes: FlowHamsterNode[] = [
        {
          id: 'input',
          type: 'inputNode',
          position: { x: 0, y: 0 },
          data: { nodeType: 'input', label: 'Input', params: {} },
        },
        {
          id: 'linear',
          type: 'linearNode',
          position: { x: 100, y: 0 },
          data: { nodeType: 'linear', label: 'Linear', params: { in_features: 10, out_features: 10 } },
        },
      ]
      const edges: FlowHamsterEdge[] = [
        { id: 'e1', source: 'input', target: 'linear', sourceHandle: null, targetHandle: null },
      ]

      const mockConfig: WorkflowTrainingConfig = {
        taskType: 'custom',
        loss: {
          type: 'composite',
          enabled: true,
          params: {
            components: [
              { type: 'custom', weight: 1.0, customCode: 'def custom_loss_fn(pred, target):\n    return torch.nn.functional.mse_loss(pred, target)' },
            ],
          },
        },
        optimizer: {
          type: 'adam',
          enabled: true,
          params: { lr: 0.001 },
        },
        scheduler: {
          type: 'step',
          enabled: false,
          params: {},
        },
        metrics: [],
        runtime: {
          device: 'cuda',
          epochs: 10,
          batchSize: 32,
          amp: false,
          gradClip: null,
          numWorkers: 4,
        },
        checkpoint: {
          enabled: false,
          saveTopK: 1,
          monitor: 'val_loss',
          mode: 'min',
          earlyStopPatience: null,
        },
      }

      const { code } = generateLocalCode(nodes, edges, undefined, mockConfig)

      // Should emit the custom code
      expect(code).toContain('def custom_loss_fn')
    })
  })
})
