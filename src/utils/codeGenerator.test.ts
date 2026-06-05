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

  describe('loss target bindings (multi-loss)', () => {
    it('unpacks 3 values from resolve_bound_inputs when loss targets exist', () => {
      const nodes: FlowHamsterNode[] = [
        {
          id: 'input',
          type: 'inputNode',
          position: { x: 0, y: 0 },
          data: { nodeType: 'input', label: 'Input', params: { name: 'x' } },
        },
        {
          id: 'linear',
          type: 'linearNode',
          position: { x: 100, y: 0 },
          data: { nodeType: 'linear', label: 'Linear', params: { in_features: 10, out_features: 10 } },
        },
        {
          id: 'loss_ce',
          type: 'crossentropylossNode',
          position: { x: 200, y: 0 },
          data: { nodeType: 'crossentropyloss', label: 'CrossEntropy', params: { num_classes: 10 } },
        },
      ]
      const edges: FlowHamsterEdge[] = [
        { id: 'e1', source: 'input', target: 'linear', sourceHandle: null, targetHandle: null },
        { id: 'e2', source: 'linear', target: 'loss_ce', sourceHandle: null, targetHandle: null },
      ]

      const dataNodes = [
        {
          id: 'ds_out',
          type: 'dataPipelineNode',
          position: { x: 0, y: 0 },
          data: {
            nodeType: 'dataset_output',
            label: 'Dataset Output',
            params: { fields: 'image,label' },
            fieldSpecs: JSON.stringify([{ name: 'image', dtype: 'image' }, { name: 'label', dtype: 'label' }]),
          },
        },
      ] as any

      const bindings = [
        { id: 'b1', sourceGraph: 'data' as const, sourceKey: 'image', target: 'model_input' as const, targetKey: 'x' },
        { id: 'b2', sourceGraph: 'data' as const, sourceKey: 'label', target: 'loss_target' as const, targetKey: 'loss_ce' },
      ]

      const { code } = generateLocalCode(nodes, edges, undefined, undefined, {
        dataGraphNodes: dataNodes,
        dataGraphEdges: [],
        bindings,
      })

      expect(code).toContain('model_feed, target, loss_targets = resolve_bound_inputs(batch, runtime_device)')
      expect(code).toContain('BOUND_LOSS_TARGETS = {')
      expect(code).toContain('"loss_ce": "label"')
    })

    it('includes loss_targets dict in config-driven training loop', () => {
      const nodes: FlowHamsterNode[] = [
        {
          id: 'input',
          type: 'inputNode',
          position: { x: 0, y: 0 },
          data: { nodeType: 'input', label: 'Input', params: { name: 'x' } },
        },
        {
          id: 'linear',
          type: 'linearNode',
          position: { x: 100, y: 0 },
          data: { nodeType: 'linear', label: 'Linear', params: { in_features: 10, out_features: 10 } },
        },
        {
          id: 'output',
          type: 'outputNode',
          position: { x: 200, y: 0 },
          data: { nodeType: 'output', label: 'Output', params: {} },
        },
      ]
      const edges: FlowHamsterEdge[] = [
        { id: 'e1', source: 'input', target: 'linear', sourceHandle: null, targetHandle: null },
        { id: 'e2', source: 'linear', target: 'output', sourceHandle: null, targetHandle: null },
      ]

      const dataNodes = [
        {
          id: 'ds_out',
          type: 'dataPipelineNode',
          position: { x: 0, y: 0 },
          data: {
            nodeType: 'dataset_output',
            label: 'Dataset Output',
            params: { fields: 'image,label' },
            fieldSpecs: JSON.stringify([{ name: 'image', dtype: 'image' }, { name: 'label', dtype: 'label' }]),
          },
        },
      ] as any

      const bindings = [
        { id: 'b1', sourceGraph: 'data' as const, sourceKey: 'image', target: 'model_input' as const, targetKey: 'x' },
        { id: 'b2', sourceGraph: 'data' as const, sourceKey: 'label', target: 'loss_target' as const, targetKey: 'loss_ce' },
      ]

      const trainingConfig: WorkflowTrainingConfig = {
        taskType: 'classification',
        loss: { type: 'cross_entropy', enabled: true, params: {} },
        optimizer: { type: 'adam', enabled: true, params: { lr: 0.001 } },
        scheduler: { type: 'step', enabled: false, params: {} },
        metrics: [],
        runtime: { device: 'cpu', epochs: 1, batchSize: 32, amp: false, gradClip: null, numWorkers: 0 },
        checkpoint: { enabled: false, saveTopK: 1, monitor: 'val_loss', mode: 'min', earlyStopPatience: null },
      }

      const { code } = generateLocalCode(nodes, edges, undefined, trainingConfig, {
        dataGraphNodes: dataNodes,
        dataGraphEdges: [],
        bindings,
      })

      expect(code).toContain('model_feed, target, loss_targets = resolve_bound_inputs(batch, runtime_device)')
      expect(code).toContain('BOUND_LOSS_TARGETS = {')
      expect(code).toContain('"loss_ce": "label"')
    })
  })
})
