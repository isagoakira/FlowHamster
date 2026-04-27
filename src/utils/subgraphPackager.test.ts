/**
 * Unit tests for subgraphPackager.ts
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  packageNodes,
  unpackageGroup,
  expandPackage,
  collapsePackage,
  getNextModuleName,
  getNextModuleNameForNodes,
  computeTopologicalLayout,
  resetModuleCounter,
} from './subgraphPackager'
import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'

// Mock localStorage for jsdom environment
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value },
    removeItem: (key: string) => { delete store[key] },
    clear: () => { store = {} },
    get length() { return Object.keys(store).length },
    key: (i: number) => Object.keys(store)[i] || null,
  }
})()

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
  configurable: true,
})

// Helper to create a basic node
function createNode(id: string, type: string, label: string, position = { x: 0, y: 0 }): FlowHamsterNode {
  return {
    id,
    type: `${type}Node`,
    position,
    data: {
      nodeType: type as any,
      label,
      params: {},
    },
  }
}

// Helper to create an edge
function createEdge(
  id: string,
  source: string,
  target: string,
  sourceHandle: string | null = null,
  targetHandle: string | null = null
): FlowHamsterEdge {
  return {
    id,
    source,
    target,
    sourceHandle,
    targetHandle,
  }
}

describe('subgraphPackager', () => {
  beforeEach(() => {
    resetModuleCounter()
    localStorageMock.clear()
  })

  describe('packageNodes()', () => {
    it('should package 2-3 nodes into a custom composite node', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
      ]

      const result = packageNodes(nodes, edges, ['node1', 'node2'], { x: 150, y: 100 })

      expect(result).toBeDefined()
      expect(result.data.isCustomComposite).toBe(true)
      expect(result.data.isExpanded).toBe(false)
      expect(result.data.internalStructure).toHaveLength(2)
      expect(result.data.childNodeIds).toEqual(['node1', 'node2'])
    })

    it('should package nodes with interconnected edges', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'input', 'Input1', { x: 0, y: 0 }),
        createNode('node2', 'relu', 'ReLU1', { x: 100, y: 0 }),
        createNode('node3', 'linear', 'Linear1', { x: 200, y: 0 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
        createEdge('e2', 'node2', 'node3'),
      ]

      const result = packageNodes(nodes, edges, ['node1', 'node2', 'node3'], { x: 100, y: 0 })

      expect(result.data.internalEdges).toHaveLength(2)
      expect(result.data.boundaryEdges).toBeDefined()
    })

    it('should handle boundary nodes (input/output)', () => {
      const externalNode = createNode('external', 'input', 'ExternalInput', { x: 0, y: 50 })
      const internalNode = createNode('node1', 'relu', 'ReLU1', { x: 100, y: 50 })

      const nodes: FlowHamsterNode[] = [externalNode, internalNode]
      const edges: FlowHamsterEdge[] = [
        { id: 'e1', source: 'external', target: 'node1', sourceHandle: null, targetHandle: null },
      ]

      const result = packageNodes(nodes, edges, ['node1'], { x: 100, y: 50 })

      // node1 should have 1 input port since it's connected to external node
      expect(result.data.inputs).toBeDefined()
      expect(result.data.outputs).toBeDefined()
    })
  })

  describe('unpackageGroup()', () => {
    it('should restore nodes and edges after unpackage', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
      ]

      // First package
      const pkgNode = packageNodes(nodes, edges, ['node1', 'node2'], { x: 150, y: 100 })

      // Simulate the packaged state (only pkgNode exists)
      const packagedNodes: FlowHamsterNode[] = [pkgNode]
      const packagedEdges: FlowHamsterEdge[] = []

      // Unpackage
      const result = unpackageGroup(pkgNode.id, packagedNodes, packagedEdges)

      // Should restore the original nodes
      expect(result.nodes).toHaveLength(2)
      const restoredNode1 = result.nodes.find(n => n.id === 'node1')
      const restoredNode2 = result.nodes.find(n => n.id === 'node2')
      expect(restoredNode1).toBeDefined()
      expect(restoredNode2).toBeDefined()
    })

    it('should preserve original edge IDs after unpackage', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('original-edge-id', 'node1', 'node2'),
      ]

      const pkgNode = packageNodes(nodes, edges, ['node1', 'node2'], { x: 150, y: 100 })

      const packagedNodes: FlowHamsterNode[] = [pkgNode]
      const packagedEdges: FlowHamsterEdge[] = []

      const result = unpackageGroup(pkgNode.id, packagedNodes, packagedEdges)

      // Check if the edge ID is preserved
      const restoredEdge = result.edges.find(e => e.source === 'node1' && e.target === 'node2')
      expect(restoredEdge).toBeDefined()
      expect(restoredEdge?.id).toBe('original-edge-id')
    })

    it('should restore current boundary edges without reconnecting stale package history', () => {
      const externalA = createNode('external-a', 'input', 'ExternalA', { x: 0, y: 0 })
      const externalB = createNode('external-b', 'input', 'ExternalB', { x: 0, y: 120 })
      const node1 = createNode('node1', 'relu', 'ReLU1', { x: 160, y: 60 })
      const node2 = createNode('node2', 'linear', 'Linear1', { x: 320, y: 60 })
      const output = createNode('output', 'output', 'Output', { x: 480, y: 60 })
      const nodes: FlowHamsterNode[] = [externalA, externalB, node1, node2, output]
      const edges: FlowHamsterEdge[] = [
        createEdge('in-a', 'external-a', 'node1', 'out', 'x'),
        createEdge('in-b', 'external-b', 'node1', 'out', 'x'),
        createEdge('mid', 'node1', 'node2', 'y', 'x'),
        createEdge('out', 'node2', 'output', 'y', 'x'),
      ]

      const pkgNode = packageNodes(nodes, edges, ['node1', 'node2'], { x: 240, y: 60 })

      expect(pkgNode.data.inputs).toHaveLength(1)
      expect(pkgNode.data.outputs).toHaveLength(1)
      expect(pkgNode.data.boundaryEdges).toHaveLength(3)

      const packagedEdges: FlowHamsterEdge[] = [
        createEdge('in-a', 'external-a', pkgNode.id, 'out', 'input_0'),
        createEdge('in-b', 'external-b', pkgNode.id, 'out', 'input_0'),
        createEdge('out', pkgNode.id, 'output', 'output_0', 'x'),
      ]
      const result = unpackageGroup(pkgNode.id, [externalA, externalB, output, pkgNode as FlowHamsterNode], packagedEdges)

      expect(result.nodes.some(n => n.id === pkgNode.id)).toBe(false)
      expect(result.nodes.some(n => n.id === 'node1')).toBe(true)
      expect(result.nodes.some(n => n.id === 'node2')).toBe(true)

      expect(result.edges.find(e => e.id === 'in-a')).toMatchObject({
        source: 'external-a',
        target: 'node1',
        sourceHandle: 'out',
        targetHandle: 'x',
      })
      expect(result.edges.find(e => e.id === 'in-b')).toMatchObject({
        source: 'external-b',
        target: 'node1',
        sourceHandle: 'out',
        targetHandle: 'x',
      })
      expect(result.edges.find(e => e.id === 'mid')).toMatchObject({
        source: 'node1',
        target: 'node2',
        sourceHandle: 'y',
        targetHandle: 'x',
      })
      expect(result.edges.find(e => e.id === 'out')).toMatchObject({
        source: 'node2',
        target: 'output',
        sourceHandle: 'y',
        targetHandle: 'x',
      })
    })

    it('should unpackage copied instances with unique restored node and edge IDs', () => {
      const node1 = createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 })
      const node2 = createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 })
      const originalEdge = createEdge('e1', 'node1', 'node2')
      const pkgNode = packageNodes([node1, node2], [originalEdge], ['node1', 'node2'], { x: 150, y: 100 })
      const copiedPkg = {
        ...pkgNode,
        id: 'copied-package',
        position: { x: 300, y: 100 },
        data: {
          ...pkgNode.data,
          label: 'copied-package',
        },
      } as FlowHamsterNode

      const result = unpackageGroup('copied-package', [node1, node2, copiedPkg], [originalEdge])

      expect(result.nodes.find(n => n.id === 'node1')).toBeDefined()
      expect(result.nodes.find(n => n.id === 'node2')).toBeDefined()
      expect(result.nodes.find(n => n.id === 'node1_1')).toBeDefined()
      expect(result.nodes.find(n => n.id === 'node2_1')).toBeDefined()
      expect(result.edges.find(e => e.id === 'e1')).toMatchObject({ source: 'node1', target: 'node2' })
      expect(result.edges.find(e => e.id === 'e1_1')).toMatchObject({ source: 'node1_1', target: 'node2_1' })
    })
  })

  describe('expandPackage()', () => {
    it('should expand package and return internal nodes', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
      ]

      const pkgNode = packageNodes(nodes, edges, ['node1', 'node2'], { x: 150, y: 100 })

      // Expand
      const result = expandPackage(pkgNode, [pkgNode], [])

      // Should have expanded nodes plus the package node
      expect(result.nodes.length).toBeGreaterThan(1)

      // The group node should have isExpanded: true
      const expandedGroup = result.nodes.find(n => n.id === pkgNode.id)
      expect(expandedGroup).toBeDefined()
      expect((expandedGroup as any)?.data?.isExpanded).toBe(true)

      // Internal nodes should be restored
      const internalNode1 = result.nodes.find(n => n.id === 'node1')
      const internalNode2 = result.nodes.find(n => n.id === 'node2')
      expect(internalNode1).toBeDefined()
      expect(internalNode2).toBeDefined()
    })

    it('should not duplicate internal nodes when expanding already expanded package', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = []

      const pkgNode = packageNodes(nodes, edges, ['node1'], { x: 150, y: 100 })

      // First expand
      const result1 = expandPackage(pkgNode, [pkgNode], [])
      expect(result1.nodes.length).toBe(2) // pkg + 1 internal

      // Second expand (re-expand without collapsing)
      // Use the result from first expand as input
      const result2 = expandPackage(pkgNode, result1.nodes, result1.edges)

      // Should still have only 2 nodes (not 3)
      expect(result2.nodes.length).toBe(2)
      const internalNodes = result2.nodes.filter(n => n.id === 'node1')
      expect(internalNodes.length).toBe(1)
    })

    it('should not duplicate edges when expanding already expanded package', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
      ]

      const pkgNode = packageNodes(nodes, edges, ['node1', 'node2'], { x: 150, y: 100 })

      // First expand
      const result1 = expandPackage(pkgNode, [pkgNode], [])

      // After first expand, edges should contain recreated internal edge
      expect(result1.edges.length).toBe(1)
      expect(result1.edges[0].id).toBe('e1')

      // Second expand (re-expand without collapsing)
      const result2 = expandPackage(pkgNode, result1.nodes, result1.edges)

      // Count edges with the preserved original ID
      const edgeCount = result2.edges.filter(e => e.id === 'e1').length
      expect(edgeCount).toBe(1) // Should only have one edge, not duplicated
    })

    it('should correctly expand after collapse and re-expand', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
      ]

      const pkgNode = packageNodes(nodes, edges, ['node1', 'node2'], { x: 150, y: 100 })

      // Expand
      const result1 = expandPackage(pkgNode, [pkgNode], [])
      expect(result1.nodes.length).toBe(3) // pkg + 2 internal
      expect(result1.edges.length).toBe(1) // 1 internal edge

      // Collapse
      const collapsed = collapsePackage(pkgNode.id, result1.nodes, result1.edges)
      expect(collapsed.nodes.length).toBe(1) // only pkg
      expect(collapsed.edges.length).toBe(0) // no edges after collapse

      // Re-expand using collapsed state
      const pkgAfterCollapse = collapsed.nodes[0]
      const reexpanded = expandPackage(pkgAfterCollapse as any, collapsed.nodes, collapsed.edges)

      expect(reexpanded.nodes.length).toBe(3) // pkg + 2 internal
      const internalNode1 = reexpanded.nodes.find(n => n.id === 'node1')
      const internalNode2 = reexpanded.nodes.find(n => n.id === 'node2')
      expect(internalNode1).toBeDefined()
      expect(internalNode2).toBeDefined()

      // Edge should be recreated (not duplicated)
      const edgeCount = reexpanded.edges.filter(e => e.id === 'e1').length
      expect(edgeCount).toBe(1)
    })

    it('should expand copied packages without deleting unrelated nodes with matching internal IDs', () => {
      const existingNode = createNode('node1', 'relu', 'ExistingReLU', { x: 0, y: 0 })
      const internalNode = createNode('node1', 'relu', 'InnerReLU', { x: 100, y: 100 })
      const internalNode2 = createNode('node2', 'linear', 'InnerLinear', { x: 200, y: 100 })
      const pkgNode = packageNodes([internalNode, internalNode2], [createEdge('e1', 'node1', 'node2')], ['node1', 'node2'], { x: 150, y: 100 })
      const copiedPkg = { ...pkgNode, id: 'copied-package' } as FlowHamsterNode

      const result = expandPackage(copiedPkg as any, [existingNode, copiedPkg], [])

      expect(result.nodes.find(n => n.id === 'node1' && n.data.label === 'ExistingReLU')).toBeDefined()
      expect(result.nodes.find(n => n.id === 'node1_1')).toBeDefined()
      expect(result.edges.find(e => e.id === 'e1')).toMatchObject({ source: 'node1_1', target: 'node2' })
    })
  })

  describe('collapsePackage()', () => {
    it('should collapse expanded package back to collapsed state', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
      ]

      const pkgNode = packageNodes(nodes, edges, ['node1', 'node2'], { x: 150, y: 100 })

      // First expand
      const expanded = expandPackage(pkgNode, [pkgNode], [])

      // Then collapse
      const result = collapsePackage(pkgNode.id, expanded.nodes, expanded.edges)

      // The group should be marked as collapsed
      const collapsedPkg = result.nodes.find(n => n.id === pkgNode.id)
      expect(collapsedPkg).toBeDefined()
      expect((collapsedPkg as any)?.data?.isExpanded).toBe(false)
    })
  })

  describe('getNextModuleName()', () => {
    it('should derive module names from the active graph instead of a persistent counter', () => {
      const name1 = getNextModuleName()
      const nameAfterReset = getNextModuleName()

      expect(name1).toBe('Module_1')
      expect(nameAfterReset).toBe('Module_1')
    })

    it('should skip active custom class IDs and visible node IDs', () => {
      const activeClassNode = {
        ...createNode('renamed-node-id', 'custom', 'EncoderBlock'),
        data: {
          nodeType: 'custom' as any,
          label: 'EncoderBlock',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'Module_1',
          isExpanded: false,
          internalStructure: [],
          internalEdges: [],
          outputVar: 'x',
          inputs: [],
          outputs: [],
          childNodeIds: [],
          internalEdgeIds: [],
        },
      } as FlowHamsterNode
      const renamedClassOldIdNode = {
        ...activeClassNode,
        id: 'Module_1',
        data: {
          ...activeClassNode.data,
          label: 'BetterBlock',
          customClassId: 'BetterBlock',
        },
      } as FlowHamsterNode
      const visibleModule2Node = createNode('Module_2', 'relu', 'VisibleModule2')

      expect(getNextModuleNameForNodes([activeClassNode])).toBe('Module_2')
      expect(getNextModuleNameForNodes([renamedClassOldIdNode])).toBe('Module_2')
      expect(getNextModuleNameForNodes([activeClassNode, visibleModule2Node])).toBe('Module_3')
    })
  })

  describe('computeTopologicalLayout()', () => {
    it('should compute correct topological order', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('a', 'input', 'Input', { x: 0, y: 0 }),
        createNode('b', 'relu', 'ReLU', { x: 100, y: 0 }),
        createNode('c', 'linear', 'Linear', { x: 200, y: 0 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'a', 'b'),
        createEdge('e2', 'b', 'c'),
      ]

      const layout = computeTopologicalLayout(nodes, edges)

      expect(layout.get('a')?.layer).toBe(0)
      expect(layout.get('b')?.layer).toBe(1)
      expect(layout.get('c')?.layer).toBe(2)
    })

    it('should handle diamond dependencies', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('a', 'input', 'Input', { x: 0, y: 0 }),
        createNode('b1', 'relu', 'ReLU1', { x: 100, y: 0 }),
        createNode('b2', 'relu', 'ReLU2', { x: 100, y: 50 }),
        createNode('c', 'linear', 'Linear', { x: 200, y: 25 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'a', 'b1'),
        createEdge('e2', 'a', 'b2'),
        createEdge('e3', 'b1', 'c'),
        createEdge('e4', 'b2', 'c'),
      ]

      const layout = computeTopologicalLayout(nodes, edges)

      expect(layout.get('a')?.layer).toBe(0)
      expect(layout.get('b1')?.layer).toBe(1)
      expect(layout.get('b2')?.layer).toBe(1)
      expect(layout.get('c')?.layer).toBe(2)
    })
  })
})
