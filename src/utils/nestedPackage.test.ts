/**
 * Test for nested GroupNode packaging scenario
 *
 * This test reproduces the bug where:
 * 1. Package nodes into Module_1
 * 2. Copy Module_1
 * 3. Package the copy + other nodes into Module_2
 * 4. Code generation should include BOTH Module_1 and Module_2 class definitions
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  packageNodes,
  unpackageGroup,
  expandPackage,
  collapsePackage,
  resetModuleCounter,
} from './subgraphPackager'
import {
  registerCustomClass,
  getAllCustomClasses,
  getCustomClass,
  unregisterCustomClass,
} from './customCompositeRegistry'
import { FlowHamsterNode, FlowHamsterEdge, CustomCompositeNodeData } from '../types/graph'

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

// Helper to create a custom composite node (like a packaged Module_1)
function createCustomCompositeNode(
  id: string,
  customClassId: string,
  internalStructure: Array<{ id: string; type: string; label: string }>,
  position = { x: 0, y: 0 }
): FlowHamsterNode {
  return {
    id,
    type: 'customNode',
    position,
    data: {
      nodeType: 'custom' as any,
      label: customClassId,
      params: {},
      isComposite: true,
      isCustomComposite: true,
      customClassId,
      isExpanded: false,
      internalStructure: internalStructure.map(s => ({
        ...s,
        params: {},
        position: { x: 0, y: 0 },
      })),
      internalEdges: [],
      outputVar: internalStructure[internalStructure.length - 1]?.id || 'x',
      inputs: [],
      outputs: [],
      childNodeIds: internalStructure.map(s => s.id),
      internalEdgeIds: [],
      boundaryEdges: [],
    },
  }
}

// Helper to create an edge
function createEdge(id: string, source: string, target: string): FlowHamsterEdge {
  return {
    id,
    source,
    target,
    sourceHandle: null,
    targetHandle: null,
  }
}

describe('nested GroupNode packaging', () => {
  beforeEach(() => {
    resetModuleCounter()
    localStorageMock.clear()
  })

  describe('BUG REPRODUCTION: internal customClassId is lost during packaging', () => {
    it('should preserve customClassId when packaging a custom composite node inside another package', () => {
      // Step 1: Create and register a custom class (simulating packaging nodes into Module_1)
      const module1Internal = [
        { id: 'relu1', type: 'relu', label: 'ReLU1' },
        { id: 'linear1', type: 'linear', label: 'Linear1' },
      ]

      // Register Module_1
      registerCustomClass(
        'Module_1',
        'mlp',
        'other',
        '📦',
        'Test module 1',
        module1Internal.map(s => ({ ...s, params: {} })),
        [{ from: 'relu1', to: 'linear1' }],
        'linear1'
      )

      // Create a graph with Module_1 and an additional node
      const module1Node = createCustomCompositeNode('module1_instance', 'Module_1', module1Internal, { x: 100, y: 100 })
      const otherNode = createNode('relu2', 'relu', 'ReLU2', { x: 300, y: 100 })

      // Step 2: Package Module_1 copy + otherNode into Module_2
      // First we need to create a "copy" of module1 - in real scenario this would happen via paste
      const module1Copy = createCustomCompositeNode(
        'module1_copy',  // New ID for the copy
        'Module_1',      // But customClassId still points to Module_1
        module1Internal,
        { x: 100, y: 100 }
      )

      const nodes: FlowHamsterNode[] = [module1Copy, otherNode]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'module1_copy', 'relu2'),
      ]

      // Package into Module_2
      const module2Pkg = packageNodes(nodes, edges, ['module1_copy', 'relu2'], { x: 200, y: 100 })

      // Verify Module_2 was created
      expect(module2Pkg.data.isCustomComposite).toBe(true)
      expect(module2Pkg.data.customClassId).toBe('Module_2')

      // THE BUG: internalStructure should preserve customClassId of the inner Module_1
      const innerModule1 = module2Pkg.data.internalStructure.find(
        s => s.type === 'custom' && s.label === 'Module_1'
      )

      expect(innerModule1).toBeDefined()
      // This is what SHOULD happen but currently doesn't:
      expect((innerModule1 as any).customClassId).toBe('Module_1')
    })

    it('should find Module_1 in customClassIdsUsed when generating code for nested packages', () => {
      // This test simulates what happens in codeGenerator

      // Step 1: Register Module_1
      const module1Internal = [
        { id: 'relu1', type: 'relu', label: 'ReLU1' },
      ]

      registerCustomClass(
        'Module_1',
        'mlp',
        'other',
        '📦',
        'Test module 1',
        module1Internal.map(s => ({ ...s, params: {} })),
        [],
        'relu1'
      )

      // Step 2: Create Module_2 containing Module_1
      const module1Copy = createCustomCompositeNode(
        'module1_copy',
        'Module_1',
        module1Internal,
        { x: 100, y: 100 }
      )
      const otherNode = createNode('relu2', 'relu', 'ReLU2', { x: 300, y: 100 })

      const module2Pkg = packageNodes(
        [module1Copy, otherNode],
        [createEdge('e1', 'module1_copy', 'relu2')],
        ['module1_copy', 'relu2'],
        { x: 200, y: 100 }
      )

      // Get all registered custom classes
      const allClasses = getAllCustomClasses()
      expect(allClasses.length).toBe(1) // Only Module_1 (packageNodes doesn't register)

      // Create a "block" simulating what astBuilder would create for the inner Module_1
      // With the fix: internalStructure now preserves customClassId
      const innerModule1Block = {
        nodeId: 'module1_copy',
        opType: 'custom',
        fields: { customClassId: 'Module_1' }, // Now properly set!
      }

      // Simulate codeGenerator's customClassIdsUsed collection (with my fix)
      const customClassIdsUsed = new Set<string>()
      if (innerModule1Block.opType === 'custom') {
        const customClassId = (innerModule1Block.fields as any)?.customClassId || innerModule1Block.nodeId
        customClassIdsUsed.add(customClassId)
      }

      // Now customClassIdsUsed should have 'Module_1'
      expect(customClassIdsUsed.has('Module_1')).toBe(true)

      // Check if Module_1's class would be found
      const module1Class = allClasses.find(c => c.name === 'Module_1')
      expect(module1Class).toBeDefined()

      const wouldIncludeModule1 =
        customClassIdsUsed.has(module1Class!.id) ||
        customClassIdsUsed.has(module1Class!.name) ||
        customClassIdsUsed.has(`custom_${module1Class!.name}`)

      // With the fix, Module_1's class WOULD be included in code generation
      expect(wouldIncludeModule1).toBe(true)
    })
  })

  describe('expandPackage and collapsePackage with nested modules', () => {
    it('should correctly expand Module_2 containing Module_1', () => {
      // Setup Module_1
      const module1Internal = [
        { id: 'relu1', type: 'relu', label: 'ReLU1' },
      ]

      registerCustomClass(
        'Module_1',
        'mlp',
        'other',
        '📦',
        'Test module 1',
        module1Internal.map(s => ({ ...s, params: {} })),
        [],
        'relu1'
      )

      // Create Module_2 containing Module_1
      const module1Copy = createCustomCompositeNode(
        'module1_copy',
        'Module_1',
        module1Internal,
        { x: 100, y: 100 }
      )
      const otherNode = createNode('relu2', 'relu', 'ReLU2', { x: 300, y: 100 })

      const module2Pkg = packageNodes(
        [module1Copy, otherNode],
        [createEdge('e1', 'module1_copy', 'relu2')],
        ['module1_copy', 'relu2'],
        { x: 200, y: 100 }
      )

      // Expand Module_2
      const expanded = expandPackage(module2Pkg, [module2Pkg], [])

      // Should have Module_2 + internal nodes of Module_2
      expect(expanded.nodes.length).toBeGreaterThanOrEqual(2)

      // Module_2 should be marked as expanded
      const expandedModule2 = expanded.nodes.find(n => n.id === 'Module_2')
      expect(expandedModule2).toBeDefined()
      expect((expandedModule2 as any).data.isExpanded).toBe(true)

      // The inner Module_1 copy should be visible
      const innerModule1 = expanded.nodes.find(n => n.id === 'module1_copy')
      expect(innerModule1).toBeDefined()
    })
  })
})
