/**
 * Unit tests for useGraphStore - Package/Unpackage operations
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { act } from 'react'
import { useGraphStore } from './useGraphStore'
import { FlowHamsterNode, FlowHamsterEdge } from '../types/graph'
import { getAllCustomClasses, registerCustomClass } from '../utils/customCompositeRegistry'

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
function createEdge(id: string, source: string, target: string): FlowHamsterEdge {
  return {
    id,
    source,
    target,
    sourceHandle: null,
    targetHandle: null,
  }
}

describe('useGraphStore - Package operations', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorageMock.clear()

    // Reset store state
    useGraphStore.setState({
      nodes: [],
      edges: [],
      selectedNodeIds: [],
      selectedEdgeIds: [],
      rfSetNodes: undefined,
      rfSetEdges: undefined,
      _history: [],
      _future: [],
    })
  })

  describe('packageSelection()', () => {
    it('should package selected nodes into a custom composite node', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [
        createEdge('e1', 'node1', 'node2'),
      ]

      // Set up initial state
      act(() => {
        useGraphStore.setState({ nodes, edges })
        useGraphStore.setState({ selectedNodeIds: ['node1', 'node2'] })
      })

      // Execute package
      act(() => {
        useGraphStore.getState().packageSelection()
      })

      const state = useGraphStore.getState()

      // Should have 1 package node instead of 2 original nodes
      expect(state.nodes.length).toBe(1)
      const pkgNode = state.nodes[0]
      expect((pkgNode.data as any).isCustomComposite).toBe(true)
      expect((pkgNode.data as any).childNodeIds).toEqual(['node1', 'node2'])
    })

    it('should not package if fewer than 2 nodes selected', () => {
      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = []

      act(() => {
        useGraphStore.setState({ nodes, edges })
        useGraphStore.setState({ selectedNodeIds: ['node1'] })
      })

      act(() => {
        useGraphStore.getState().packageSelection()
      })

      const state = useGraphStore.getState()

      // Should still have the original node (no packaging happened)
      expect(state.nodes.length).toBe(1)
    })

    it('should overwrite a stale same-name registry class with the current package structure', () => {
      registerCustomClass(
        'Module_1',
        'custom',
        'other',
        '📦',
        'stale',
        [{ id: 'stale-node', type: 'relu', label: 'Stale', params: {} }],
        [],
        'stale-node'
      )

      const nodes: FlowHamsterNode[] = [
        createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 }),
        createNode('node2', 'linear', 'Linear1', { x: 200, y: 100 }),
      ]
      const edges: FlowHamsterEdge[] = [createEdge('e1', 'node1', 'node2')]

      act(() => {
        useGraphStore.setState({ nodes, edges, selectedNodeIds: ['node1', 'node2'] })
        useGraphStore.getState().packageSelection()
      })

      const module1 = getAllCustomClasses().find((cls) => cls.name === 'Module_1')
      expect(module1?.internalStructure.map((sub) => sub.id)).toEqual(['node1', 'node2'])
      expect(module1?.internalEdges[0]).toMatchObject({ id: 'e1', from: 'node1', to: 'node2' })
    })

    it('should allow an existing custom instance to be nested inside a new package', () => {
      registerCustomClass(
        'Module_1',
        'custom',
        'other',
        '📦',
        'inner',
        [{ id: 'inner-node', type: 'relu', label: 'Inner', params: {} }],
        [],
        'inner-node'
      )

      const module1Instance: FlowHamsterNode = {
        id: 'module1-instance',
        type: 'customNode',
        position: { x: 100, y: 100 },
        data: {
          nodeType: 'custom',
          label: 'encoder_0',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'Module_1',
          isExpanded: false,
          internalStructure: [{ id: 'inner-node', type: 'relu', label: 'Inner', params: {} }],
          internalEdges: [],
          outputVar: 'inner-node',
          inputs: [],
          outputs: [],
          childNodeIds: ['inner-node'],
          internalEdgeIds: [],
          boundaryEdges: [],
        },
      }
      const tailNode = createNode('tail', 'linear', 'Tail', { x: 260, y: 100 })

      act(() => {
        useGraphStore.setState({
          nodes: [module1Instance, tailNode],
          edges: [createEdge('e1', 'module1-instance', 'tail')],
          selectedNodeIds: ['module1-instance', 'tail'],
        })
        useGraphStore.getState().packageSelection()
      })

      const state = useGraphStore.getState()
      const module2 = state.nodes.find((node) => (node.data as any).customClassId === 'Module_2')
      expect(module2).toBeDefined()
      const innerModule = (module2!.data as any).internalStructure.find((sub: any) => sub.id === 'module1-instance')
      expect(innerModule.customClassId).toBe('Module_1')
      expect(innerModule.data.customClassId).toBe('Module_1')
      expect(innerModule.data.label).toBe('encoder_0')
    })

    it('should route same-named internal handles through distinct package ports and restore them on unpackage', () => {
      const extA = createNode('ext-a', 'input', 'ExtA', { x: 0, y: 0 })
      const extB = createNode('ext-b', 'input', 'ExtB', { x: 0, y: 160 })
      const branchA = createNode('branch-a', 'relu', 'BranchA', { x: 160, y: 0 })
      const branchB = createNode('branch-b', 'relu', 'BranchB', { x: 160, y: 160 })
      const join = createNode('join', 'add', 'Join', { x: 320, y: 80 })
      const edges: FlowHamsterEdge[] = [
        { id: 'in-a', source: 'ext-a', target: 'branch-a', sourceHandle: 'result', targetHandle: 'x' },
        { id: 'in-b', source: 'ext-b', target: 'branch-b', sourceHandle: 'result', targetHandle: 'x' },
        { id: 'a-join', source: 'branch-a', target: 'join', sourceHandle: 'result', targetHandle: 'a' },
        { id: 'b-join', source: 'branch-b', target: 'join', sourceHandle: 'result', targetHandle: 'b' },
      ]

      act(() => {
        useGraphStore.setState({
          nodes: [extA, extB, branchA, branchB, join],
          edges,
          selectedNodeIds: ['branch-a', 'branch-b', 'join'],
        })
        useGraphStore.getState().packageSelection()
      })

      let state = useGraphStore.getState()
      const pkgNode = state.nodes.find((node) => (node.data as any).isCustomComposite)!
      const inputHandles = ((pkgNode.data as any).inputs as any[]).map((input) => input.handleId)
      const redirectedInputHandles = state.edges
        .filter((edge) => edge.target === pkgNode.id)
        .map((edge) => edge.targetHandle)

      expect(inputHandles).toHaveLength(2)
      expect(new Set(inputHandles).size).toBe(2)
      expect(new Set(redirectedInputHandles)).toEqual(new Set(inputHandles))

      act(() => {
        useGraphStore.getState().unpackageGroup(pkgNode.id)
      })

      state = useGraphStore.getState()
      expect(state.edges.find((edge) => edge.id === 'in-a')).toMatchObject({
        source: 'ext-a',
        target: 'branch-a',
        targetHandle: 'x',
      })
      expect(state.edges.find((edge) => edge.id === 'in-b')).toMatchObject({
        source: 'ext-b',
        target: 'branch-b',
        targetHandle: 'x',
      })
    })
  })

  describe('addNode()', () => {
    it('should instantiate reusable custom classes from the registry multiple times', () => {
      const registered = registerCustomClass(
        'ReusableBlock',
        'custom',
        'other',
        '📦',
        'reusable',
        [
          { id: 'inner-relu', type: 'relu', label: 'InnerReLU', params: {}, position: { x: 10, y: 10 } },
          { id: 'inner-linear', type: 'linear', label: 'InnerLinear', params: {}, position: { x: 120, y: 10 } },
        ],
        [{ id: 'inner-edge', from: 'inner-relu', to: 'inner-linear' }],
        'inner-linear',
        {
          inputs: [{ id: 'input_0', handleId: 'input_0', label: 'input', nodeId: 'inner-relu', edgeId: 'in-edge' }],
          outputs: [{ id: 'output_0', handleId: 'output_0', label: 'output', nodeId: 'inner-linear', edgeId: 'out-edge' }],
        }
      )

      act(() => {
        useGraphStore.getState().addNode({
          nodeType: 'custom',
          label: 'block_a',
          params: {},
          customClassRegistryId: registered.id,
          customClassId: registered.name,
        } as any, { x: 100, y: 100 })
        useGraphStore.getState().addNode({
          nodeType: 'custom',
          label: 'block_b',
          params: {},
          customClassRegistryId: registered.id,
          customClassId: registered.name,
        } as any, { x: 300, y: 100 })
      })

      const state = useGraphStore.getState()
      expect(state.nodes).toHaveLength(2)
      expect(new Set(state.nodes.map((node) => node.id)).size).toBe(2)
      expect(state.nodes.map((node) => node.data.label)).toEqual(['block_a', 'block_b'])
      for (const node of state.nodes) {
        expect(node.type).toBe('customNode')
        expect((node.data as any).isCustomComposite).toBe(true)
        expect((node.data as any).customClassId).toBe('ReusableBlock')
        expect((node.data as any).customClassRegistryId).toBe(registered.id)
        expect((node.data as any).internalStructure.map((sub: any) => sub.id)).toEqual(['inner-relu', 'inner-linear'])
        expect((node.data as any).inputs[0].handleId).toBe('input_0')
        expect((node.data as any).outputs[0].handleId).toBe('output_0')
      }
    })
  })

  describe('unpackageGroup()', () => {
    it('should unpackage a group and restore original nodes', () => {
      // Create a pre-packaged state
      const pkgNode: FlowHamsterNode = {
        id: 'Module_1',
        type: 'customNode',
        position: { x: 150, y: 100 },
        data: {
          nodeType: 'custom',
          label: 'Module_1',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'Module_1',
          isExpanded: false,
          internalStructure: [
            { id: 'node1', type: 'relu', label: 'ReLU1', params: {}, position: { x: 100, y: 100 } },
            { id: 'node2', type: 'linear', label: 'Linear1', params: {}, position: { x: 200, y: 100 } },
          ],
          internalEdges: [{ from: 'node1', to: 'node2' }],
          outputVar: 'node2',
          inputs: [],
          outputs: [],
          childNodeIds: ['node1', 'node2'],
          internalEdgeIds: ['e1'],
          boundaryEdges: [],
        },
      }

      act(() => {
        useGraphStore.setState({ nodes: [pkgNode], edges: [] })
      })

      act(() => {
        useGraphStore.getState().unpackageGroup('Module_1')
      })

      const state = useGraphStore.getState()

      // Should restore 2 original nodes
      expect(state.nodes.length).toBe(2)
      const restoredNode1 = state.nodes.find(n => n.id === 'node1')
      const restoredNode2 = state.nodes.find(n => n.id === 'node2')
      expect(restoredNode1).toBeDefined()
      expect(restoredNode2).toBeDefined()
    })

    it('should dissolve the selected instance even when another instance uses the same class', () => {
      const makePkg = (id: string): FlowHamsterNode => ({
        id,
        type: 'customNode',
        position: { x: 150, y: 100 },
        data: {
          nodeType: 'custom',
          label: id,
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'Module_1',
          isExpanded: false,
          internalStructure: [
            { id: 'node1', type: 'relu', label: 'ReLU1', params: {}, position: { x: 100, y: 100 } },
            { id: 'node2', type: 'linear', label: 'Linear1', params: {}, position: { x: 200, y: 100 } },
          ],
          internalEdges: [{ id: 'e1', from: 'node1', to: 'node2' }],
          outputVar: 'node2',
          inputs: [],
          outputs: [],
          childNodeIds: ['node1', 'node2'],
          internalEdgeIds: ['e1'],
          boundaryEdges: [],
        },
      })

      act(() => {
        useGraphStore.setState({ nodes: [makePkg('instance-a'), makePkg('instance-b')], edges: [] })
        useGraphStore.getState().unpackageGroup('instance-b')
      })

      const state = useGraphStore.getState()
      expect(state.nodes.find(n => n.id === 'instance-a')).toBeDefined()
      expect(state.nodes.find(n => n.id === 'instance-b')).toBeUndefined()
      expect(state.nodes.find(n => n.id === 'node1')).toBeDefined()
      expect(state.edges.find(e => e.id === 'e1')).toMatchObject({ source: 'node1', target: 'node2' })
  })
    })

  describe('expandGroup() / collapseGroup()', () => {
    it('should expand a package to show internal structure', () => {
      const pkgNode: FlowHamsterNode = {
        id: 'Module_1',
        type: 'customNode',
        position: { x: 150, y: 100 },
        data: {
          nodeType: 'custom',
          label: 'Module_1',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'Module_1',
          isExpanded: false,
          internalStructure: [
            { id: 'node1', type: 'relu', label: 'ReLU1', params: {}, position: { x: 100, y: 100 } },
          ],
          internalEdges: [],
          outputVar: 'node1',
          inputs: [],
          outputs: [],
          childNodeIds: ['node1'],
          internalEdgeIds: [],
          boundaryEdges: [],
        },
      }

      act(() => {
        useGraphStore.setState({ nodes: [pkgNode], edges: [] })
      })

      act(() => {
        useGraphStore.getState().expandGroup('Module_1')
      })

      const state = useGraphStore.getState()

      // Should have expanded the package node
      const expandedPkg = state.nodes.find(n => n.id === 'Module_1')
      expect(expandedPkg).toBeDefined()
      expect((expandedPkg?.data as any).isExpanded).toBe(true)

      // Internal node should be visible
      const internalNode = state.nodes.find(n => n.id === 'node1')
      expect(internalNode).toBeDefined()
    })

    it('should collapse expanded package back to package node', () => {
      const pkgNode: FlowHamsterNode = {
        id: 'Module_1',
        type: 'customNode',
        position: { x: 150, y: 100 },
        data: {
          nodeType: 'custom',
          label: 'Module_1',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'Module_1',
          isExpanded: true, // Already expanded
          internalStructure: [
            { id: 'node1', type: 'relu', label: 'ReLU1', params: {}, position: { x: 100, y: 100 } },
          ],
          internalEdges: [],
          outputVar: 'node1',
          inputs: [],
          outputs: [],
          childNodeIds: ['node1'],
          internalEdgeIds: [],
          boundaryEdges: [],
        },
      }

      const internalNode = createNode('node1', 'relu', 'ReLU1', { x: 100, y: 100 })

      act(() => {
        useGraphStore.setState({ nodes: [pkgNode, internalNode], edges: [] })
      })

      act(() => {
        useGraphStore.getState().collapseGroup('Module_1')
      })

      const state = useGraphStore.getState()

      // Should have collapsed the package
      const collapsedPkg = state.nodes.find(n => n.id === 'Module_1')
      expect(collapsedPkg).toBeDefined()
      expect((collapsedPkg?.data as any).isExpanded).toBe(false)

      // Internal node should be hidden (not in nodes array)
      const internalNodeStill = state.nodes.find(n => n.id === 'node1')
      expect(internalNodeStill).toBeUndefined()
    })
  })

  describe('renamePackage()', () => {
    it('should rename a package instance label', () => {
      const pkgNode: FlowHamsterNode = {
        id: 'Module_1',
        type: 'customNode',
        position: { x: 150, y: 100 },
        data: {
          nodeType: 'custom',
          label: 'Module_1',
          params: {},
          isComposite: true,
          isCustomComposite: true,
          customClassId: 'Module_1',
          isExpanded: false,
          internalStructure: [],
          internalEdges: [],
          outputVar: '',
          inputs: [],
          outputs: [],
          childNodeIds: [],
          internalEdgeIds: [],
          boundaryEdges: [],
        },
      }

      act(() => {
        useGraphStore.setState({ nodes: [pkgNode], edges: [] })
      })

      act(() => {
        useGraphStore.getState().renamePackage('Module_1', 'MyRenamedModule')
      })

      const state = useGraphStore.getState()
      const renamedNode = state.nodes.find(n => n.id === 'Module_1')

      expect((renamedNode?.data as any).label).toBe('MyRenamedModule')
    })

    it('should rename a custom class across instances, nested snapshots, and the registry', () => {
      registerCustomClass(
        'Module_1',
        'custom',
        'other',
        '📦',
        'inner',
        [{ id: 'inner-node', type: 'relu', label: 'Inner', params: {} }],
        [],
        'inner-node'
      )
      registerCustomClass(
        'Module_2',
        'custom',
        'other',
        '📦',
        'outer',
        [{
          id: 'module1-instance',
          type: 'custom',
          label: 'Module_1',
          params: {},
          customClassId: 'Module_1',
          data: {
            nodeType: 'custom' as any,
            label: 'Module_1',
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
        }],
        [],
        'module1-instance'
      )

      const defaultLabelInstance: FlowHamsterNode = {
        id: 'module-a',
        type: 'customNode',
        position: { x: 0, y: 0 },
        data: {
          nodeType: 'custom',
          label: 'Module_1',
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
          boundaryEdges: [],
        },
      }
      const customLabelInstance: FlowHamsterNode = {
        ...defaultLabelInstance,
        id: 'module-b',
        data: {
          ...defaultLabelInstance.data,
          label: 'encoder_0',
        },
      }
      const outerInstance: FlowHamsterNode = {
        ...defaultLabelInstance,
        id: 'module-c',
        data: {
          ...defaultLabelInstance.data,
          label: 'Module_2',
          customClassId: 'Module_2',
          internalStructure: [{
            id: 'module1-instance',
            type: 'custom',
            label: 'Module_1',
            params: {},
            customClassId: 'Module_1',
            data: {
              nodeType: 'custom' as any,
              label: 'Module_1',
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
          }],
        },
      }

      act(() => {
        useGraphStore.setState({ nodes: [defaultLabelInstance, customLabelInstance, outerInstance], edges: [] })
        useGraphStore.getState().renamePackageClass('module-a', 'Better Block')
      })

      const state = useGraphStore.getState()
      const renamedA = state.nodes.find(n => n.id === 'module-a')!
      const renamedB = state.nodes.find(n => n.id === 'module-b')!
      const renamedOuter = state.nodes.find(n => n.id === 'module-c')!
      expect((renamedA.data as any).customClassId).toBe('Better_Block')
      expect((renamedA.data as any).label).toBe('Better_Block')
      expect((renamedB.data as any).customClassId).toBe('Better_Block')
      expect((renamedB.data as any).label).toBe('encoder_0')
      expect((renamedOuter.data as any).internalStructure[0].customClassId).toBe('Better_Block')
      expect((renamedOuter.data as any).internalStructure[0].data.customClassId).toBe('Better_Block')

      const classes = getAllCustomClasses()
      expect(classes.find(cls => cls.name === 'Better_Block')).toBeDefined()
      expect(classes.find(cls => cls.name === 'Module_2')?.internalStructure[0].customClassId).toBe('Better_Block')
    })

    it('should allow class rename to replace a stale unused registry name', () => {
      registerCustomClass(
        'Module_1',
        'custom',
        'other',
        '📦',
        'active',
        [{ id: 'active-node', type: 'relu', label: 'Active', params: {} }],
        [],
        'active-node'
      )
      registerCustomClass(
        'Better_Block',
        'custom',
        'other',
        '📦',
        'stale',
        [{ id: 'stale-node', type: 'relu', label: 'Stale', params: {} }],
        [],
        'stale-node'
      )

      const pkgNode: FlowHamsterNode = {
        id: 'module-a',
        type: 'customNode',
        position: { x: 0, y: 0 },
        data: {
          nodeType: 'custom',
          label: 'Module_1',
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
          boundaryEdges: [],
        },
      }

      act(() => {
        useGraphStore.setState({ nodes: [pkgNode], edges: [] })
        useGraphStore.getState().renamePackageClass('module-a', 'Better Block')
      })

      const renamedNode = useGraphStore.getState().nodes.find(n => n.id === 'module-a')!
      expect((renamedNode.data as any).customClassId).toBe('Better_Block')
      const classes = getAllCustomClasses()
      expect(classes.filter(cls => cls.name === 'Better_Block')).toHaveLength(1)
      expect(classes.find(cls => cls.name === 'Better_Block')?.internalStructure[0].id).toBe('active-node')
    })
  })
})
