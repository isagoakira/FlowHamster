/**
 * Unit tests for useGraphStore - Package/Unpackage operations
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { act } from 'react'
import { useGraphStore } from './useGraphStore'
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
  })
})
