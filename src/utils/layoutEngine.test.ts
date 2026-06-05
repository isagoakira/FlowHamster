import { describe, it, expect } from 'vitest'
import { Node, Edge } from 'reactflow'
import { computeLayout, hasOverlappingNodes } from './layoutEngine'

describe('layoutEngine', () => {
  function makeNodes(count: number): Node[] {
    return Array.from({ length: count }, (_, i) => ({
      id: `n${i}`,
      type: 'default',
      position: { x: Math.random() * 1000, y: Math.random() * 1000 },
      data: {},
    }))
  }

  function makeChainEdges(count: number): Edge[] {
    const edges: Edge[] = []
    for (let i = 0; i < count - 1; i++) {
      edges.push({ id: `e${i}`, source: `n${i}`, target: `n${i + 1}` })
    }
    return edges
  }

  function makeBranchEdges(): Edge[] {
    return [
      { id: 'e0', source: 'n0', target: 'n1' },
      { id: 'e1', source: 'n0', target: 'n2' },
      { id: 'e2', source: 'n1', target: 'n3' },
      { id: 'e3', source: 'n2', target: 'n3' },
    ]
  }

  describe('computeLayout', () => {
    it('should return empty result for empty node list', () => {
      const result = computeLayout([], [])
      expect(result.nodes).toEqual([])
      expect(result.edges).toEqual([])
      expect(result.bounds.width).toBe(0)
      expect(result.durationMs).toBe(0)
    })

    it('should not produce overlapping nodes for a linear chain', () => {
      const nodes = makeNodes(5)
      const edges = makeChainEdges(5)
      const result = computeLayout(nodes, edges)
      expect(hasOverlappingNodes(result.nodes)).toBe(false)
    })

    it('should not produce overlapping nodes for a branched graph', () => {
      const nodes = makeNodes(4)
      const edges = makeBranchEdges()
      const result = computeLayout(nodes, edges)
      expect(hasOverlappingNodes(result.nodes)).toBe(false)
    })

    it('should layout left-to-right by default (LR)', () => {
      const nodes: Node[] = [
        { id: 'a', type: 'default', position: { x: 0, y: 0 }, data: {} },
        { id: 'b', type: 'default', position: { x: 0, y: 0 }, data: {} },
        { id: 'c', type: 'default', position: { x: 0, y: 0 }, data: {} },
      ]
      const edges: Edge[] = [
        { id: 'e1', source: 'a', target: 'b' },
        { id: 'e2', source: 'b', target: 'c' },
      ]
      const result = computeLayout(nodes, edges, { direction: 'LR' })
      const posA = result.nodes.find((n) => n.id === 'a')!.position
      const posB = result.nodes.find((n) => n.id === 'b')!.position
      const posC = result.nodes.find((n) => n.id === 'c')!.position

      // In LR layout, later nodes should generally have larger x
      expect(posB.x).toBeGreaterThan(posA.x)
      expect(posC.x).toBeGreaterThan(posB.x)
    })

    it('should layout top-to-bottom when direction is TB', () => {
      const nodes: Node[] = [
        { id: 'a', type: 'default', position: { x: 0, y: 0 }, data: {} },
        { id: 'b', type: 'default', position: { x: 0, y: 0 }, data: {} },
        { id: 'c', type: 'default', position: { x: 0, y: 0 }, data: {} },
      ]
      const edges: Edge[] = [
        { id: 'e1', source: 'a', target: 'b' },
        { id: 'e2', source: 'b', target: 'c' },
      ]
      const result = computeLayout(nodes, edges, { direction: 'TB' })
      const posA = result.nodes.find((n) => n.id === 'a')!.position
      const posB = result.nodes.find((n) => n.id === 'b')!.position
      const posC = result.nodes.find((n) => n.id === 'c')!.position

      expect(posB.y).toBeGreaterThan(posA.y)
      expect(posC.y).toBeGreaterThan(posB.y)
    })

    it('should preserve child node positions (nodes with parentNode)', () => {
      const nodes: Node[] = [
        { id: 'parent', type: 'group', position: { x: 100, y: 100 }, data: {}, style: { width: 400, height: 300 } },
        { id: 'child', type: 'default', position: { x: 50, y: 60 }, data: {}, parentNode: 'parent' },
        { id: 'other', type: 'default', position: { x: 0, y: 0 }, data: {} },
      ]
      const edges: Edge[] = [
        { id: 'e1', source: 'other', target: 'parent' },
      ]
      const result = computeLayout(nodes, edges)
      const childNode = result.nodes.find((n) => n.id === 'child')!
      expect(childNode.position.x).toBe(50)
      expect(childNode.position.y).toBe(60)
    })

    it('should compute non-zero bounds', () => {
      const nodes = makeNodes(3)
      const edges = makeChainEdges(3)
      const result = computeLayout(nodes, edges)
      expect(result.bounds.width).toBeGreaterThan(0)
      expect(result.bounds.height).toBeGreaterThan(0)
    })

    it('should layout 100 nodes within 100ms', () => {
      const nodes = makeNodes(100)
      const edges = makeChainEdges(100)
      const result = computeLayout(nodes, edges)
      expect(result.durationMs).toBeLessThan(100)
      expect(hasOverlappingNodes(result.nodes)).toBe(false)
    })

    it('should use custom node dimensions when provided', () => {
      const nodes: Node[] = [
        { id: 'wide', type: 'default', position: { x: 0, y: 0 }, data: {}, style: { width: 300, height: 60 } },
        { id: 'tall', type: 'default', position: { x: 0, y: 0 }, data: {}, style: { width: 120, height: 200 } },
      ]
      const edges: Edge[] = [{ id: 'e1', source: 'wide', target: 'tall' }]
      const result = computeLayout(nodes, edges, { nodeSep: 40, rankSep: 80 })
      expect(hasOverlappingNodes(result.nodes)).toBe(false)
    })
  })

  describe('hasOverlappingNodes', () => {
    it('should return false for non-overlapping nodes', () => {
      const nodes: Node[] = [
        { id: 'a', type: 'default', position: { x: 0, y: 0 }, data: {} },
        { id: 'b', type: 'default', position: { x: 500, y: 500 }, data: {} },
      ]
      expect(hasOverlappingNodes(nodes)).toBe(false)
    })

    it('should return true for overlapping nodes', () => {
      const nodes: Node[] = [
        { id: 'a', type: 'default', position: { x: 0, y: 0 }, data: {} },
        { id: 'b', type: 'default', position: { x: 10, y: 10 }, data: {} },
      ]
      expect(hasOverlappingNodes(nodes)).toBe(true)
    })

    it('should ignore child nodes (with parentNode)', () => {
      const nodes: Node[] = [
        { id: 'a', type: 'default', position: { x: 0, y: 0 }, data: {} },
        { id: 'b', type: 'default', position: { x: 0, y: 0 }, data: {}, parentNode: 'parent' },
      ]
      expect(hasOverlappingNodes(nodes)).toBe(false)
    })
  })
})
