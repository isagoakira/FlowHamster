import { describe, expect, it } from 'vitest'
import {
  packageNodes,
  unpackageGroup,
  expandPackage,
  collapsePackage,
} from './subgraphPackager'
import { FlowHamsterEdge, FlowHamsterNode } from '../types/graph'

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

function expectGraphIntegrity(nodes: FlowHamsterNode[], edges: FlowHamsterEdge[]) {
  const nodeIds = new Set(nodes.map((node) => node.id))
  const edgeIds = edges.map((edge) => edge.id).filter(Boolean)

  expect(nodeIds.size).toBe(nodes.length)
  expect(new Set(edgeIds).size).toBe(edgeIds.length)

  for (const edge of edges) {
    expect(nodeIds.has(edge.source), `missing source node ${edge.source} for edge ${edge.id}`).toBe(true)
    expect(nodeIds.has(edge.target), `missing target node ${edge.target} for edge ${edge.id}`).toBe(true)
  }
}

describe('package robustness invariants', () => {
  it('exposes unique package handles when different internal nodes use the same handle name', () => {
    const extA = createNode('ext-a', 'input', 'ExtA', { x: 0, y: 0 })
    const extB = createNode('ext-b', 'input', 'ExtB', { x: 0, y: 160 })
    const branchA = createNode('branch-a', 'relu', 'BranchA', { x: 160, y: 0 })
    const branchB = createNode('branch-b', 'relu', 'BranchB', { x: 160, y: 160 })
    const join = createNode('join', 'add', 'Join', { x: 320, y: 80 })

    const pkg = packageNodes(
      [extA, extB, branchA, branchB, join],
      [
        createEdge('in-a', 'ext-a', 'branch-a', 'result', 'x'),
        createEdge('in-b', 'ext-b', 'branch-b', 'result', 'x'),
        createEdge('a-join', 'branch-a', 'join', 'result', 'a'),
        createEdge('b-join', 'branch-b', 'join', 'result', 'b'),
      ],
      ['branch-a', 'branch-b', 'join'],
      { x: 220, y: 80 }
    )

    const inputHandles = pkg.data.inputs.map((input) => input.handleId)
    const boundaryHandles = (pkg.data.boundaryEdges ?? [])
      .filter((edge) => edge.direction === 'input')
      .map((edge) => edge.groupHandleId)

    expect(pkg.data.inputs).toHaveLength(2)
    expect(new Set(inputHandles).size).toBe(inputHandles.length)
    expect(boundaryHandles).toEqual(inputHandles)
  })

  it('keeps newly added external edges on an existing package port when unpackaging', () => {
    const extA = createNode('ext-a', 'input', 'ExtA', { x: 0, y: 0 })
    const extB = createNode('ext-b', 'input', 'ExtB', { x: 0, y: 120 })
    const inner = createNode('inner', 'relu', 'Inner', { x: 160, y: 60 })

    const pkg = packageNodes(
      [extA, extB, inner],
      [createEdge('in-a', 'ext-a', 'inner', 'result', 'x')],
      ['inner'],
      { x: 160, y: 60 }
    )
    const packageInputHandle = pkg.data.inputs[0].handleId

    const result = unpackageGroup(
      pkg.id,
      [extA, extB, pkg as FlowHamsterNode],
      [
        createEdge('in-a', 'ext-a', pkg.id, 'result', packageInputHandle),
        createEdge('in-b-new', 'ext-b', pkg.id, 'result', packageInputHandle),
      ]
    )

    expectGraphIntegrity(result.nodes, result.edges)
    expect(result.edges.find((edge) => edge.id === 'in-a')).toMatchObject({
      source: 'ext-a',
      target: 'inner',
      sourceHandle: 'result',
      targetHandle: 'x',
    })
    expect(result.edges.find((edge) => edge.id === 'in-b-new')).toMatchObject({
      source: 'ext-b',
      target: 'inner',
      sourceHandle: 'result',
      targetHandle: 'x',
    })
  })

  it('survives repeated expand and collapse without duplicating nodes or edges', () => {
    const a = createNode('a', 'relu', 'A', { x: 100, y: 100 })
    const b = createNode('b', 'linear', 'B', { x: 240, y: 100 })
    const pkg = packageNodes([a, b], [createEdge('a-b', 'a', 'b', 'result', 'x')], ['a', 'b'], { x: 170, y: 100 })

    const expanded = expandPackage(pkg, [pkg as FlowHamsterNode], [])
    const collapsed = collapsePackage(pkg.id, expanded.nodes, expanded.edges)
    const reexpanded = expandPackage(collapsed.nodes[0] as any, collapsed.nodes, collapsed.edges)

    expectGraphIntegrity(reexpanded.nodes, reexpanded.edges)
    expect(reexpanded.nodes.filter((node) => node.id === 'a')).toHaveLength(1)
    expect(reexpanded.nodes.filter((node) => node.id === 'b')).toHaveLength(1)
    expect(reexpanded.edges.filter((edge) => edge.id === 'a-b')).toHaveLength(1)
  })
})
