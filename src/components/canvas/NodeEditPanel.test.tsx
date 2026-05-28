import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { act } from 'react'
import { createRoot, Root } from 'react-dom/client'
import NodeEditPanel from './NodeEditPanel'
import { useGraphStore } from '../../hooks/useGraphStore'
import { FlowHamsterNode } from '../../types/graph'

function makeLinearNode(): FlowHamsterNode {
  return {
    id: 'linear-1',
    type: 'linearNode',
    position: { x: 0, y: 0 },
    data: {
      nodeType: 'linear',
      label: 'Linear',
      params: { in_features: 4, out_features: 2, bias: true },
    },
  }
}

function setInputValue(input: HTMLInputElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set
  setter?.call(input, value)
  input.dispatchEvent(new Event('input', { bubbles: true }))
  input.dispatchEvent(new Event('change', { bubbles: true }))
}

describe('NodeEditPanel', () => {
  let container: HTMLDivElement
  let root: Root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
    useGraphStore.setState({
      nodes: [],
      edges: [],
      selectedNodeIds: [],
      selectedEdgeIds: [],
      rfSetNodes: null,
      rfSetEdges: null,
      _history: [],
      _future: [],
    })
  })

  afterEach(() => {
    act(() => {
      root.unmount()
    })
    container.remove()
  })

  it('keeps parameter inputs bound to the live store node after each edit', () => {
    const staleNode = makeLinearNode()
    act(() => {
      useGraphStore.setState({ nodes: [staleNode] })
      root.render(<NodeEditPanel node={staleNode} onClose={() => {}} />)
    })

    const outFeaturesInput = Array.from(container.querySelectorAll('input[type="number"]'))
      .find((input) => (input as HTMLInputElement).value === '2') as HTMLInputElement | undefined
    expect(outFeaturesInput).toBeDefined()

    act(() => {
      setInputValue(outFeaturesInput!, '8')
    })

    expect(useGraphStore.getState().nodes[0].data.params.out_features).toBe(8)
    expect(outFeaturesInput!.value).toBe('8')
  })
})
