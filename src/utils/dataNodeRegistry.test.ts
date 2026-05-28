import { describe, expect, it } from 'vitest'
import { DATA_NODE_CATEGORIES } from './dataNodeRegistry'

describe('data node registry', () => {
  it('exposes CSV feature and label column parameters in the UI defaults', () => {
    const csvNode = DATA_NODE_CATEGORIES.sources.nodes.find((node) => node.type === 'csv_source')

    expect(csvNode).toBeDefined()
    expect(csvNode?.defaultParams).toMatchObject({
      path: './data/train.csv',
      delimiter: ',',
      feature_columns: 'f0,f1,f2,f3',
      label_column: 'label',
    })
    expect(csvNode?.fieldOrder).toEqual(['path', 'delimiter', 'feature_columns', 'label_column'])
  })
})
