// Re-export everything from the single source of truth
export type { Template, TemplateNode, TemplateEdge, TemplateGraph } from '../hooks/useTemplateStore'
export { useTemplateStore, LOCAL_TEMPLATES } from '../hooks/useTemplateStore'

// Backward compatibility: re-export LOCAL_TEMPLATES as TEMPLATES
import { LOCAL_TEMPLATES } from '../hooks/useTemplateStore'
export { LOCAL_TEMPLATES as TEMPLATES }
