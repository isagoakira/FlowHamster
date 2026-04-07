import { useEffect } from 'react'
import { useGraphStore } from '../hooks/useGraphStore'

function isEditableTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false
  const tagName = target.tagName.toLowerCase()
  return (
    target.isContentEditable ||
    tagName === 'input' ||
    tagName === 'textarea' ||
    tagName === 'select'
  )
}

export function useKeyboardShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (isEditableTarget(e.target)) return

      const store = useGraphStore.getState()

      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        if (e.shiftKey) {
          store.redo()
        } else {
          store.undo()
        }
        e.preventDefault()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        store.redo()
        e.preventDefault()
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
        store.copySelection()
        e.preventDefault()
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') {
        store.pasteClipboard()
        e.preventDefault()
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        store.deleteSelection()
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])
}
