import { useEffect, useRef, useState } from 'react'
import { useGraphStore } from './useGraphStore'
import { WS_CODE_URL } from '../utils/runtimeConfig'

export function useWebSocketCode() {
  const [connected, setConnected] = useState(false)
  const [code, setCode] = useState('')
  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const nodes = useGraphStore(s => s.nodes)
  const edges = useGraphStore(s => s.edges)

  const connect = () => {
    const ws = new WebSocket(WS_CODE_URL)
    ws.onopen = () => setConnected(true)
    ws.onclose = () => { setConnected(false); wsRef.current = null }
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.code) setCode(data.code)
    }
    wsRef.current = ws
  }

  const disconnect = () => {
    wsRef.current?.close()
    wsRef.current = null
    setConnected(false)
  }

  // 防抖发送：节点/边变动后 300ms 再推送
  useEffect(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      wsRef.current?.send(JSON.stringify({ nodes, edges }))
    }, 300)
  }, [nodes, edges])

  return { connected, code, setCode, connect, disconnect }
}
