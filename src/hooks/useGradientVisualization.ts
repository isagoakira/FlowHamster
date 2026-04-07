import { useState } from 'react'
import { API_BASE_URL } from '../utils/runtimeConfig'

export interface GradientEdge {
  source: string
  target: string
  gradient_strength: number
  direction: string
}

export interface GradientNode {
  id: string
  gradient_score: number
  risk: 'vanishing' | 'exploding' | 'normal'
}

export interface GradientData {
  edges: GradientEdge[]
  nodes: GradientNode[]
}

export function useGradientVisualization() {
  const [gradientData, setGradientData] = useState<GradientData | null>(null)
  const [gradientMode, setGradientMode] = useState(false)

  const analyzeGradients = async (nodes: any[], edges: any[]) => {
    try {
      const res = await fetch(`${API_BASE_URL}/execute/gradients`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes, edges }),
      })
      const data = await res.json()
      if (data.success) {
        setGradientData(data.data)
        setGradientMode(true)
      }
    } catch (err) {
      console.error('Gradient analysis failed:', err)
    }
  }

  const exitGradientMode = () => {
    setGradientMode(false)
    setGradientData(null)
  }

  return { gradientData, gradientMode, analyzeGradients, exitGradientMode }
}
