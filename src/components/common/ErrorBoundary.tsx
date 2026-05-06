import { Component, ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
  level?: 'page' | 'section'
}

interface State {
  hasError: boolean
  error: Error | null
}

/**
 * ErrorBoundary component that catches JavaScript errors in child components.
 * Uses React error boundary API (getDerivedStateFromError, componentDidCatch).
 *
 * @param fallback Custom fallback UI to display when an error occurs
 * @param onError Callback invoked when an error is caught
 * @param level 'page' for full-page error, 'section' for inline section error
 */
export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    this.props.onError?.(error, errorInfo)
  }

  resetError = (): void => {
    this.setState({ hasError: false, error: null })
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      const { level = 'section' } = this.props

      if (level === 'page') {
        return (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100vh',
              width: '100vw',
              backgroundColor: '#f8d7da',
              color: '#721c24',
              fontFamily: 'system-ui, -apple-system, sans-serif',
              padding: '2rem',
              boxSizing: 'border-box',
            }}
          >
            <h1 style={{ margin: '0 0 1rem 0', fontSize: '1.5rem' }}>
              Something went wrong
            </h1>
            <p style={{ margin: '0 0 1.5rem 0', color: '#495057' }}>
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => window.location.href = '/'}
              style={{
                padding: '0.5rem 1.5rem',
                fontSize: '1rem',
                backgroundColor: '#dc3545',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Reload Page
            </button>
          </div>
        )
      }

      // Section-level fallback (inline error)
      return (
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
            backgroundColor: '#fff3cd',
            border: '1px solid #ffc107',
            borderRadius: '8px',
            color: '#856404',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            minHeight: '120px',
          }}
        >
          <div style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
            ⚠ Component Error
          </div>
          <p style={{ margin: '0 0 1rem 0', fontSize: '0.875rem', textAlign: 'center' }}>
            {this.state.error?.message || 'An unexpected error occurred in this section'}
          </p>
          <button
            onClick={this.resetError}
            style={{
              padding: '0.375rem 1rem',
              fontSize: '0.875rem',
              backgroundColor: '#ffc107',
              color: '#000',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Try Again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}