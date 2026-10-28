/** Loading / empty / error state components used across all pages. */
import { AlertTriangle, Inbox, RefreshCw } from 'lucide-react'
import * as React from 'react'
import { Button, Card } from './primitives'

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div role="status" aria-live="polite" className="space-y-3 py-4">
      <span className="sr-only">{label}</span>
      <div className="skeleton h-4 w-2/3 rounded" />
      <div className="skeleton h-4 w-full rounded" />
      <div className="skeleton h-4 w-5/6 rounded" />
    </div>
  )
}

export function EmptyState({
  title,
  message,
  action,
}: {
  title: string
  message: string
  action?: React.ReactNode
}) {
  return (
    <div className="flex flex-col items-center gap-3 py-10 text-center">
      <Inbox className="h-8 w-8 text-muted" aria-hidden />
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="max-w-md text-sm text-muted">{message}</p>
      {action}
    </div>
  )
}

export function ErrorState({
  title = 'Something went wrong',
  message,
  onRetry,
  compact,
}: {
  title?: string
  message: string
  onRetry?: () => void
  compact?: boolean
}) {
  const body = (
    <div className={compact ? 'flex items-start gap-3' : 'flex flex-col items-center gap-3 py-8 text-center'}>
      <AlertTriangle className={compact ? 'h-5 w-5 shrink-0 text-warn' : 'h-8 w-8 text-warn'} aria-hidden />
      <div className={compact ? '' : 'flex flex-col items-center gap-2'}>
        <h3 className="text-sm font-semibold">{title}</h3>
        <p className="max-w-md text-sm text-muted">{message}</p>
        {onRetry && (
          <Button variant="secondary" onClick={onRetry} className="mt-2">
            <RefreshCw className="h-4 w-4" aria-hidden /> Retry
          </Button>
        )}
      </div>
    </div>
  )
  if (compact) return <div role="alert">{body}</div>
  return (
    <Card>
      <div role="alert">{body}</div>
    </Card>
  )
}
