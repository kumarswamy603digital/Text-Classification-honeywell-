/** Metric card used on Dashboard and Comparison pages. */
import { cn } from '../../utils/format'

export function MetricCard({
  label,
  value,
  sub,
  tone = 'default',
}: {
  label: string
  value: string
  sub?: string
  tone?: 'default' | 'good' | 'bad' | 'accent'
}) {
  const tones = {
    default: 'text-ink',
    good: 'text-good',
    bad: 'text-bad',
    accent: 'text-accent',
  }
  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <p className="text-xs font-medium uppercase tracking-wider text-muted">{label}</p>
      <p className={cn('mt-1.5 text-2xl font-bold tabular-nums', tones[tone])}>{value}</p>
      {sub && <p className="mt-1 text-xs text-muted">{sub}</p>}
    </div>
  )
}
