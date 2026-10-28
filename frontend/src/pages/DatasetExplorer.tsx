/** Dataset Explorer: provenance, distribution, length stats, quality notes. */
import { ExternalLink } from 'lucide-react'
import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card, Badge } from '../components/ui/primitives'
import { ErrorState, LoadingState } from '../components/ui/states'
import { DatasetStatistics } from '../components/analytics/DatasetStatistics'
import type { DatasetStats } from '../types/api'

type StatsWithTerms = DatasetStats & { top_terms?: Record<string, { term: string; count: number }[]> }

export default function DatasetExplorer() {
  const stats = useApi(() => api.datasetStats() as Promise<StatsWithTerms>)

  if (stats.loading) {
    return (
      <Card title="Dataset Explorer">
        <LoadingState label="Loading dataset statistics" />
      </Card>
    )
  }
  if (stats.error || !stats.data) {
    return (
      <ErrorState
        title="Dataset statistics unavailable"
        message={stats.error ?? 'Run scripts/explore_data.py and scripts/train_models.py first.'}
        onRetry={stats.reload}
      />
    )
  }

  const s = stats.data
  const topTerms = s.top_terms ?? {}

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-2xl font-bold tracking-tight">Dataset Explorer</h1>
        <Badge tone="info">{s.n_samples.toLocaleString()} labelled messages</Badge>
      </div>

      <Card title="Provenance">
        <div className="space-y-2 text-sm">
          <p className="font-semibold">{s.source.name}</p>
          <p className="text-muted">{s.source.notes}</p>
          <a
            href={s.source.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-accent hover:underline"
          >
            {s.source.url} <ExternalLink className="h-3.5 w-3.5" aria-hidden />
          </a>
          <p className="text-xs text-muted">License / terms: {s.source.license}</p>
        </div>
      </Card>

      <Card title="Summary statistics (computed at acquisition time)">
        <DatasetStatistics stats={s} />
      </Card>

      <Card
        title="Most frequent terms per class"
        subtitle="Frequency counts after stopword removal — common words, not automatically 'spammy' words"
      >
        <div className="grid gap-4 md:grid-cols-2">
          {Object.entries(topTerms).map(([cls, terms]) => (
            <div key={cls}>
              <h3 className="mb-2 text-sm font-semibold capitalize">{cls}</h3>
              <div className="flex flex-wrap gap-1.5">
                {(terms ?? []).map((t) => (
                  <span
                    key={t.term}
                    className="rounded-md border border-line bg-panel2 px-2 py-1 text-xs"
                    title={`${t.count} occurrences`}
                  >
                    {t.term}
                    <span className="ml-1 font-mono text-muted">{t.count}</span>
                  </span>
                ))}
                {(terms ?? []).length === 0 && <span className="text-xs text-muted">—</span>}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">
          Caveat: raw frequency favours common language shared by both classes; the
          models rely on TF-IDF weighting, which discounts exactly these generic terms.
        </p>
      </Card>

      <Card title="Data quality notes">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted">
          <li>Strong class imbalance — accuracy alone is misleading; we report macro-F1 and per-class metrics.</li>
          <li>
            {s.n_duplicates_removed} exact duplicate messages were inspected and removed
            before splitting, preventing train/test leakage.
          </li>
          <li>Empty / missing message bodies were dropped during validation (logged, not silent).</li>
          <li>URLs, e-mail addresses and long digit runs are normalised to placeholders inside the pipeline.</li>
          <li>Messages may contain phone numbers from the public dataset; raw examples are not reproduced in documentation.</li>
        </ul>
      </Card>
    </div>
  )
}
