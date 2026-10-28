/** Dashboard: project summary, dataset stats, selected model, real metrics, history. */
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card, Badge, Button } from '../components/ui/primitives'
import { ErrorState, LoadingState } from '../components/ui/states'
import { MetricCard } from '../components/analytics/MetricCard'
import { DatasetStatistics } from '../components/analytics/DatasetStatistics'
import { ConfusionMatrixChart } from '../components/analytics/ConfusionMatrixChart'
import { fmtDateTime, num, pct } from '../utils/format'

export default function Dashboard() {
  const overview = useApi(() => api.overview())
  const confusion = useApi(
    () => api.confusionMatrix(),
    !overview.loading && !overview.error,
  )
  const history = useApi(() => api.history(8))

  if (overview.loading) {
    return (
      <Card title="Dashboard">
        <LoadingState label="Loading analytics overview" />
      </Card>
    )
  }

  if (overview.error || !overview.data) {
    return (
      <ErrorState
        title="No trained models available yet"
        message={
          overview.errorCode === 503
            ? 'The backend is running but no model artifacts were found. Train models with: python scripts/train_models.py (from the backend folder), then refresh this page.'
            : (overview.error ?? 'Overview unavailable.')
        }
        onRetry={overview.reload}
      />
    )
  }

  const o = overview.data
  const m = o.selected_metrics
  const stats = o.dataset_stats

  return (
    <div className="space-y-6">
      <div>
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          {o.selected_model && (
            <Badge tone="info">
              selected: {o.selected_model.name} · {o.selected_model.model_id}
            </Badge>
          )}
        </div>
        <p className="mt-1 text-sm text-muted">
          SMS spam detection · {o.n_models_trained} traditional ML models trained ·
          predictions served by fitted scikit-learn pipelines.
          {o.trained_at && <span className="ml-1">Last training: {fmtDateTime(o.trained_at)}</span>}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard
          label="Test accuracy"
          value={pct(m?.accuracy)}
          sub={`held-out split, n=${m?.n_evaluated ?? '—'}`}
        />
        <MetricCard
          label="Spam F1"
          value={pct(m?.f1_spam)}
          tone="accent"
          sub="positive class = spam"
        />
        <MetricCard
          label="Macro-F1"
          value={pct(m?.macro_f1)}
          tone="good"
          sub="mean of per-class F1"
        />
        <MetricCard
          label="ROC-AUC (spam)"
          value={m?.roc_auc_spam != null ? num(m.roc_auc_spam, 4) : '—'}
          sub={m?.roc_auc_spam != null ? 'probability-capable model' : 'not available for selected model'}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Model evaluation (real test-split results)"
          subtitle={o.selection_criterion ?? undefined}
        >
          {m ? (
            <div className="space-y-4">
              <ul className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm">
                {(
                  [
                    ['Precision (spam)', pct(m.precision_spam)],
                    ['Recall (spam)', pct(m.recall_spam)],
                    ['Weighted-F1', pct(m.weighted_f1)],
                    ['Evaluated rows', String(m.n_evaluated)],
                  ] as const
                ).map(([k, v]) => (
                  <li key={k} className="flex justify-between border-b border-line/50 py-1">
                    <span className="text-muted">{k}</span>
                    <span className="font-mono tabular-nums">{v}</span>
                  </li>
                ))}
              </ul>
              {confusion.data && (
                <ConfusionMatrixChart
                  labels={confusion.data.labels}
                  matrix={confusion.data.matrix}
                />
              )}
            </div>
          ) : (
            <p className="text-sm text-muted">Metrics unavailable.</p>
          )}
        </Card>

        <Card title="Dataset snapshot" subtitle={stats?.source.name}>
          {stats ? (
            <DatasetStatistics stats={stats} />
          ) : (
            <p className="text-sm text-muted">
              Run scripts/explore_data.py to generate dataset statistics.
            </p>
          )}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Recent predictions"
          subtitle="Stored locally in SQLite; input text is not retained by default"
          actions={
            <Link to="/studio">
              <Button variant="secondary">Open studio</Button>
            </Link>
          }
        >
          {history.loading && <LoadingState />}
          {!history.loading && history.data && history.data.records.length === 0 && (
            <p className="text-sm text-muted">
              No predictions yet — classify something in the Studio.
            </p>
          )}
          {history.data && history.data.records.length > 0 && (
            <ul className="divide-y divide-line/60 text-sm">
              {history.data.records.map((r) => (
                <li key={r.prediction_id} className="flex items-center justify-between py-2">
                  <div>
                    <Badge tone={r.predicted_label === 'spam' ? 'danger' : 'success'}>
                      {r.predicted_label}
                    </Badge>
                    <span className="ml-2 font-mono text-xs text-muted">{r.model_id}</span>
                  </div>
                  <span className="text-xs text-muted">
                    {r.confidence != null ? `${pct(r.confidence)} · ` : 'no prob · '}
                    {fmtDateTime(r.created_at)}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {history.error && <p className="text-sm text-bad">{history.error}</p>}
        </Card>

        <Card title="Pipeline at a glance">
          <ol className="space-y-2 text-sm">
            {[
              ['1. Acquire & validate', 'UCI SMS Spam Collection, schema-checked, deduplicated'],
              ['2. Preprocess', 'Unicode NFKC, whitespace, URL/number placeholders (inside pipeline)'],
              ['3. Featurize', 'TF-IDF word unigrams+bigrams, fitted on train data only'],
              ['4. Train & tune', 'GridSearchCV, 3-fold stratified CV, macro-F1'],
              ['5. Select & evaluate', 'Best CV model; single final pass on untouched test split'],
              ['6. Serve', 'FastAPI + registry → React dashboard & studio'],
            ].map(([t, d]) => (
              <li key={t} className="rounded-lg border border-line bg-panel2/40 p-3">
                <p className="font-semibold">{t}</p>
                <p className="text-xs text-muted">{d}</p>
              </li>
            ))}
          </ol>
        </Card>
      </div>
    </div>
  )
}
