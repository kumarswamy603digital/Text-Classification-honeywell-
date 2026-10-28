/** Model Comparison: real metrics for all four classifiers + confusion matrices. */
import * as React from 'react'
import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Card, Badge, Button } from '../components/ui/primitives'
import { ErrorState, LoadingState } from '../components/ui/states'
import { ModelComparisonTable } from '../components/analytics/ModelComparisonTable'
import { ConfusionMatrixChart } from '../components/analytics/ConfusionMatrixChart'
import { num, pct } from '../utils/format'

export default function Comparison() {
  const comparison = useApi(() => api.modelComparison())
  const [selected, setSelected] = React.useState<string | null>(null)
  const confusion = useApi(
    () => api.confusionMatrix(selected ?? undefined),
    !comparison.loading && !comparison.error,
  )

  if (comparison.loading) {
    return (
      <Card title="Model Comparison">
        <LoadingState label="Loading comparison results" />
      </Card>
    )
  }
  if (comparison.error || !comparison.data) {
    return (
      <ErrorState
        title="No evaluation results available"
        message={comparison.error ?? 'Train models first, then refresh.'}
        onRetry={comparison.reload}
      />
    )
  }

  const { rows, protocol } = comparison.data
  const bestByTest = [...rows].sort((a, b) => (b.macro_f1 ?? 0) - (a.macro_f1 ?? 0))[0]
  const selectedRow = rows.find((r) => r.selected)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Model Comparison</h1>
        <p className="mt-1 text-sm text-muted">
          Every number below comes from the actual held-out test split — identical
          protocol for all four classifiers.
        </p>
      </div>

      <Card title="Evaluation protocol">
        <ul className="grid gap-2 text-sm sm:grid-cols-2">
          <li className="rounded-lg border border-line bg-panel2/40 p-3">
            <p className="text-xs uppercase tracking-wider text-muted">Split</p>
            <p>{String(protocol.split?.strategy ?? '—')}</p>
            <p className="text-xs text-muted">
              train {num((protocol.split?.n_train as number | undefined) ?? null, 0)} · test{' '}
              {num((protocol.split?.n_test as number | undefined) ?? null, 0)} (n_evaluated ={' '}
              {protocol.n_evaluated ?? '—'})
            </p>
          </li>
          <li className="rounded-lg border border-line bg-panel2/40 p-3">
            <p className="text-xs uppercase tracking-wider text-muted">Selection criterion</p>
            <p>{protocol.selection_criterion ?? '—'}</p>
          </li>
          <li className="rounded-lg border border-line bg-panel2/40 p-3">
            <p className="text-xs uppercase tracking-wider text-muted">Final evaluation</p>
            <p>{protocol.final_evaluation ?? '—'}</p>
          </li>
          <li className="rounded-lg border border-line bg-panel2/40 p-3">
            <p className="text-xs uppercase tracking-wider text-muted">Positive class</p>
            <p>
              {protocol.positive_class ?? 'spam'} — misses (false negatives) cost user
              exposure to spam; false positives cost trust.
            </p>
          </li>
        </ul>
      </Card>

      <Card title="Metrics — all trained models">
        <ModelComparisonTable rows={rows} />
        {selectedRow && bestByTest && (
          <p className="mt-3 text-xs leading-relaxed text-muted">
            <strong className="text-ink">Why this model was selected:</strong>{' '}
            {selectedRow.name} had the best validation macro-F1
            ({pct(selectedRow.validation_macro_f1)}) under 3-fold stratified CV, so it
            became the default served model. It is not automatically “the best”:{' '}
            {bestByTest.model_id !== selectedRow.model_id
              ? `${bestByTest.name} edges it on test macro-F1 (${pct(bestByTest.macro_f1)} vs ${pct(selectedRow.macro_f1)}) — selection uses validation data so the test split stays an honest estimate.`
              : 'test results agree with the validation ranking here, but the decision was made before touching the test split.'}{' '}
            Accuracy alone would be misleading on this imbalanced dataset — see the
            Dataset Explorer for the class distribution.
          </p>
        )}
      </Card>

      <Card
        title="Confusion matrices"
        subtitle="Choose a model to inspect its test-split mistakes"
      >
        <div className="mb-4 flex flex-wrap gap-2" role="tablist" aria-label="Model for confusion matrix">
          {rows.map((r) => (
            <Button
              key={r.model_id}
              variant={r.model_id === (confusion.data?.model_id ?? selectedRow?.model_id) ? 'primary' : 'secondary'}
              onClick={() => setSelected(r.model_id)}
              className="px-3 py-1.5 text-xs"
              aria-pressed={r.model_id === (confusion.data?.model_id ?? selectedRow?.model_id)}
            >
              {r.name}
              {r.selected && (
                <Badge tone="info" className="ml-2">
                  ★
                </Badge>
              )}
            </Button>
          ))}
        </div>
        {confusion.loading && <LoadingState />}
        {confusion.data && (
          <div className="grid items-start gap-6 md:grid-cols-2">
            <ConfusionMatrixChart labels={confusion.data.labels} matrix={confusion.data.matrix} />
            <div className="space-y-2 text-sm">
              <h3 className="font-semibold text-muted">
                Error breakdown · {confusion.data.model_id} · {confusion.data.n_evaluated} rows
              </h3>
              {confusion.data.error_breakdown && (
                <ul className="space-y-1">
                  <li className="flex justify-between border-b border-line/50 py-1">
                    <span>Correct ham (TN)</span>
                    <span className="font-mono">{confusion.data.error_breakdown.true_negatives}</span>
                  </li>
                  <li className="flex justify-between border-b border-line/50 py-1">
                    <span>Ham wrongly flagged spam (FP)</span>
                    <span className="font-mono text-bad">
                      {confusion.data.error_breakdown.false_positives_spam_flag_on_ham}
                    </span>
                  </li>
                  <li className="flex justify-between border-b border-line/50 py-1">
                    <span>Spam missed (FN)</span>
                    <span className="font-mono text-warn">
                      {confusion.data.error_breakdown.false_negives_spam_missed}
                    </span>
                  </li>
                  <li className="flex justify-between py-1">
                    <span>Spam caught (TP)</span>
                    <span className="font-mono text-good">{confusion.data.error_breakdown.true_positives}</span>
                  </li>
                </ul>
              )}
              <p className="text-xs text-muted">
                A false positive means a real message got flagged as spam — usually the
                costlier mistake for a personal inbox.
              </p>
            </div>
          </div>
        )}
        {confusion.error && <p className="text-sm text-bad">{confusion.error}</p>}
      </Card>
    </div>
  )
}
