/** Comparison table of all trained models on the held-out test split. */
import type { MetricRow } from '../../types/api'
import { num, pct } from '../../utils/format'
import { Badge } from '../ui/primitives'

const cols: { key: keyof MetricRow; label: string; fmt: (r: MetricRow) => string }[] = [
  { key: 'accuracy', label: 'Accuracy', fmt: (r) => pct(r.accuracy) },
  { key: 'precision_spam', label: 'Precision (spam)', fmt: (r) => pct(r.precision_spam) },
  { key: 'recall_spam', label: 'Recall (spam)', fmt: (r) => pct(r.recall_spam) },
  { key: 'f1_spam', label: 'F1 (spam)', fmt: (r) => pct(r.f1_spam) },
  { key: 'macro_f1', label: 'Macro-F1', fmt: (r) => pct(r.macro_f1) },
  { key: 'weighted_f1', label: 'Weighted-F1', fmt: (r) => pct(r.weighted_f1) },
  { key: 'roc_auc_spam', label: 'ROC-AUC (spam)', fmt: (r) => num(r.roc_auc_spam, 4) },
  {
    key: 'validation_macro_f1',
    label: 'CV macro-F1 (selection)',
    fmt: (r) => pct(r.validation_macro_f1),
  },
]

export function ModelComparisonTable({ rows }: { rows: MetricRow[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[880px] text-sm">
        <caption className="sr-only">
          Model comparison on the held-out test split with validation scores used for
          selection.
        </caption>
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wider text-muted">
            <th scope="col" className="px-3 py-2">
              Model
            </th>
            {cols.map((c) => (
              <th key={c.key} scope="col" className="px-3 py-2 text-right font-medium">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.model_id}
              className={`border-b border-line/60 ${r.selected ? 'bg-accent/8' : ''}`}
            >
              <th scope="row" className="px-3 py-2.5 text-left font-medium">
                {r.name}
                {r.selected && (
                  <Badge tone="info" className="ml-2">
                    selected
                  </Badge>
                )}
                <span className="block font-mono text-[11px] font-normal text-muted">
                  {r.model_id}
                </span>
              </th>
              {cols.map((c) => (
                <td key={c.key} className="px-3 py-2.5 text-right font-mono tabular-nums">
                  {c.fmt(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
