/** Feature-weight explanation for a prediction (with honest caveats). */
import type { ExplanationTerm } from '../../types/api'
import { cn } from '../../utils/format'

export function ExplanationPanel({
  terms,
  predictedLabel,
  modelName,
}: {
  terms: ExplanationTerm[]
  predictedLabel: string
  modelName: string
}) {
  if (!terms || terms.length === 0) {
    return (
      <p className="text-sm text-muted">
        No explanation available for this input — either no vocabulary terms matched
        the fitted TF-IDF features, or this model type does not expose usable feature
        weights.
      </p>
    )
  }
  const maxAbs = Math.max(...terms.map((t) => Math.abs(t.weight)), 0.0001)

  return (
    <div>
      <h4 className="text-sm font-semibold">Influential vocabulary terms</h4>
      <p className="mt-0.5 text-xs text-muted">
        Weighted TF-IDF features of <em>this message</em> that push {modelName} toward{' '}
        <strong>{predictedLabel}</strong> (green) or away from it (red). These are
        feature attributions for a linear model — informative, but not a proof of why
        the model decided as it did.
      </p>
      <ul className="mt-3 space-y-1.5">
        {terms.map(({ term, weight }) => (
          <li key={term} className="flex items-center gap-2 text-xs">
            <span className="w-32 truncate font-mono" title={term}>
              {term}
            </span>
            <span className="relative h-3 flex-1 overflow-hidden rounded bg-panel2">
              <span className="absolute left-1/2 top-0 h-full w-px bg-line" aria-hidden />
              <span
                aria-hidden
                className={cn(
                  'absolute top-0 h-full rounded',
                  weight >= 0 ? 'bg-good/70' : 'bg-bad/70',
                )}
                style={
                  weight >= 0
                    ? { left: '50%', width: `${(Math.abs(weight) / maxAbs) * 50}%` }
                    : { right: '50%', width: `${(Math.abs(weight) / maxAbs) * 50}%` }
                }
              />
            </span>
            <span className="w-14 text-right font-mono tabular-nums text-muted">
              {weight >= 0 ? '+' : ''}
              {weight.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
