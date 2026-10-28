/** Renders a single classification result: label, probabilities, caveats. */
import { CheckCircle2, XCircle } from 'lucide-react'
import type { ClassifyResponse } from '../../types/api'
import { pct } from '../../utils/format'
import { Badge } from '../ui/primitives'
import { ClassProbabilityChart } from '../analytics/ClassProbabilityChart'
import { ExplanationPanel } from './ExplanationPanel'

export function PredictionResult({ result }: { result: ClassifyResponse }) {
  const isSpam = result.predicted_label === 'spam'
  const lowConfidence =
    result.confidence !== null && result.confidence < 0.6 && !isSpam

  return (
    <div className="space-y-4" aria-live="polite">
      <div className="flex flex-wrap items-center gap-3">
        <span
          className={`grid h-11 w-11 place-items-center rounded-full ${
            isSpam ? 'bg-bad/15 text-bad' : 'bg-good/15 text-good'
          }`}
          aria-hidden
        >
          {isSpam ? <XCircle className="h-6 w-6" /> : <CheckCircle2 className="h-6 w-6" />}
        </span>
        <div>
          <p className="text-lg font-bold leading-tight">{result.label_display}</p>
          <p className="text-xs text-muted">
            {result.model_name} v{result.model_version} · {result.latency_ms} ms ·{' '}
            {new Date(result.timestamp).toLocaleTimeString()}
          </p>
        </div>
        <Badge tone={isSpam ? 'danger' : 'success'} className="ml-auto">
          {result.predicted_label}
        </Badge>
      </div>

      {result.probabilities ? (
        <div className="rounded-lg border border-line bg-panel2/40 p-3">
          <h3 className="mb-1 text-sm font-semibold">Class probabilities</h3>
          <ClassProbabilityChart probabilities={result.probabilities} />
          <p className="mt-2 text-xs text-muted">
            Reported confidence is the model&rsquo;s estimated probability for the
            predicted class ({pct(result.confidence)}) — a probability, not a guarantee
            of correctness.
            {lowConfidence &&
              ' Confidence is low: consider this a "needs review" prediction rather than a trustworthy one.'}
          </p>
        </div>
      ) : (
        <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-warn">
          {result.confidence_note ??
            'This model does not provide calibrated probabilities; only the discrete label is shown.'}
        </p>
      )}

      <div className="rounded-lg border border-line bg-panel2/40 p-3">
        <ExplanationPanel
          terms={result.explanation_terms}
          predictedLabel={result.predicted_label}
          modelName={result.model_name}
        />
      </div>
    </div>
  )
}
