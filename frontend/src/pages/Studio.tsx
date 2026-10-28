/** Text Classification Studio: single-text prediction against the live API. */
import * as React from 'react'
import { Eraser, Sparkles } from 'lucide-react'
import { api, ApiError } from '../api/client'
import { useApi } from '../hooks/useApi'
import type { ClassifyResponse } from '../types/api'
import { Button, Card } from '../components/ui/primitives'
import { ErrorState, LoadingState } from '../components/ui/states'
import { TextInputPanel, MAX_TEXT_LENGTH } from '../components/classification/TextInputPanel'
import { ModelSelector } from '../components/classification/ModelSelector'
import { PredictionResult } from '../components/classification/PredictionResult'

const EXAMPLES = [
  {
    label: 'likely spam',
    text: 'URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Call 09061234567 now to claim.',
  },
  {
    label: 'likely ham',
    text: "Hey, are we still on for coffee tomorrow at 10? I'll bring the report draft we talked about.",
  },
  {
    label: 'tricky',
    text: 'Congratulations dear! Win a guaranteed £2000 reward. Reply WIN to 62223 to collect now!',
  },
]

export default function Studio() {
  const [text, setText] = React.useState('')
  const [modelId, setModelId] = React.useState<string>('')
  const [result, setResult] = React.useState<ClassifyResponse | null>(null)
  const [submitting, setSubmitting] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  const models = useApi(() => api.listModels())

  // default to the selected model once loaded
  React.useEffect(() => {
    if (models.data && !modelId) {
      const sel = models.data.find((m) => m.is_selected) ?? models.data[0]
      if (sel) setModelId(sel.model_id)
    }
  }, [models.data, modelId])

  const tooLong = text.length > MAX_TEXT_LENGTH
  const empty = text.trim().length === 0

  async function classify() {
    if (empty || tooLong) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await api.classify(text, modelId || null)
      setResult(res)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Prediction failed unexpectedly.')
      setResult(null)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Classification Studio</h1>
        <p className="mt-1 text-sm text-muted">
          Type or paste a message. The request goes to the FastAPI backend and is
          classified by the actual fitted pipeline you choose — nothing is simulated.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          {models.loading ? (
            <LoadingState label="Loading model registry" />
          ) : models.error ? (
            <ErrorState
              compact
              title="Model registry unavailable"
              message={models.error}
              onRetry={models.reload}
            />
          ) : (
            <div className="space-y-5">
              <ModelSelector
                models={models.data ?? []}
                value={modelId}
                onChange={setModelId}
              />
              <TextInputPanel
                value={text}
                onChange={setText}
                onSubmit={classify}
                submitting={submitting}
                examples={EXAMPLES}
              />
              <div className="flex flex-wrap gap-3">
                <Button onClick={classify} disabled={empty || tooLong} loading={submitting}>
                  {!submitting && <Sparkles className="h-4 w-4" aria-hidden />}
                  Classify message
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => {
                    setText('')
                    setResult(null)
                    setError(null)
                  }}
                >
                  <Eraser className="h-4 w-4" aria-hidden />
                  Clear
                </Button>
              </div>
              {error && (
                <p role="alert" className="rounded-lg border border-bad/40 bg-bad/10 p-3 text-sm text-bad">
                  {error}
                </p>
              )}
            </div>
          )}
        </Card>

        <Card
          className="lg:col-span-2"
          title="Prediction"
          subtitle={result ? undefined : 'Results appear here after classification'}
        >
          {submitting && <LoadingState label="Classifying…" />}
          {!submitting && result && <PredictionResult result={result} />}
          {!submitting && !result && !error && (
            <div className="py-8 text-center text-sm text-muted">
              <p>No prediction yet.</p>
              <p className="mt-1 text-xs">
                Try one of the example chips, then press “Classify message”.
              </p>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
