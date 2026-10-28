/** Model dropdown driven by the registry API. */
import type { ModelSummary } from '../../types/api'
import { Select } from '../ui/primitives'

export function ModelSelector({
  models,
  value,
  onChange,
  loading,
}: {
  models: ModelSummary[]
  value: string
  onChange: (id: string) => void
  loading?: boolean
}) {
  return (
    <Select
      label="Model"
      disabled={loading || models.length === 0}
      value={value}
      onChange={onChange}
      options={models.map((m) => ({
        value: m.model_id,
        label: `${m.name}${m.is_selected ? '  ★ selected' : ''}`,
        description: m.supports_predict_proba
          ? 'reports class probabilities'
          : 'label only — no calibrated probabilities',
      }))}
    />
  )
}
