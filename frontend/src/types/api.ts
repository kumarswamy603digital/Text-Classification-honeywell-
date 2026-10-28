/** Typed API contract mirroring backend/app/schemas/api_schemas.py */

export interface ExplanationTerm {
  term: string
  weight: number
}

export interface ClassifyResponse {
  prediction_id: string
  model_id: string
  model_name: string
  model_version: string
  predicted_label: string
  label_display: string
  probabilities: Record<string, number> | null
  confidence: number | null
  confidence_type: string | null
  confidence_note: string | null
  explanation_terms: ExplanationTerm[]
  latency_ms: number
  status: 'success'
  timestamp: string
}

export interface BatchResponse {
  results: ClassifyResponse[]
  count: number
  model_id: string
}

export interface ModelSummary {
  model_id: string
  name: string
  version: string
  supports_predict_proba: boolean
  notes: string
  is_selected: boolean
}

export interface PerClassMetric {
  precision: number
  recall: number
  f1: number
  support: number
}

export interface TestMetrics {
  n_evaluated: number
  accuracy: number
  precision_spam: number
  recall_spam: number
  f1_spam: number
  macro_f1: number
  weighted_f1: number
  roc_auc_spam?: number
  pr_auc_spam?: number
  confusion_matrix: number[][]
  confusion_labels: string[]
  per_class: Record<string, PerClassMetric>
  error_breakdown: {
    true_negatives: number
    false_positives_spam_flag_on_ham: number
    false_negives_spam_missed: number
    true_positives: number
  }
}

export interface ModelDetail extends ModelSummary {
  best_params: Record<string, unknown>
  validation_macro_f1: number | null
  test_metrics: TestMetrics | null
}

export interface MetricRow {
  model_id: string
  name: string
  accuracy: number | null
  precision_spam: number | null
  recall_spam: number | null
  f1_spam: number | null
  macro_f1: number | null
  weighted_f1: number | null
  roc_auc_spam: number | null
  validation_macro_f1: number | null
  best_params: Record<string, unknown>
  selected: boolean
}

export interface ModelComparisonResponse {
  protocol: {
    split?: Record<string, unknown>
    selection_criterion?: string
    final_evaluation?: string
    positive_class?: string
    n_evaluated?: number | null
  }
  rows: MetricRow[]
}

export interface ConfusionMatrixResponse {
  model_id: string
  labels: string[]
  matrix: number[][]
  n_evaluated: number | null
  positive_class: string
  error_breakdown?: Record<string, number>
}

export interface DatasetStats {
  task: string
  n_samples: number
  n_classes: number
  class_names: string[]
  class_counts: Record<string, number>
  class_balance: Record<string, number>
  imbalance_ratio: number
  n_duplicates_removed: number
  text_length_chars: { min: number; max: number; mean: number; median: number }
  text_length_words: { min: number; max: number; mean: number; median: number }
  per_class: Record<
    string,
    {
      count: number
      share: number
      mean_chars: number
      median_chars: number
      min_chars: number
      max_chars: number
    }
  >
  source: { name: string; url: string; license: string; notes: string }
  top_terms?: Record<string, { term: string; count: number }[]>
}

export interface OverviewResponse {
  task: string
  dataset: Record<string, unknown>
  dataset_stats: DatasetStats | null
  selected_model: ModelSummary | null
  selected_metrics: TestMetrics | null
  n_models_trained: number
  history_enabled: boolean
  trained_at?: string | null
  selection_criterion?: string | null
}

export interface HistoryRecord {
  prediction_id: string
  created_at: string
  model_id: string
  predicted_label: string
  confidence: number | null
}

export interface HistoryListResponse {
  records: HistoryRecord[]
  total: number
  enabled: boolean
}

export interface ReadyResponse {
  status: 'ready' | 'not_ready'
  models_loaded: boolean
  selected_model_id: string | null
  detail: string | null
}
