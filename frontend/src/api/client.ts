/** Typed API client - the single place the frontend talks to the backend. */
import type {
  BatchResponse,
  ClassifyResponse,
  ConfusionMatrixResponse,
  HistoryListResponse,
  ModelComparisonResponse,
  ModelDetail,
  ModelSummary,
  OverviewResponse,
  ReadyResponse,
} from '../types/api'

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/+$/, '') + '/api/v1'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError('Cannot reach the backend. Is the API running on port 8000?', 0)
  }

  if (!resp.ok) {
    let detail = `Request failed (${resp.status})`
    try {
      const body = await resp.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail) && body.detail[0]?.msg)
        detail = body.detail.map((e: { msg: string }) => e.msg).join('; ')
    } catch {
      /* keep default detail */
    }
    throw new ApiError(detail, resp.status)
  }
  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

export const api = {
  ready: () => request<ReadyResponse>('/ready'),

  classify: (text: string, modelId?: string | null) =>
    request<ClassifyResponse>('/classify', {
      method: 'POST',
      body: JSON.stringify({ text, model_id: modelId ?? null }),
    }),

  classifyBatch: (texts: string[], modelId?: string | null) =>
    request<BatchResponse>('/classify/batch', {
      method: 'POST',
      body: JSON.stringify({ texts, model_id: modelId ?? null }),
    }),

  listModels: () => request<ModelSummary[]>('/models'),
  modelDetail: (id: string) => request<ModelDetail>(`/models/${id}`),
  selectedModel: () => request<ModelSummary>('/models/selected'),

  overview: () => request<OverviewResponse>('/analytics/overview'),
  modelComparison: () => request<ModelComparisonResponse>('/analytics/model-comparison'),
  confusionMatrix: (modelId?: string | null) =>
    request<ConfusionMatrixResponse>(
      `/analytics/confusion-matrix${modelId ? `?model_id=${encodeURIComponent(modelId)}` : ''}`,
    ),
  datasetStats: () => request<OverviewResponse['dataset_stats']>('/analytics/dataset-stats'),

  history: (limit = 20) => request<HistoryListResponse>(`/history?limit=${limit}`),
  deleteHistory: async (id: string) => {
    await request<void>(`/history/${encodeURIComponent(id)}`, { method: 'DELETE' })
  },
}
