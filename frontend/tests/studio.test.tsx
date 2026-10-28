import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ClassifyResponse, ModelSummary } from '../src/types/api'

// Mock the API client module; keep ApiError so useApi's error mapping still works.
vi.mock('../src/api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/api/client')>()
  return {
    ...actual,
    api: {
      listModels: vi.fn(),
      classify: vi.fn(),
    },
  }
})

import { api } from '../src/api/client'
import Studio from '../src/pages/Studio'

const MODELS: ModelSummary[] = [
  {
    model_id: 'linearsvc',
    name: 'Linear SVM',
    version: '1.0.0',
    supports_predict_proba: false,
    notes: '',
    is_selected: true,
  },
  {
    model_id: 'mnb',
    name: 'Multinomial NB',
    version: '1.0.0',
    supports_predict_proba: true,
    notes: '',
    is_selected: false,
  },
]

const RESULT: ClassifyResponse = {
  prediction_id: 'p-1',
  model_id: 'linearsvc',
  model_name: 'Linear SVM',
  model_version: '1.0.0',
  predicted_label: 'spam',
  label_display: 'Spam',
  probabilities: null,
  confidence: null,
  confidence_type: null,
  confidence_note: 'This model does not provide calibrated probabilities.',
  explanation_terms: [{ term: 'urgent', weight: 1.2 }],
  latency_ms: 4,
  status: 'success',
  timestamp: '2026-10-27T10:00:00Z',
}

function renderStudio() {
  return render(
    <MemoryRouter>
      <Studio />
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.mocked(api.listModels).mockReset().mockResolvedValue(MODELS)
  vi.mocked(api.classify).mockReset()
})

describe('Studio page', () => {
  it('loads the model registry and defaults to the selected model', async () => {
    renderStudio()
    expect(await screen.findByText('Classification Studio')).toBeInTheDocument()
    await waitFor(() => expect(api.listModels).toHaveBeenCalled())
    // The selected model name should be surfaced once the registry resolves.
    expect(await screen.findByText(/Linear SVM/i)).toBeInTheDocument()
  })

  it('shows the empty prompt before any prediction', async () => {
    renderStudio()
    expect(await screen.findByText(/No prediction yet/i)).toBeInTheDocument()
  })

  it('classifies text via the real API and renders the result', async () => {
    vi.mocked(api.classify).mockResolvedValue(RESULT)
    renderStudio()

    const box = await screen.findByLabelText('Message text')
    fireEvent.change(box, { target: { value: 'URGENT! You have won a prize' } })
    fireEvent.click(screen.getByRole('button', { name: /Classify message/i }))

    await waitFor(() => expect(api.classify).toHaveBeenCalledTimes(1))
    const [, sentModel] = vi.mocked(api.classify).mock.calls[0]
    expect(sentModel).toBe('linearsvc')

    expect(await screen.findByText('Spam')).toBeInTheDocument()
    // Non-probability model shows the honest caveat instead of a fake confidence.
    expect(
      await screen.findByText(/does not provide calibrated probabilities/i),
    ).toBeInTheDocument()
  })

  it('disables classify for empty input', async () => {
    renderStudio()
    const button = await screen.findByRole('button', { name: /Classify message/i })
    expect(button).toBeDisabled()
  })

  it('surfaces API errors without crashing', async () => {
    vi.mocked(api.classify).mockRejectedValue(new Error('boom'))
    renderStudio()
    const box = await screen.findByLabelText('Message text')
    fireEvent.change(box, { target: { value: 'some message' } })
    fireEvent.click(screen.getByRole('button', { name: /Classify message/i }))
    expect(await screen.findByRole('alert')).toBeInTheDocument()
  })
})
