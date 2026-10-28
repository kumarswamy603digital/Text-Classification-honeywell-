/** Batch Classification: CSV upload -> chunked /classify/batch -> downloadable results. */
import * as React from 'react'
import { Download, Play } from 'lucide-react'
import { api, ApiError } from '../api/client'
import { useApi } from '../hooks/useApi'
import type { ClassifyResponse } from '../types/api'
import { Badge, Button, Card } from '../components/ui/primitives'
import { LoadingState } from '../components/ui/states'
import { FileUploadPanel, MAX_FILE_BYTES } from '../components/classification/FileUploadPanel'
import { ModelSelector } from '../components/classification/ModelSelector'
import { buildCsv, detectTextColumn, parseCsv, type ParsedCsv } from '../utils/csv'
import { downloadCsv, pct, truncate } from '../utils/format'

const CHUNK = 50 // rows per API call (backend limit is 200)
type RowResult = { index: number; text: string; result: ClassifyResponse | null; error: string | null }

export default function Batch() {
  const [fileName, setFileName] = React.useState<string | null>(null)
  const [parsed, setParsed] = React.useState<ParsedCsv | null>(null)
  const [parseError, setParseError] = React.useState<string | null>(null)
  const [textCol, setTextCol] = React.useState(0)
  const [modelId, setModelId] = React.useState('')
  const [rows, setRows] = React.useState<RowResult[] | null>(null)
  const [progress, setProgress] = React.useState<{ done: number; total: number } | null>(null)
  const [runError, setRunError] = React.useState<string | null>(null)

  const models = useApi(() => api.listModels())
  React.useEffect(() => {
    if (models.data && !modelId) {
      const sel = models.data.find((m) => m.is_selected) ?? models.data[0]
      if (sel) setModelId(sel.model_id)
    }
  }, [models.data, modelId])

  async function handleFile(file: File) {
    setParseError(null)
    setRows(null)
    setRunError(null)
    setProgress(null)
    if (file.size > MAX_FILE_BYTES) {
      setParsed(null)
      setFileName(file.name)
      setParseError(`File is ${(file.size / 1024 / 1024).toFixed(1)} MB — the limit is 2 MB.`)
      return
    }
    let textContent: string
    try {
      textContent = await file.text()
    } catch {
      setParseError('Could not read the file (invalid encoding?). Please upload UTF-8 CSV.')
      return
    }
    try {
      const p = parseCsv(textContent)
      setParsed(p)
      setFileName(file.name)
      setTextCol(detectTextColumn(p))
    } catch (e) {
      setParsed(null)
      setFileName(file.name)
      setParseError(e instanceof Error ? e.message : 'Unparseable CSV.')
    }
  }

  async function runBatch() {
    if (!parsed) return
    const texts = parsed.rows.map((r) => r[textCol] ?? '')
    const total = texts.length
    setRows(null)
    setRunError(null)
    setProgress({ done: 0, total })
    const acc: RowResult[] = []
    try {
      for (let start = 0; start < total; start += CHUNK) {
        const slice = texts.slice(start, start + CHUNK)
        const res = await api.classifyBatch(
          slice.map((t) => t.trim() || ' '), // backend rejects empty strings; blanks reported separately
          modelId || null,
        )
        slice.forEach((t, i) =>
          acc.push({
            index: start + i,
            text: t,
            result: res.results[i] ?? null,
            error: t.trim() === '' ? 'empty cell - skipped' : null,
          }),
        )
        setProgress({ done: Math.min(start + CHUNK, total), total })
        setRows([...acc])
      }
    } catch (e) {
      setRunError(e instanceof ApiError ? e.message : 'Batch run failed.')
      setRows(acc.length ? acc : null)
    }
  }

  function exportResults() {
    if (!rows) return
    const csv = buildCsv(
      ['row', 'input_text', 'predicted_label', 'confidence', 'model_id', 'note'],
      rows.map((r) => [
        r.index + 2, // +2: 1-based + header row in the original file
        r.text,
        r.result?.predicted_label ?? '',
        r.result?.confidence ?? '',
        r.result?.model_id ?? '',
        r.error ?? '',
      ]),
    )
    downloadCsv('classification_results.csv', csv)
  }

  const spamCount = rows?.filter((r) => r.result?.predicted_label === 'spam').length ?? 0

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Batch Classification</h1>
        <p className="mt-1 text-sm text-muted">
          Upload a CSV of messages; every row is classified by the real model, in the
          original order, and results are downloadable.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="1 · Upload CSV">
          <FileUploadPanel onFile={handleFile} fileName={fileName} error={parseError} />
          {parsed && (
            <div className="mt-4 space-y-2 text-sm">
              <p className="text-muted">
                Parsed <strong className="text-ink">{parsed.rows.length}</strong> data rows ·{' '}
                {parsed.headers.length} column(s)
              </p>
              <label htmlFor="col-select" className="block text-xs uppercase tracking-wider text-muted">
                Text column
              </label>
              <select
                id="col-select"
                value={textCol}
                onChange={(e) => setTextCol(Number(e.target.value))}
                className="w-full rounded-lg border border-line bg-panel2 px-3 py-2 text-sm"
              >
                {parsed.headers.map((h, i) => (
                  <option key={`${h}-${i}`} value={i}>
                    {h || `column ${i + 1}`}
                  </option>
                ))}
              </select>
            </div>
          )}
        </Card>

        <Card title="2 · Run">
          {models.loading ? (
            <LoadingState />
          ) : (
            <div className="space-y-4">
              <ModelSelector models={models.data ?? []} value={modelId} onChange={setModelId} />
              <Button onClick={runBatch} disabled={!parsed} loading={!!progress}>
                {!progress && <Play className="h-4 w-4" aria-hidden />}
                Classify {parsed ? `${parsed.rows.length} rows` : 'uploaded rows'}
              </Button>
              {progress && (
                <div aria-live="polite">
                  <div className="mb-1 flex justify-between text-xs text-muted">
                    <span>
                      {progress.done} / {progress.total} rows
                    </span>
                    <span>{pct(progress.done / progress.total, 0)}</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-panel2">
                    <div
                      className="h-full rounded-full bg-accent transition-all"
                      style={{ width: `${(progress.done / progress.total) * 100}%` }}
                    />
                  </div>
                </div>
              )}
              {runError && (
                <p role="alert" className="rounded-lg border border-bad/40 bg-bad/10 p-3 text-sm text-bad">
                  {runError}
                </p>
              )}
            </div>
          )}
        </Card>
      </div>

      {rows && rows.length > 0 && (
        <Card
          title={`3 · Results (${rows.length} rows)`}
          subtitle={`${spamCount} classified as spam · ${rows.length - spamCount} ham`}
          actions={
            <Button variant="secondary" onClick={exportResults}>
              <Download className="h-4 w-4" aria-hidden />
              Download CSV
            </Button>
          }
        >
          <div className="max-h-96 overflow-auto rounded-lg border border-line">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-panel2 text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th scope="col" className="px-3 py-2">
                    #
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Input (original order)
                  </th>
                  <th scope="col" className="px-3 py-2">
                    Label
                  </th>
                  <th scope="col" className="px-3 py-2 text-right">
                    Confidence
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.index} className="border-t border-line/60">
                    <td className="px-3 py-2 font-mono text-xs text-muted">{r.index + 1}</td>
                    <td className="max-w-md px-3 py-2" title={r.text}>
                      {truncate(r.text, 70)}
                    </td>
                    <td className="px-3 py-2">
                      {r.error ? (
                        <Badge tone="warn">{r.error}</Badge>
                      ) : r.result ? (
                        <Badge tone={r.result.predicted_label === 'spam' ? 'danger' : 'success'}>
                          {r.result.predicted_label}
                        </Badge>
                      ) : (
                        <Badge tone="neutral">—</Badge>
                      )}
                    </td>
                    <td className="px-3 py-2 text-right font-mono tabular-nums">
                      {r.result?.confidence != null ? pct(r.result.confidence) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
