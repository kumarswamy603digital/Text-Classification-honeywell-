/** Minimal RFC-4180 CSV parser (quotes, escaped quotes, CRLF) - no deps. */

export interface ParsedCsv {
  headers: string[]
  rows: string[][]
}

export function parseCsv(text: string): ParsedCsv {
  const content = text.replace(/^\uFEFF/, '') // strip BOM
  if (!content.trim()) {
    throw new Error('The file is empty.')
  }
  const records: string[][] = []
  let field = ''
  let record: string[] = []
  let inQuotes = false
  let i = 0

  const pushField = () => {
    record.push(field)
    field = ''
  }
  const pushRecord = () => {
    pushField()
    // skip fully-empty records produced by trailing newlines
    if (!(record.length === 1 && record[0] === '')) records.push(record)
    record = []
  }

  while (i < content.length) {
    const ch = content[i]
    if (inQuotes) {
      if (ch === '"') {
        if (content[i + 1] === '"') {
          field += '"'
          i += 2
          continue
        }
        inQuotes = false
        i++
        continue
      }
      field += ch
      i++
      continue
    }
    if (ch === '"' && field === '') {
      inQuotes = true
      i++
      continue
    }
    if (ch === ',') {
      pushField()
      i++
      continue
    }
    if (ch === '\r') {
      if (content[i + 1] === '\n') i++
      pushRecord()
      i++
      continue
    }
    if (ch === '\n') {
      pushRecord()
      i++
      continue
    }
    field += ch
    i++
  }
  if (field !== '' || record.length > 0) pushRecord()

  if (records.length === 0) throw new Error('No records found.')
  const headers = records[0].map((h) => h.trim())
  const width = headers.length
  const rows = records.slice(1).filter((r) => r.length === width || r.some((c) => c.trim() !== ''))
  if (rows.length === 0) throw new Error('The file has a header but no data rows.')
  const ragged = records.slice(1).find((r) => r.length !== width)
  if (ragged) {
    throw new Error(
      `Row has ${ragged.length} columns but the header has ${width}. Check for stray delimiters.`,
    )
  }
  return { headers, rows }
}

export function detectTextColumn(parsed: ParsedCsv): number {
  const preferred = ['text', 'message', 'body', 'sms', 'content', 'comment']
  const lower = parsed.headers.map((h) => h.toLowerCase())
  for (const name of preferred) {
    const idx = lower.indexOf(name)
    if (idx >= 0) return idx
  }
  // fallback: column with the longest average content
  let best = 0
  let bestLen = -1
  parsed.headers.forEach((_, col) => {
    const avg =
      parsed.rows.reduce((acc, r) => acc + (r[col]?.length ?? 0), 0) / Math.max(1, parsed.rows.length)
    if (avg > bestLen) {
      bestLen = avg
      best = col
    }
  })
  return best
}

export function toCsvCell(value: string | number | null | undefined): string {
  const s = value === null || value === undefined ? '' : String(value)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

export function buildCsv(headers: string[], rows: (string | number | null)[][]): string {
  return [
    headers.map(toCsvCell).join(','),
    ...rows.map((r) => r.map(toCsvCell).join(',')),
  ].join('\r\n')
}
