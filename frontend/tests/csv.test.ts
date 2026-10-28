import { describe, expect, it } from 'vitest'
import { buildCsv, detectTextColumn, parseCsv, toCsvCell } from '../src/utils/csv'
import { num, pct, truncate } from '../src/utils/format'

describe('parseCsv', () => {
  it('parses simple header + rows', () => {
    const p = parseCsv('text,label\nhello world,ham\nfree prize,spam')
    expect(p.headers).toEqual(['text', 'label'])
    expect(p.rows).toEqual([
      ['hello world', 'ham'],
      ['free prize', 'spam'],
    ])
  })

  it('handles quoted fields with commas and escaped quotes', () => {
    const p = parseCsv('text\n"Hello, ""world""!"\nplain')
    expect(p.rows.map((r) => r[0])).toEqual(['Hello, "world"!', 'plain'])
  })

  it('handles CRLF and BOM', () => {
    const p = parseCsv('\uFEFFtext,n\r\na,b\r\nc,d\r\n')
    expect(p.headers).toEqual(['text', 'n'])
    expect(p.rows).toEqual([
      ['a', 'b'],
      ['c', 'd'],
    ])
  })

  it('rejects empty files', () => {
    expect(() => parseCsv('   ')).toThrow(/empty/i)
  })

  it('rejects header-only files', () => {
    expect(() => parseCsv('text\n')).toThrow(/no data rows/i)
  })

  it('rejects ragged rows (missing column / stray delimiters)', () => {
    expect(() => parseCsv('a,b\n1,2\n3,4,5')).toThrow(/columns/i)
  })

  it('skips stray blank lines but keeps empty cells inside a row', () => {
    // Single-column: a blank line is a stray newline, not an empty record.
    const single = parseCsv('text\nfirst\n\nthird')
    expect(single.rows.map((r) => r[0])).toEqual(['first', 'third'])
    // Multi-column: an empty cell within a real row must be preserved.
    const multi = parseCsv('a,b\n1,\n,3')
    expect(multi.rows).toEqual([
      ['1', ''],
      ['', '3'],
    ])
  })
})

describe('detectTextColumn', () => {
  it('prefers a column literally named text/message', () => {
    const p = parseCsv('id,Message,length\n1,hi,2')
    expect(detectTextColumn(p)).toBe(1)
  })

  it('falls back to the longest-content column', () => {
    const p = parseCsv('a,b\n1,a very much longer piece of text here')
    expect(detectTextColumn(p)).toBe(1)
  })
})

describe('buildCsv', () => {
  it('quotes and escapes values', () => {
    expect(toCsvCell('has, comma')).toBe('"has, comma"')
    expect(toCsvCell('has "quotes"')).toBe('"has ""quotes"""')
    expect(toCsvCell(null)).toBe('')
  })

  it('round-trips through parseCsv', () => {
    const csv = buildCsv(
      ['text', 'label'],
      [
        ['Hello, "spam" world', 'spam'],
        ['line\nbreak inside', 'ham'],
      ],
    )
    const p = parseCsv(csv)
    expect(p.rows[0][0]).toBe('Hello, "spam" world')
  })
})

describe('format helpers', () => {
  it('pct renders nulls honestly', () => {
    expect(pct(null)).toBe('—')
    expect(pct(0.9826)).toBe('98.3%')
  })
  it('num handles undefined', () => {
    expect(num(undefined)).toBe('—')
    expect(num(0.9909), '0.991')
  })
  it('truncate shortens long text', () => {
    expect(truncate('abcdefghij', 5)).toBe('abcd…')
  })
})
