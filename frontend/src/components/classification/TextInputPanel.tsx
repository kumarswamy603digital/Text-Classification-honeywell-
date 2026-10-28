/** Large text input with character validation, clear + example actions. */
import * as React from 'react'

export const MAX_TEXT_LENGTH = 5000

export function TextInputPanel({
  value,
  onChange,
  onSubmit,
  submitting,
  examples,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  submitting: boolean
  examples?: { label: string; text: string }[]
}) {
  const tooLong = value.length > MAX_TEXT_LENGTH
  const empty = value.trim().length === 0
  const counterClass = tooLong
    ? 'text-bad'
    : value.length > MAX_TEXT_LENGTH * 0.9
      ? 'text-warn'
      : 'text-muted'

  return (
    <div className="space-y-2">
      <label htmlFor="classify-input" className="block text-sm font-medium">
        Message text
      </label>
      <textarea
        id="classify-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !empty && !tooLong) onSubmit()
        }}
        rows={7}
        aria-invalid={tooLong || undefined}
        aria-describedby="char-count"
        placeholder="Paste an SMS message here, e.g. 'URGENT! You have won a cash prize, call 09061234567 now…'"
        className="w-full resize-y rounded-lg border border-line bg-panel2/60 p-3 font-mono text-sm leading-relaxed placeholder:text-muted/60 focus:border-accent"
      />
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <p id="char-count" className={counterClass} aria-live="polite">
          {value.length.toLocaleString()} / {MAX_TEXT_LENGTH.toLocaleString()} characters
          {tooLong && ' — too long'}
          {empty && !tooLong && ' — enter some text to classify'}
        </p>
        <p className="text-muted">Ctrl+Enter to classify</p>
      </div>
      {examples && examples.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-1">
          <span className="text-xs text-muted">Try:</span>
          {examples.map((ex) => (
            <button
              key={ex.label}
              type="button"
              onClick={() => onChange(ex.text)}
              className="rounded-full border border-line bg-panel2 px-2.5 py-1 text-xs text-muted transition-colors hover:border-accent/60 hover:text-ink"
            >
              {ex.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
