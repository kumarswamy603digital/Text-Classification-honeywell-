/** CSV upload with drag & drop, validation feedback and column selection. */
import * as React from 'react'
import { UploadCloud } from 'lucide-react'
import { cn } from '../../utils/format'

export const MAX_FILE_BYTES = 2 * 1024 * 1024 // 2 MB

export function FileUploadPanel({
  onFile,
  fileName,
  error,
  disabled,
}: {
  onFile: (file: File) => void
  fileName: string | null
  error: string | null
  disabled?: boolean
}) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = React.useState(false)

  function accept(file: File | undefined) {
    if (!file) return
    onFile(file)
  }

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload a CSV file"
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault()
          inputRef.current?.click()
        }
      }}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        if (!disabled) accept(e.dataTransfer.files[0])
      }}
      className={cn(
        'flex cursor-pointer flex-col items-center gap-2 rounded-xl border-2 border-dashed p-8 text-center transition-colors',
        dragging ? 'border-accent bg-accent/10' : 'border-line bg-panel2/40 hover:border-accent/50',
        disabled && 'pointer-events-none opacity-50',
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="sr-only"
        onChange={(e) => accept(e.target.files?.[0])}
      />
      <UploadCloud className="h-8 w-8 text-accent" aria-hidden />
      <p className="text-sm font-medium">
        {fileName ? `Loaded: ${fileName}` : 'Drop a CSV here or click to browse'}
      </p>
      <p className="text-xs text-muted">
        One text column; max {MAX_FILE_BYTES / 1024 / 1024} MB · up to 200 rows per run
      </p>
      {error && (
        <p role="alert" className="mt-1 text-xs font-medium text-bad">
          {error}
        </p>
      )}
    </div>
  )
}
