/** Confusion matrix rendered as an accessible colour-coded table. */
import { cn } from '../../utils/format'

export function ConfusionMatrixChart({
  labels,
  matrix,
  positiveClass = 'spam',
}: {
  labels: string[]
  matrix: number[][]
  positiveClass?: string
}) {
  const max = Math.max(1, ...matrix.flat())
  return (
    <figure>
      <table className="w-full border-separate border-spacing-1 text-sm">
        <caption className="sr-only">
          Confusion matrix. Rows are true labels, columns are predicted labels.
        </caption>
        <thead>
          <tr>
            <th scope="col" className="p-1 text-xs text-muted">
              true \ pred
            </th>
            {labels.map((l) => (
              <th key={l} scope="col" className="p-1 text-center text-xs font-semibold text-muted">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th scope="row" className="p-1 text-right text-xs font-semibold text-muted">
                {labels[i]}
              </th>
              {row.map((v, j) => {
                const isDiagonal = i === j
                const intensity = 0.12 + 0.55 * (v / max)
                const color = isDiagonal ? '52, 211, 153' : '248, 113, 113'
                return (
                  <td
                    key={j}
                    className={cn(
                      'rounded-md p-3 text-center font-mono tabular-nums',
                      v === 0 && 'text-muted',
                    )}
                    style={{ background: `rgba(${color}, ${v === 0 ? 0.05 : intensity})` }}
                    aria-label={`true ${labels[i]}, predicted ${labels[j]}: ${v}`}
                  >
                    {v}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <figcaption className="mt-2 text-xs text-muted">
        Diagonal (green) = correct. Off-diagonal red = mistakes; false positives flag
        legitimate mail as {positiveClass}, false negatives miss {positiveClass}.
      </figcaption>
    </figure>
  )
}
