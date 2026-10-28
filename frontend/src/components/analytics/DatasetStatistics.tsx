/** Dataset summary blocks for the Dataset Explorer and Dashboard. */
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import type { DatasetStats } from '../../types/api'
import { num } from '../../utils/format'

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line bg-panel2/50 px-3 py-2">
      <p className="text-[11px] uppercase tracking-wider text-muted">{label}</p>
      <p className="text-sm font-semibold tabular-nums">{value}</p>
    </div>
  )
}

export function DatasetStatistics({ stats }: { stats: DatasetStats }) {
  const pieData = stats.class_names.map((name) => ({
    name,
    value: stats.class_counts[name],
  }))
  const colors = stats.class_names.map((n) => (n === 'spam' ? '#f87171' : '#34d399'))

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Samples" value={stats.n_samples.toLocaleString()} />
        <Stat label="Classes" value={String(stats.n_classes)} />
        <Stat label="Duplicates removed" value={String(stats.n_duplicates_removed)} />
        <Stat label="Imbalance ratio" value={`${num(stats.imbalance_ratio, 1)} : 1`} />
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Class distribution</h3>
          <div className="h-48" role="img" aria-label="Pie chart of class distribution">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={45}
                  outerRadius={80}
                  paddingAngle={3}
                  label={(e) => `${e.name} ${(((e.value as number) / stats.n_samples) * 100).toFixed(1)}%`}
                  labelLine={false}
                  style={{ fontSize: 11, fill: '#e6edf8' }}
                >
                  {pieData.map((d, i) => (
                    <Cell key={d.name} fill={colors[i]} stroke="transparent" />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v) => [`${Number(v).toLocaleString()} messages`, '']}
                  contentStyle={{
                    background: '#16213a',
                    border: '1px solid #24304d',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <h3 className="mb-2 text-sm font-semibold">Text length (characters)</h3>
          <div className="grid grid-cols-2 gap-3">
            {stats.class_names.map((name) => {
              const pc = stats.per_class[name]
              return (
                <div key={name} className="rounded-lg border border-line bg-panel2/50 p-3">
                  <p className="text-xs font-semibold capitalize">{name}</p>
                  <ul className="mt-1 space-y-0.5 text-xs text-muted">
                    <li>mean: {num(pc.mean_chars, 1)} chars</li>
                    <li>median: {num(pc.median_chars, 0)} chars</li>
                    <li>min / max: {pc.min_chars} / {pc.max_chars}</li>
                    <li>count: {pc.count.toLocaleString()}</li>
                  </ul>
                </div>
              )
            })}
          </div>
          <p className="mt-2 text-xs text-muted">
            Overall: min {stats.text_length_chars.min}, median{' '}
            {num(stats.text_length_chars.median, 0)}, max {stats.text_length_chars.max}; mean{' '}
            {num(stats.text_length_words.mean, 1)} words.
          </p>
        </div>
      </div>
    </div>
  )
}
