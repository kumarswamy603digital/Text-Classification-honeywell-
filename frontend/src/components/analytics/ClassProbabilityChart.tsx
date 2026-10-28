/** Bar chart of predicted class probabilities (Recharts). */
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { pct } from '../../utils/format'

export function ClassProbabilityChart({
  probabilities,
}: {
  probabilities: Record<string, number>
}) {
  const entries = Object.entries(probabilities)
  const data = entries.map(([cls, p]) => ({
    name: cls === 'spam' ? 'spam' : 'ham (legitimate)',
    probability: p,
    fill: cls === 'spam' ? '#f87171' : '#34d399',
  }))

  return (
    <figure>
      <div className="h-44" role="img" aria-label="Class probability bar chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#24304d" vertical={false} />
            <XAxis dataKey="name" tick={{ fill: '#8b98b3', fontSize: 12 }} stroke="#24304d" />
            <YAxis domain={[0, 1]} tick={{ fill: '#8b98b3', fontSize: 11 }} stroke="#24304d" />
            <Tooltip
              formatter={(v) => [pct(Number(v)), 'probability']}
              contentStyle={{
                background: '#16213a',
                border: '1px solid #24304d',
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
              {data.map((d) => (
                <Cell key={d.name} fill={d.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-1 text-xs text-muted">
        {entries.map(([cls, p]) => `${cls}: ${pct(p)}`).join(' · ')}
      </figcaption>
    </figure>
  )
}
