import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { COLORS } from '../../lib/theme';

export default function FeatureImportanceChart({ features }) {
  const data = [...features].reverse().map((f) => ({
    feature: f.feature,
    importance: Number((f.importance * 100).toFixed(2)),
  }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(260, data.length * 28)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 20, left: 8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fill: COLORS.textMuted }} axisLine={{ stroke: COLORS.border }} tickLine={false} unit="%" />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fontSize: 12, fill: COLORS.textSecondary, fontFamily: 'JetBrains Mono, monospace' }}
          axisLine={false}
          tickLine={false}
          width={150}
        />
        <Tooltip contentStyle={{ borderRadius: 10, border: `1px solid ${COLORS.border}`, fontSize: 12.5 }} formatter={(v) => `${v}%`} />
        <Bar dataKey="importance" fill={COLORS.accent} radius={[0, 5, 5, 0]} maxBarSize={16} />
      </BarChart>
    </ResponsiveContainer>
  );
}
