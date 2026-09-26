import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts';
import { COLORS } from '../../lib/theme';

export default function LatentSpaceChart({ realNormal, syntheticNormal, anomaly }) {
  const toPoints = (arr) => arr.map(([x, y]) => ({ x, y }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
        <XAxis type="number" dataKey="x" tick={{ fontSize: 11, fill: COLORS.textMuted }} axisLine={{ stroke: COLORS.border }} tickLine={false} />
        <YAxis type="number" dataKey="y" tick={{ fontSize: 11, fill: COLORS.textMuted }} axisLine={false} tickLine={false} />
        <ZAxis range={[36, 36]} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: 10, border: `1px solid ${COLORS.border}`, fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12.5 }} iconType="circle" iconSize={8} />
        <Scatter name="Real normal" data={toPoints(realNormal)} fill={COLORS.success} fillOpacity={0.85} />
        <Scatter name="Synthetic normal" data={toPoints(syntheticNormal)} fill="none" stroke={COLORS.accent} strokeWidth={1.4} />
        <Scatter name="Anomaly (attack)" data={toPoints(anomaly)} fill={COLORS.threat} fillOpacity={0.55} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
