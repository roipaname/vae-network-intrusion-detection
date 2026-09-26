import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { AUGMENTED_COLOR, BASELINE_COLOR, COLORS } from '../../lib/theme';

export default function RocCurveChart({ baselineCurve, augmentedCurve }) {
  const data = baselineCurve.fpr.map((fpr, i) => ({
    fpr: Number(fpr.toFixed(3)),
    Baseline: Number(baselineCurve.tpr[i].toFixed(4)),
    Augmented: Number(augmentedCurve.tpr[i].toFixed(4)),
    Random: Number(fpr.toFixed(3)),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
        <XAxis
          dataKey="fpr"
          type="number"
          domain={[0, 1]}
          tick={{ fontSize: 11, fill: COLORS.textMuted }}
          axisLine={{ stroke: COLORS.border }}
          tickLine={false}
          label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -2, fontSize: 11, fill: COLORS.textMuted }}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fontSize: 11, fill: COLORS.textMuted }}
          axisLine={false}
          tickLine={false}
          width={34}
          label={{ value: 'TPR', angle: -90, position: 'insideLeft', fontSize: 11, fill: COLORS.textMuted }}
        />
        <Tooltip contentStyle={{ borderRadius: 10, border: `1px solid ${COLORS.border}`, fontSize: 12.5 }} />
        <Legend wrapperStyle={{ fontSize: 12.5 }} iconType="plainline" />
        <Line type="monotone" dataKey="Random" stroke={COLORS.border} strokeDasharray="4 4" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="Baseline" stroke={BASELINE_COLOR} dot={false} strokeWidth={2.2} />
        <Line type="monotone" dataKey="Augmented" stroke={AUGMENTED_COLOR} dot={false} strokeWidth={2.2} />
      </LineChart>
    </ResponsiveContainer>
  );
}
