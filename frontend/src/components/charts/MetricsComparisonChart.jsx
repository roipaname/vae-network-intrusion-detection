import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { AUGMENTED_COLOR, BASELINE_COLOR, COLORS } from '../../lib/theme';

const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1_score: 'F1',
  roc_auc: 'ROC-AUC',
  false_positive_rate: 'FPR',
};

export default function MetricsComparisonChart({ baseline, augmented }) {
  const data = Object.keys(METRIC_LABELS).map((key) => ({
    metric: METRIC_LABELS[key],
    Baseline: Number((baseline[key] * 100).toFixed(2)),
    Augmented: Number((augmented[key] * 100).toFixed(2)),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }} barGap={6}>
        <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} vertical={false} />
        <XAxis dataKey="metric" tick={{ fontSize: 12, fill: COLORS.textSecondary }} axisLine={{ stroke: COLORS.border }} tickLine={false} />
        <YAxis tick={{ fontSize: 11, fill: COLORS.textMuted }} axisLine={false} tickLine={false} unit="%" width={40} />
        <Tooltip
          contentStyle={{ borderRadius: 10, border: `1px solid ${COLORS.border}`, fontSize: 12.5 }}
          formatter={(value) => `${value}%`}
        />
        <Legend wrapperStyle={{ fontSize: 12.5 }} iconType="circle" iconSize={8} />
        <Bar dataKey="Baseline" fill={BASELINE_COLOR} radius={[5, 5, 0, 0]} maxBarSize={26} />
        <Bar dataKey="Augmented" fill={AUGMENTED_COLOR} radius={[5, 5, 0, 0]} maxBarSize={26} />
      </BarChart>
    </ResponsiveContainer>
  );
}
