import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { COLORS } from '../../lib/theme';

export default function VaeLossChart({ history }) {
  const data = history.map((h) => ({
    epoch: h.epoch,
    Total: Number(h.loss.toFixed(3)),
    Reconstruction: Number(h.recon_loss.toFixed(3)),
    KL: Number(h.kl_loss.toFixed(3)),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
        <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: COLORS.textMuted }} axisLine={{ stroke: COLORS.border }} tickLine={false} />
        <YAxis tick={{ fontSize: 11, fill: COLORS.textMuted }} axisLine={false} tickLine={false} width={36} />
        <Tooltip contentStyle={{ borderRadius: 10, border: `1px solid ${COLORS.border}`, fontSize: 12.5 }} />
        <Legend wrapperStyle={{ fontSize: 12.5 }} iconType="plainline" />
        <Line type="monotone" dataKey="Total" stroke={COLORS.secondaryDark} dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="Reconstruction" stroke={COLORS.accent} dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="KL" stroke={COLORS.warning} dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}
