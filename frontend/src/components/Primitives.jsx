import { AlertTriangle, Loader2 } from 'lucide-react';

export function Card({ children, className = '', padded = true, ...rest }) {
  return (
    <div className={`card ${padded ? 'card-padded' : ''} ${className}`} {...rest}>
      {children}
    </div>
  );
}

export function CardHeader({ title, hint, action }) {
  return (
    <div className="card-header">
      <div>
        <div className="card-title">{title}</div>
        {hint && <div className="card-hint">{hint}</div>}
      </div>
      {action}
    </div>
  );
}

export function KpiCard({ icon: Icon, label, value, delta, deltaTone = 'neutral' }) {
  const deltaColor =
    deltaTone === 'good' ? '#3d7f63' : deltaTone === 'bad' ? '#b4433f' : 'var(--text-muted)';
  return (
    <Card className="kpi-card">
      <div className="kpi-icon">
        <Icon size={17} strokeWidth={2} />
      </div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-label">{label}</div>
      {delta !== undefined && (
        <div className="kpi-delta" style={{ color: deltaColor }}>
          {delta}
        </div>
      )}
    </Card>
  );
}

export function Badge({ tone = 'neutral', children }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function LiveDot({ active }) {
  return <span className={`dot ${active ? 'dot-mint dot-pulse' : 'dot-coral'}`} />;
}

export function StateBlock({ loading, error, empty, emptyLabel = 'No data yet', children }) {
  if (loading) {
    return (
      <div className="state-block">
        <Loader2 size={18} className="spin" />
        <span>Loading…</span>
      </div>
    );
  }
  if (error) {
    return (
      <div className="state-block">
        <AlertTriangle size={18} color="#b4433f" />
        <span>{error.message || 'Something went wrong.'}</span>
      </div>
    );
  }
  if (empty) {
    return (
      <div className="state-block">
        <span>{emptyLabel}</span>
      </div>
    );
  }
  return children;
}
