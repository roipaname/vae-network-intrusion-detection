import { AlertTriangle, ShieldAlert, ShieldCheck } from 'lucide-react';
import TopBar from '../components/TopBar';
import { Badge, Card, CardHeader, KpiCard, StateBlock } from '../components/Primitives';
import FeatureImportanceChart from '../components/charts/FeatureImportanceChart';
import { useDataset } from '../context/DatasetContext';
import { api } from '../lib/api';
import { usePolling, useApi } from '../lib/hooks';

export default function Detection() {
  const { dataset, variant } = useDataset();
  const summary = useApi(() => api.getSummary(dataset, variant), [dataset, variant]);
  const alerts = usePolling(() => api.getAlerts(dataset, 30, variant), [dataset, variant], 6000);
  const importance = useApi(() => api.getFeatureImportance(dataset, variant), [dataset, variant]);

  const critical = alerts.data?.alerts.filter((a) => a.severity === 'critical').length ?? 0;
  const anomalous = alerts.data?.alerts.filter((a) => a.severity === 'anomalous').length ?? 0;

  return (
    <>
      <TopBar title="Detection" subtitle="Real-time classification of replayed traffic by the active detector." showVariantSwitch />
      <div className="page-content fade-in">
        <StateBlock loading={summary.loading || alerts.loading} error={summary.error || alerts.error}>
          <div className="grid grid-3">
            <KpiCard icon={ShieldCheck} label="Normal" value={summary.data?.normal_traffic.toLocaleString() ?? '—'} />
            <KpiCard icon={AlertTriangle} label="Anomalous" value={anomalous} />
            <KpiCard icon={ShieldAlert} label="Critical" value={critical} />
          </div>
        </StateBlock>

        <div className="grid grid-2">
          <Card>
            <CardHeader title="Latest Detections" hint="Attacks the detector actually flagged, most confident first." />
            <StateBlock loading={alerts.loading} error={alerts.error} empty={alerts.data?.alerts.length === 0} emptyLabel="No alerts flagged in this sample.">
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Source</th>
                      <th>Category</th>
                      <th>Confidence</th>
                      <th>Severity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {alerts.data?.alerts.map((a) => (
                      <tr key={a.id}>
                        <td className="mono">{a.source_ip}</td>
                        <td>{a.attack_category}</td>
                        <td className="mono">{(a.confidence * 100).toFixed(1)}%</td>
                        <td>
                          <Badge tone={a.severity === 'critical' ? 'coral' : 'amber'}>{a.severity}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </StateBlock>
          </Card>

          <Card>
            <CardHeader title="Why Was This Flagged?" hint="Random Forest feature importance for the active detector." />
            <StateBlock loading={importance.loading} error={importance.error}>
              {importance.data && <FeatureImportanceChart features={importance.data.top_features.slice(0, 10)} />}
            </StateBlock>
          </Card>
        </div>
      </div>
    </>
  );
}
