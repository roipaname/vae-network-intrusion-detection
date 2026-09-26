import { AlertOctagon, Gauge, ShieldCheck, Waypoints } from 'lucide-react';
import TopBar from '../components/TopBar';
import { Card, CardHeader, KpiCard, StateBlock } from '../components/Primitives';
import LiveNetworkVisualization from '../components/LiveNetworkVisualization';
import LatentSpaceChart from '../components/charts/LatentSpaceChart';
import { useDataset } from '../context/DatasetContext';
import { api } from '../lib/api';
import { useApi, usePolling } from '../lib/hooks';

export default function Overview() {
  const { dataset } = useDataset();
  const summary = useApi(() => api.getSummary(dataset), [dataset]);
  const traffic = usePolling(() => api.getTraffic(dataset, 14), [dataset], 5000);
  const latent = useApi(() => api.getLatentProjection(dataset, 180), [dataset]);

  return (
    <>
      <TopBar title="Overview" subtitle="What is happening right now, at a glance." showVariantSwitch={false} />
      <div className="page-content fade-in">
        <StateBlock loading={summary.loading} error={summary.error}>
          {summary.data && (
            <div className="grid grid-4">
              <KpiCard icon={Waypoints} label="Total Connections" value={summary.data.total_connections.toLocaleString()} />
              <KpiCard icon={ShieldCheck} label="Normal Traffic" value={summary.data.normal_traffic.toLocaleString()} />
              <KpiCard icon={AlertOctagon} label="Anomalies" value={summary.data.anomalies.toLocaleString()} />
              <KpiCard icon={Gauge} label="False Positive Rate" value={`${(summary.data.false_positive_rate * 100).toFixed(2)}%`} />
            </div>
          )}
        </StateBlock>

        <Card>
          <CardHeader
            title="Live Network Activity"
            hint="Real test-set records replayed through the trained detector, refreshed every few seconds."
          />
          <StateBlock loading={traffic.loading} error={traffic.error} empty={traffic.data?.events.length === 0}>
            {traffic.data && <LiveNetworkVisualization events={traffic.data.events} />}
          </StateBlock>
        </Card>

        <Card>
          <CardHeader
            title="AI Learned Traffic Profile"
            hint="Real normal, VAE-synthesized normal, and real attack traffic, encoded through the trained VAE and projected to 2D with PCA."
          />
          <StateBlock loading={latent.loading} error={latent.error}>
            {latent.data && (
              <LatentSpaceChart
                realNormal={latent.data.real_normal}
                syntheticNormal={latent.data.synthetic_normal}
                anomaly={latent.data.anomaly}
              />
            )}
          </StateBlock>
        </Card>
      </div>
    </>
  );
}
