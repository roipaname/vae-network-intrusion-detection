import TopBar from '../components/TopBar';
import { Badge, Card, CardHeader, StateBlock } from '../components/Primitives';
import LiveNetworkVisualization from '../components/LiveNetworkVisualization';
import { useDataset } from '../context/DatasetContext';
import { api } from '../lib/api';
import { usePolling } from '../lib/hooks';

export default function LiveTraffic() {
  const { dataset, variant } = useDataset();
  const traffic = usePolling(() => api.getTraffic(dataset, 24, variant), [dataset, variant], 4000);

  return (
    <>
      <TopBar title="Live Traffic" subtitle="Replayed real test-set connections, scored by the active detector in real time." showVariantSwitch />
      <div className="page-content fade-in">
        <Card>
          <CardHeader title="Network Flow" hint="Mint = normal, coral = flagged as attack. Refreshes every 4 seconds." />
          <StateBlock loading={traffic.loading} error={traffic.error}>
            {traffic.data && <LiveNetworkVisualization events={traffic.data.events} />}
          </StateBlock>
        </Card>

        <Card>
          <CardHeader title="Recent Events" hint={`${traffic.data?.events.length ?? 0} connections shown`} />
          <StateBlock loading={traffic.loading} error={traffic.error} empty={traffic.data?.events.length === 0}>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Source</th>
                    <th>Destination</th>
                    <th>Protocol</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {traffic.data?.events.map((e) => (
                    <tr key={e.id}>
                      <td className="mono">{new Date(e.timestamp).toLocaleTimeString()}</td>
                      <td className="mono">{e.source_ip}</td>
                      <td className="mono">{e.destination_ip}</td>
                      <td className="mono">{e.protocol}</td>
                      <td>
                        <Badge tone={e.prediction === 'attack' ? 'coral' : 'mint'}>{e.prediction}</Badge>
                      </td>
                      <td className="mono">{(e.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </StateBlock>
        </Card>
      </div>
    </>
  );
}
