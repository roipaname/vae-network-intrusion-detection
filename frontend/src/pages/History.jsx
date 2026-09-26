import { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import TopBar from '../components/TopBar';
import { Badge, Card, CardHeader, StateBlock } from '../components/Primitives';
import { useDataset } from '../context/DatasetContext';
import { api } from '../lib/api';
import { useApi } from '../lib/hooks';

const PAGE_SIZE = 20;

const STATUS_TONE = {
  detected: 'mint',
  normal: 'neutral',
  missed: 'coral',
  false_alarm: 'amber',
};

const STATUS_LABEL = {
  detected: 'Detected',
  normal: 'Normal',
  missed: 'Missed',
  false_alarm: 'False Alarm',
};

export default function History() {
  const { dataset, variant } = useDataset();
  const [page, setPage] = useState(0);
  const connections = useApi(() => api.getConnections(dataset, PAGE_SIZE, page * PAGE_SIZE, variant), [dataset, variant, page]);

  const total = connections.data?.total ?? 0;
  const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);

  return (
    <>
      <TopBar title="History" subtitle="A stable, paginated view over the real test set." showVariantSwitch />
      <div className="page-content fade-in">
        <Card>
          <CardHeader
            title="Historical Connections"
            hint={total ? `${total.toLocaleString()} records total` : undefined}
            action={
              <div className="pager">
                <button className="btn btn-outline" disabled={page === 0} onClick={() => setPage((p) => Math.max(0, p - 1))}>
                  <ChevronLeft size={15} />
                </button>
                <span className="mono pager-label">
                  {page + 1} / {maxPage + 1}
                </span>
                <button className="btn btn-outline" disabled={page >= maxPage} onClick={() => setPage((p) => Math.min(maxPage, p + 1))}>
                  <ChevronRight size={15} />
                </button>
              </div>
            }
          />
          <StateBlock loading={connections.loading} error={connections.error}>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Source</th>
                    <th>Destination</th>
                    <th>Protocol</th>
                    <th>Traffic Type</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {connections.data?.rows.map((row) => (
                    <tr key={row.id}>
                      <td className="mono">{new Date(row.timestamp).toLocaleString()}</td>
                      <td className="mono">{row.source_ip}</td>
                      <td className="mono">{row.destination_ip}</td>
                      <td className="mono">{row.protocol}</td>
                      <td>{row.traffic_type}</td>
                      <td>
                        <Badge tone={row.prediction === 'attack' ? 'coral' : 'mint'}>{row.prediction}</Badge>
                      </td>
                      <td className="mono">{(row.confidence * 100).toFixed(1)}%</td>
                      <td>
                        <Badge tone={STATUS_TONE[row.status]}>{STATUS_LABEL[row.status]}</Badge>
                      </td>
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
