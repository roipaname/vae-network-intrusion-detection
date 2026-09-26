import TopBar from '../components/TopBar';
import { Badge, Card, CardHeader, StateBlock } from '../components/Primitives';
import MetricsComparisonChart from '../components/charts/MetricsComparisonChart';
import RocCurveChart from '../components/charts/RocCurveChart';
import { useDataset } from '../context/DatasetContext';
import { api } from '../lib/api';
import { useApi } from '../lib/hooks';

const ROWS = [
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'precision', label: 'Precision' },
  { key: 'recall', label: 'Recall' },
  { key: 'f1_score', label: 'F1 Score' },
  { key: 'roc_auc', label: 'ROC-AUC' },
  { key: 'false_positive_rate', label: 'False Positive Rate' },
];

export default function Performance() {
  const { dataset } = useDataset();
  const metrics = useApi(() => api.getMetrics(dataset), [dataset]);

  return (
    <>
      <TopBar title="Performance" subtitle="Baseline vs VAE-augmented detector, evaluated on the same untouched real test set." showVariantSwitch={false} />
      <div className="page-content fade-in">
        <StateBlock loading={metrics.loading} error={metrics.error}>
          {metrics.data && (
            <>
              <Card>
                <CardHeader
                  title="Metric Comparison"
                  hint={
                    <>
                      <Badge tone="neutral">Baseline</Badge> vs <Badge tone="mint">Augmented</Badge>
                    </>
                  }
                />
                <MetricsComparisonChart baseline={metrics.data.baseline} augmented={metrics.data.augmented} />
              </Card>

              <div className="grid grid-2">
                <Card>
                  <CardHeader title="ROC Curve" hint="Interpolated onto a fixed FPR grid for direct overlay." />
                  <RocCurveChart baselineCurve={metrics.data.baseline_roc_curve} augmentedCurve={metrics.data.augmented_roc_curve} />
                </Card>

                <Card>
                  <CardHeader title="Metric Deltas" hint="Augmented minus baseline." />
                  <div className="table-wrap">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Baseline</th>
                          <th>Augmented</th>
                          <th>Δ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ROWS.map((row) => {
                          const delta = metrics.data.deltas[row.key];
                          const improved = row.key === 'false_positive_rate' ? delta < 0 : delta > 0;
                          return (
                            <tr key={row.key}>
                              <td>{row.label}</td>
                              <td className="mono">{(metrics.data.baseline[row.key] * 100).toFixed(2)}%</td>
                              <td className="mono">{(metrics.data.augmented[row.key] * 100).toFixed(2)}%</td>
                              <td className="mono">
                                <Badge tone={Math.abs(delta) < 0.0005 ? 'neutral' : improved ? 'mint' : 'coral'}>
                                  {delta >= 0 ? '+' : ''}
                                  {(delta * 100).toFixed(2)}pt
                                </Badge>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            </>
          )}
        </StateBlock>
      </div>
    </>
  );
}
