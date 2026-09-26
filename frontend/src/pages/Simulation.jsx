import { useState } from 'react';
import { CheckCircle2, PlayCircle, XCircle, Zap } from 'lucide-react';
import TopBar from '../components/TopBar';
import { Badge, Card, CardHeader, StateBlock } from '../components/Primitives';
import LiveNetworkVisualization from '../components/LiveNetworkVisualization';
import { useDataset } from '../context/DatasetContext';
import { api } from '../lib/api';

const TRAFFIC_TYPES = [
  { id: 'normal', label: 'Normal Traffic' },
  { id: 'anomaly', label: 'Anomaly' },
  { id: 'attack', label: 'Attack Simulation' },
];

export default function Simulation() {
  const { dataset, variant } = useDataset();
  const [trafficType, setTrafficType] = useState('normal');
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const runSimulation = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await api.simulateAttack({ dataset, traffic_type: trafficType, variant });
      setResult(res);
    } catch (err) {
      setError(err);
    } finally {
      setRunning(false);
    }
  };

  const vizEvents = result
    ? [{ id: result.record_index, prediction: result.prediction, confidence: result.confidence }]
    : [];

  return (
    <>
      <TopBar
        title="Simulation"
        subtitle="Replays a real dataset record through the detector. No real attacks are performed."
        showVariantSwitch
      />
      <div className="page-content fade-in">
        <Card>
          <CardHeader title="Network Simulation" hint="A real record from the test set is pulled and scored on demand." />
          <LiveNetworkVisualization events={vizEvents} />

          <div className="sim-controls">
            <div className="segmented">
              {TRAFFIC_TYPES.map((t) => (
                <button key={t.id} className={trafficType === t.id ? 'active' : ''} onClick={() => setTrafficType(t.id)}>
                  {t.label}
                </button>
              ))}
            </div>
            <button className="btn btn-primary" onClick={runSimulation} disabled={running}>
              <PlayCircle size={16} />
              {running ? 'Running…' : 'Start Simulation'}
            </button>
          </div>
        </Card>

        <StateBlock error={error}>
          {result && (
            <div className="grid grid-4 fade-in">
              <Card className="kpi-card">
                <div className="kpi-icon">
                  <Zap size={17} />
                </div>
                <div className="kpi-value">{result.prediction}</div>
                <div className="kpi-label">Prediction</div>
              </Card>
              <Card className="kpi-card">
                <div className="kpi-icon">
                  <Zap size={17} />
                </div>
                <div className="kpi-value">{(result.confidence * 100).toFixed(1)}%</div>
                <div className="kpi-label">Confidence</div>
              </Card>
              <Card className="kpi-card">
                <div className="kpi-icon">
                  <Zap size={17} />
                </div>
                <div className="kpi-value">{result.latency_ms.toFixed(2)} ms</div>
                <div className="kpi-label">Inference Latency</div>
              </Card>
              <Card className="kpi-card">
                <div className="kpi-icon">
                  {result.detected_correctly ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
                </div>
                <div className="kpi-value" style={{ fontSize: 18 }}>
                  <Badge tone={result.detected_correctly ? 'mint' : 'coral'}>
                    {result.detected_correctly ? 'Correctly Classified' : 'Misclassified'}
                  </Badge>
                </div>
                <div className="kpi-label">Detection Status</div>
              </Card>
            </div>
          )}
        </StateBlock>

        {result && (
          <Card>
            <CardHeader title="Record Detail" hint={`Test set index ${result.record_index} · ground truth: ${result.true_label}`} />
            <div className="detail-grid mono">
              <div>
                <span className="eyebrow">Protocol</span>
                <div>{result.protocol}</div>
              </div>
              <div>
                <span className="eyebrow">Service</span>
                <div>{result.service}</div>
              </div>
              <div>
                <span className="eyebrow">Attack Category</span>
                <div>{result.attack_category}</div>
              </div>
              <div>
                <span className="eyebrow">Model</span>
                <div>random_forest_{variant}</div>
              </div>
            </div>
          </Card>
        )}
      </div>
    </>
  );
}
