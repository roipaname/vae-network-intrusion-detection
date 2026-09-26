import { FlaskConical } from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { useBackendStatus } from '../lib/hooks';
import { DATASETS } from '../lib/api';
import { LiveDot } from './Primitives';

export default function TopBar({ title, subtitle, showDatasetSwitch = true, showVariantSwitch = false }) {
  const online = useBackendStatus();
  const { dataset, setDataset, variant, setVariant } = useDataset();

  return (
    <header className="topbar">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>

      <div className="topbar-controls">
        {showDatasetSwitch && (
          <div className="segmented">
            {DATASETS.map((d) => (
              <button key={d.id} className={dataset === d.id ? 'active' : ''} onClick={() => setDataset(d.id)}>
                {d.label}
              </button>
            ))}
          </div>
        )}

        {showVariantSwitch && (
          <div className="segmented">
            {['baseline', 'augmented'].map((v) => (
              <button key={v} className={variant === v ? 'active' : ''} onClick={() => setVariant(v)}>
                {v === 'baseline' ? 'Baseline' : 'Augmented'}
              </button>
            ))}
          </div>
        )}

        <div className="badge badge-mint">
          <FlaskConical size={12} />
          Research Mode
        </div>

        <div className="live-indicator">
          <LiveDot active={online === true} />
          <span>{online === null ? 'Connecting' : online ? 'Live' : 'Offline'}</span>
        </div>
      </div>
    </header>
  );
}
