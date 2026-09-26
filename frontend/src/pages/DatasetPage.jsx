import { useParams } from 'react-router-dom';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import TopBar from '../components/TopBar';
import { Badge, Card, CardHeader, KpiCard, StateBlock } from '../components/Primitives';
import { Database, Files, Layers, ListTree } from 'lucide-react';
import { api, DATASETS } from '../lib/api';
import { useApi } from '../lib/hooks';
import { COLORS } from '../lib/theme';

export default function DatasetPage() {
  const { datasetId } = useParams();
  const info = useApi(() => api.getDataset(datasetId), [datasetId]);
  const displayName = DATASETS.find((d) => d.id === datasetId)?.label ?? datasetId;

  const attackCategoryData = info.data
    ? Object.entries(info.data.attack_categories_train).map(([category, count]) => ({ category, count }))
    : [];

  return (
    <>
      <TopBar title={displayName} subtitle="Dataset composition and preprocessing summary." showDatasetSwitch={false} showVariantSwitch={false} />
      <div className="page-content fade-in">
        <StateBlock loading={info.loading} error={info.error}>
          {info.data && (
            <>
              <div className="grid grid-4">
                <KpiCard icon={Files} label="Training Records" value={info.data.train_records.toLocaleString()} />
                <KpiCard icon={Files} label="Test Records" value={info.data.test_records.toLocaleString()} />
                <KpiCard icon={Layers} label="Raw Features" value={info.data.num_features} />
                <KpiCard icon={Database} label="Duplicates Removed" value={info.data.duplicate_rows_in_train.toLocaleString()} />
              </div>

              <div className="grid grid-2">
                <Card>
                  <CardHeader title="Class Balance" hint="Real record counts, before any augmentation." />
                  <div className="balance-rows">
                    <BalanceRow label="Train" normal={info.data.class_distribution_train.normal} attack={info.data.class_distribution_train.attack} />
                    <BalanceRow label="Test" normal={info.data.class_distribution_test.normal} attack={info.data.class_distribution_test.attack} />
                  </div>
                </Card>

                <Card>
                  <CardHeader title="Preprocessing" hint="Applied identically to train and test." />
                  <div className="stat-grid">
                    <div className="stat-item">
                      <span className="eyebrow">Missing Values Found</span>
                      <div className="mono stat-value">{info.data.missing_values_found}</div>
                    </div>
                    <div className="stat-item">
                      <span className="eyebrow">Categorical Features</span>
                      <div>
                        {info.data.categorical_features.map((f) => (
                          <Badge key={f} tone="neutral">
                            {f}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                  <div className="stat-item" style={{ marginTop: 16 }}>
                    <span className="eyebrow">
                      <ListTree size={12} style={{ verticalAlign: -2, marginRight: 5 }} />
                      Numeric Features ({info.data.numeric_features.length})
                    </span>
                    <div className="feature-chip-list mono">{info.data.numeric_features.join(', ')}</div>
                  </div>
                </Card>
              </div>

              <Card>
                <CardHeader title="Attack Categories (Training Set)" hint="Real counts from the raw labeled data." />
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={attackCategoryData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} vertical={false} />
                    <XAxis dataKey="category" tick={{ fontSize: 12, fill: COLORS.textSecondary }} axisLine={{ stroke: COLORS.border }} tickLine={false} />
                    <YAxis tick={{ fontSize: 11, fill: COLORS.textMuted }} axisLine={false} tickLine={false} width={50} />
                    <Tooltip contentStyle={{ borderRadius: 10, border: `1px solid ${COLORS.border}`, fontSize: 12.5 }} />
                    <Bar dataKey="count" fill={COLORS.accent} radius={[5, 5, 0, 0]} maxBarSize={48} />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}
        </StateBlock>
      </div>
    </>
  );
}

function BalanceRow({ label, normal, attack }) {
  const total = normal + attack;
  const normalPct = (normal / total) * 100;
  return (
    <div className="balance-row">
      <div className="balance-row-label">
        <span>{label}</span>
        <span className="mono">{total.toLocaleString()} records</span>
      </div>
      <div className="balance-bar">
        <div className="balance-bar-fill" style={{ width: `${normalPct}%` }} />
      </div>
      <div className="balance-row-legend mono">
        <span>{normal.toLocaleString()} normal</span>
        <span>{attack.toLocaleString()} attack</span>
      </div>
    </div>
  );
}
