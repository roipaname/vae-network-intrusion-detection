import TopBar from '../components/TopBar';
import { Card, CardHeader, StateBlock } from '../components/Primitives';
import VaeLossChart from '../components/charts/VaeLossChart';
import LatentSpaceChart from '../components/charts/LatentSpaceChart';
import { useDataset } from '../context/DatasetContext';
import { api } from '../lib/api';
import { useApi } from '../lib/hooks';

function Stat({ label, value }) {
  return (
    <div className="stat-item">
      <span className="eyebrow">{label}</span>
      <div className="mono stat-value">{value}</div>
    </div>
  );
}

export default function AIModel() {
  const { dataset } = useDataset();
  const model = useApi(() => api.getModelInfo(dataset), [dataset]);
  const latent = useApi(() => api.getLatentProjection(dataset, 180), [dataset]);

  return (
    <>
      <TopBar title="AI Model" subtitle="The trained VAE and Random Forest detector behind this dashboard." showVariantSwitch={false} />
      <div className="page-content fade-in">
        <StateBlock loading={model.loading} error={model.error}>
          {model.data && (
            <>
              <div className="grid grid-2">
                <Card>
                  <CardHeader title="Variational Autoencoder" hint="Trained on real normal traffic only" />
                  <div className="stat-grid">
                    <Stat label="Status" value="Trained" />
                    <Stat label="Latent Dimension" value={model.data.vae.latent_dim} />
                    <Stat label="Hidden Layers" value={model.data.vae.hidden_dims.join(' → ')} />
                    <Stat label="Training Samples" value={model.data.vae.num_training_samples.toLocaleString()} />
                    <Stat label="Epochs" value={model.data.vae.epochs_trained} />
                    <Stat label="Synthetic Samples" value={model.data.synthetic_validation_summary.synthetic_samples.toLocaleString()} />
                    <Stat label="Final Reconstruction Loss" value={model.data.vae.final_reconstruction_loss.toFixed(4)} />
                    <Stat label="Final KL Loss" value={model.data.vae.final_kl_loss.toFixed(4)} />
                  </div>
                </Card>

                <Card>
                  <CardHeader title="Intrusion Detector" hint="Random Forest, identical hyperparameters across both experiments" />
                  <div className="stat-grid">
                    <Stat label="Model Type" value="Random Forest" />
                    <Stat label="Baseline Training Samples" value={model.data.detector.baseline.training_samples.toLocaleString()} />
                    <Stat label="Baseline Accuracy" value={`${(model.data.detector.baseline.accuracy * 100).toFixed(2)}%`} />
                    <Stat label="Baseline F1" value={model.data.detector.baseline.f1_score.toFixed(4)} />
                    <Stat label="Augmented Training Samples" value={model.data.detector.augmented.training_samples.toLocaleString()} />
                    <Stat label="Augmented Accuracy" value={`${(model.data.detector.augmented.accuracy * 100).toFixed(2)}%`} />
                    <Stat label="Augmented F1" value={model.data.detector.augmented.f1_score.toFixed(4)} />
                    <Stat label="Synthetic Fidelity (mean KS)" value={model.data.synthetic_validation_summary.mean_ks_statistic.toFixed(4)} />
                  </div>
                </Card>
              </div>

              <Card>
                <CardHeader title="VAE Training Loss" hint="Reconstruction + KL divergence, logged every epoch during training." />
                <VaeLossChart history={model.data.vae.loss_history} />
              </Card>
            </>
          )}
        </StateBlock>

        <Card>
          <CardHeader title="Real vs Synthetic Latent Space" hint="Same projection as the Overview page, for closer inspection." />
          <StateBlock loading={latent.loading} error={latent.error}>
            {latent.data && (
              <LatentSpaceChart realNormal={latent.data.real_normal} syntheticNormal={latent.data.synthetic_normal} anomaly={latent.data.anomaly} />
            )}
          </StateBlock>
        </Card>
      </div>
    </>
  );
}
