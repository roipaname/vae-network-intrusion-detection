const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

function qs(params = {}) {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) search.set(key, value);
  });
  const str = search.toString();
  return str ? `?${str}` : '';
}

export const api = {
  health: () => request('/health'),
  listDatasets: () => request('/api/datasets'),
  getDataset: (dataset) => request(`/api/datasets/${dataset}`),
  getModelInfo: (dataset) => request(`/api/model/${dataset}`),
  getLatentProjection: (dataset, nPerGroup = 200) =>
    request(`/api/latent-projection${qs({ dataset, n_per_group: nPerGroup })}`),
  getSummary: (dataset, variant) => request(`/api/summary${qs({ dataset, variant })}`),
  getMetrics: (dataset) => request(`/api/metrics${qs({ dataset })}`),
  getFeatureImportance: (dataset, variant) => request(`/api/feature-importance${qs({ dataset, variant })}`),
  getTraffic: (dataset, limit = 20, variant) => request(`/api/traffic${qs({ dataset, limit, variant })}`),
  getAlerts: (dataset, limit = 20, variant) => request(`/api/alerts${qs({ dataset, limit, variant })}`),
  getConnections: (dataset, limit = 50, offset = 0, variant) =>
    request(`/api/connections${qs({ dataset, limit, offset, variant })}`),
  predict: (payload) => request('/api/predict', { method: 'POST', body: JSON.stringify(payload) }),
  simulateAttack: (payload) => request('/api/simulate-attack', { method: 'POST', body: JSON.stringify(payload) }),
};

export const DATASETS = [
  { id: 'nsl_kdd', label: 'NSL-KDD' },
  { id: 'unsw_nb15', label: 'UNSW-NB15' },
];
