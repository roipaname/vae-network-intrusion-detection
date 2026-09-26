import { createContext, useContext, useMemo, useState } from 'react';
import { DATASETS } from '../lib/api';

const DatasetContext = createContext(null);

export function DatasetProvider({ children }) {
  const [dataset, setDataset] = useState(DATASETS[0].id);
  const [variant, setVariant] = useState('baseline');

  const value = useMemo(() => ({ dataset, setDataset, variant, setVariant }), [dataset, variant]);

  return <DatasetContext.Provider value={value}>{children}</DatasetContext.Provider>;
}

export function useDataset() {
  const ctx = useContext(DatasetContext);
  if (!ctx) throw new Error('useDataset must be used within DatasetProvider');
  return ctx;
}
