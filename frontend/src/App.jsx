import { HashRouter, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import { DatasetProvider } from './context/DatasetContext';
import Overview from './pages/Overview';
import LiveTraffic from './pages/LiveTraffic';
import Detection from './pages/Detection';
import Simulation from './pages/Simulation';
import AIModel from './pages/AIModel';
import Performance from './pages/Performance';
import History from './pages/History';
import DatasetPage from './pages/DatasetPage';

export default function App() {
  return (
    <DatasetProvider>
      <HashRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/live-traffic" element={<LiveTraffic />} />
            <Route path="/detection" element={<Detection />} />
            <Route path="/simulation" element={<Simulation />} />
            <Route path="/ai-model" element={<AIModel />} />
            <Route path="/performance" element={<Performance />} />
            <Route path="/history" element={<History />} />
            <Route path="/datasets/:datasetId" element={<DatasetPage />} />
          </Routes>
        </Layout>
      </HashRouter>
    </DatasetProvider>
  );
}
