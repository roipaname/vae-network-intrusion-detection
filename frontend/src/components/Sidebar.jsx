import { NavLink } from 'react-router-dom';
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  History,
  LayoutDashboard,
  PlayCircle,
  ShieldAlert,
} from 'lucide-react';
import logo from '../assets/logo.png';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutDashboard, end: true },
  { to: '/live-traffic', label: 'Live Traffic', icon: Activity },
  { to: '/detection', label: 'Detection', icon: ShieldAlert },
  { to: '/simulation', label: 'Simulation', icon: PlayCircle },
  { to: '/ai-model', label: 'AI Model', icon: BrainCircuit },
  { to: '/performance', label: 'Performance', icon: BarChart3 },
  { to: '/history', label: 'History', icon: History },
];

const DATASET_LINKS = [
  { to: '/datasets/nsl_kdd', label: 'NSL-KDD' },
  { to: '/datasets/unsw_nb15', label: 'UNSW-NB15' },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <img src={logo} alt="NEXUS AI" className="sidebar-logo" />
        <div>
          <div className="sidebar-brand-name">NEXUS AI</div>
          <div className="sidebar-brand-tagline">Intelligent Network Security</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
            <Icon size={17} strokeWidth={2} />
            <span>{label}</span>
          </NavLink>
        ))}

        <div className="sidebar-section">
          <div className="sidebar-section-label">
            <Database size={13} />
            <span>Datasets</span>
          </div>
          {DATASET_LINKS.map(({ to, label }) => (
            <NavLink key={to} to={to} className={({ isActive }) => `sidebar-link sidebar-link-sub ${isActive ? 'active' : ''}`}>
              <span className="sidebar-sub-dot" />
              <span>{label}</span>
            </NavLink>
          ))}
        </div>
      </nav>

      <div className="sidebar-footer">
        <span className="eyebrow" style={{ color: 'var(--text-on-dark-muted)' }}>
          Research Prototype
        </span>
      </div>
    </aside>
  );
}
