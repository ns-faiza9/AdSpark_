import { NavLink, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  Database,
  BarChart3,
  Filter,
  TrendingUp,
  Binary,
  SlidersHorizontal,
  GitBranch,
  Zap,
  Sun,
  Moon,
} from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

const navItems = [
  { to: '/',                    label: 'Dashboard',           icon: LayoutDashboard,   index: '01' },
  { to: '/data-loading',        label: 'Data Loading',        icon: Database,          index: '02' },
  { to: '/eda',                 label: 'EDA',                 icon: BarChart3,         index: '03' },
  { to: '/feature-engineering', label: 'Preprocessing',       icon: Filter,            index: '04' },
  { to: '/linear-regression',   label: 'Linear Regression',   icon: TrendingUp,        index: '05' },
  { to: '/logistic-regression', label: 'Logistic Regression', icon: Binary,            index: '06' },
  { to: '/regularization',      label: 'Regularization',      icon: SlidersHorizontal, index: '07' },
  { to: '/decision-tree',       label: 'Decision Tree',       icon: GitBranch,         index: '08' },
  { to: '/ensemble',            label: 'Ensemble Learning',   icon: Zap,               index: '09' },
]

export default function Sidebar() {
  const location = useLocation()
  const { theme, toggleTheme } = useTheme()

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="brand-logo">As</div>
        <div className="brand-text">
          <div className="brand-name">AdSpark</div>
          <div className="brand-tagline">CTR Prediction</div>
        </div>
      </div>

      {/* Navigation */}
      <div className="sidebar-section-label">Navigation</div>
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.to
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={`nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-index">{item.index}</span>
              <Icon className="nav-icon" />
              <span>{item.label}</span>
            </NavLink>
          )
        })}
      </nav>

      {/* Theme Toggle */}
      <div className="theme-toggle-wrap">
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
        </button>
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <p>AdSpark v1.0<br />Avazu CTR Prediction</p>
      </div>
    </aside>
  )
}
