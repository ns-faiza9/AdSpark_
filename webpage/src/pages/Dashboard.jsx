import { Link } from 'react-router-dom'
import {
  Database,
  BarChart3,
  Filter,
  TrendingUp,
  Binary,
  SlidersHorizontal,
  GitBranch,
  Zap,
} from 'lucide-react'
import useJson from '../hooks/useJson'

const pages = [
  { to: '/data-loading',        label: 'Data Loading',         desc: 'Load, inspect & preview the Avazu dataset',         icon: Database,              color: 'blue' },
  { to: '/eda',                 label: 'Exploratory Data Analysis', desc: '15-step statistical & visual analysis',          icon: BarChart3,             color: 'green' },
  { to: '/feature-engineering', label: 'Preprocessing',        desc: 'Missing value strategies & feature transforms',     icon: Filter,                color: 'orange' },
  { to: '/linear-regression',   label: 'Linear Regression',    desc: 'OLS baseline model & performance evaluation',       icon: TrendingUp,            color: 'purple' },
  { to: '/logistic-regression', label: 'Logistic Regression',  desc: 'Classification model for click prediction',         icon: Binary,                color: 'blue' },
  { to: '/regularization',      label: 'Regularization',       desc: 'Lasso, Ridge & Elastic Net techniques',             icon: SlidersHorizontal,     color: 'green' },
  { to: '/decision-tree',       label: 'Decision Tree',        desc: 'Tree-based classifier with depth sweep',             icon: GitBranch,             color: 'purple' },
  { to: '/ensemble',            label: 'Ensemble Learning',    desc: 'Random Forest, Gradient Boosting & AdaBoost',        icon: Zap,                   color: 'orange' },
]

export default function Dashboard() {
  const { data: logreg } = useJson('05_logistic_regression_summary.json')
  const { data: linreg } = useJson('04_linear_regression_summary.json')
  const { data: reg } = useJson('06_regularization_summary.json')

  const bestAuc = reg?.classification_view
    ? Math.max(...Object.values(reg.classification_view).map((m) => m.roc_auc))
    : null

  return (
    <>
      {/* Hero */}
      <div className="page-header" style={{ paddingBottom: 0, borderBottom: 'none' }}>
        <div className="dashboard-hero">
          <div className="hero-logo">As</div>

          <h1 className="hero-title">
            <span className="gradient-text">AdSpark</span>
            <br />
            CTR Prediction
          </h1>

          <p className="hero-tagline">
            An end-to-end machine learning project that predicts whether an online
            ad impression results in a click — built on the{' '}
            <strong>Avazu CTR Kaggle dataset</strong> with ~40 million anonymised
            ad impressions.
          </p>

          <div className="hero-stats">
            <div className="hero-stat">
              <div className="stat-number">40M+</div>
              <div className="stat-label">Impressions</div>
            </div>
            <div className="hero-stat">
              <div className="stat-number">24</div>
              <div className="stat-label">Features</div>
            </div>
            <div className="hero-stat">
              <div className="stat-number">9</div>
              <div className="stat-label">Pipeline Stages</div>
            </div>
            <div className="hero-stat">
              <div className="stat-number">6</div>
              <div className="stat-label">Models</div>
            </div>
          </div>

          <div style={{ position: 'relative', zIndex: 1, display: 'flex', gap: '14px', flexWrap: 'wrap', justifyContent: 'center' }}>
            <Link to="/data-loading" className="btn">
              <Zap size={18} /> Start Exploring
            </Link>
            <a href="#pipeline" className="btn btn-outline">
              View Pipeline
            </a>
          </div>
        </div>
      </div>

      {/* Pipeline Overview */}
      <div className="page-main" id="pipeline">
        <div className="section">
          <h2 className="section-title">
            <span className="section-badge">01</span>
            Project Pipeline
          </h2>
          <p className="section-desc">
            The complete machine learning workflow — from raw data to regularised models — 
            presented step by step. Each stage builds on the previous one.
          </p>

          <div className="quick-access-grid">
            {pages.map((page, i) => {
              const Icon = page.icon
              return (
                <Link key={page.to} to={page.to} className="quick-card">
                  <div className={`card-icon ${page.color}`}>
                    <Icon size={22} />
                  </div>
                  <h3>{page.label}</h3>
                  <p>{page.desc}</p>
                  <span className="card-index">Stage {String(i + 1).padStart(2, '0')}</span>
                </Link>
              )
            })}
          </div>
        </div>

        {/* About Section */}
        <div className="section">
          <h2 className="section-title">
            <span className="section-badge">02</span>
            About the Project
          </h2>
          <p className="section-desc">
            AdSpark studies <em>click-through rate</em> (CTR) — the probability that a user
            clicks on a displayed ad. Click prediction is the core signal behind ad ranking,
            bidding, and revenue optimisation in the digital advertising industry.
          </p>

          <div className="callout blue">
            <span className="callout-title">Goal: </span>
            Given the context of an ad impression (publisher site/app, device type,
            banner position, anonymised features), predict whether that impression
            results in a click.
          </div>

          <div className="callout green">
            <span className="callout-title">Method: </span>
            A Python pipeline loads the dataset, explores it through 15 EDA steps,
            engineers features, then trains and evaluates three classifiers —
            <strong> Linear Regression</strong>, <strong>Logistic Regression</strong>, and
            regularised variants — using log loss, ROC-AUC, precision, recall and F1.
          </div>
        </div>

        {/* Tools & Tech */}
        <div className="section">
          <h2 className="section-title">
            <span className="section-badge">03</span>
            Tools &amp; Technology
          </h2>
          <div className="chip-row">
            {['Python', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Scikit-learn', 'LightGBM', 'React.js'].map(tool => (
              <span key={tool} className="chip">{tool}</span>
            ))}
          </div>
        </div>

        {/* Model Results */}
        <div className="section">
          <h2 className="section-title">
            <span className="section-badge">04</span>
            Model Results
          </h2>
          <p className="section-desc">
            Final evaluation on the held-out 20% of the sample. All models are trained on the
            same engineered features; logistic regression is the primary model for CTR prediction.
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>ROC-AUC</th>
                  <th>Log Loss</th>
                  <th>F1</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><code>Linear Regression (OLS)</code></td>
                  <td>{linreg ? linreg.roc_auc.toFixed(4) : '—'}</td>
                  <td>{linreg ? linreg.log_loss.toFixed(4) : '—'}</td>
                  <td>{linreg ? linreg.f1.toFixed(4) : '—'}</td>
                  <td>Baseline — uncalibrated probabilities</td>
                </tr>
                <tr className="highlight">
                  <td><code>Logistic Regression</code></td>
                  <td>{logreg ? logreg.roc_auc.toFixed(4) : '—'}</td>
                  <td>{logreg ? logreg.log_loss.toFixed(4) : '—'}</td>
                  <td>{logreg ? logreg.f1.toFixed(4) : '—'}</td>
                  <td>Primary model — calibrated probabilities</td>
                </tr>
                <tr>
                  <td><code>Regularised (L1 / L2 / Elastic Net)</code></td>
                  <td>{bestAuc !== null ? bestAuc.toFixed(4) : '—'}</td>
                  <td>—</td>
                  <td>—</td>
                  <td>Same AUC, sparser coefficients</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="callout green">
            <span className="callout-title">Takeaway: </span>
            Logistic regression delivers <strong>ROC-AUC ≈ 0.65</strong> with well-calibrated
            click probabilities — the right tool for ranking and bidding. Regularization
            (Lasso) removes 4 of 23 features with no loss in performance.
          </div>
        </div>
      </div>

      <footer className="page-footer">
        AdSpark — Avazu CTR Prediction Project · Built with React.js
      </footer>
    </>
  )
}
