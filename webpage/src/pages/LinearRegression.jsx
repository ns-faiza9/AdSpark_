import { useState } from 'react'
import { TrendingUp, Gauge, Target, Scale } from 'lucide-react'
import useJson from '../hooks/useJson'
import FigureCard from '../components/FigureCard'

const REG_LABELS = ['L2 (Ridge Regression)', 'L1 (Lasso Regression)', 'None (Ordinary Least Squares)']

export default function LinearRegression() {
  const { data: linreg } = useJson('04_linear_regression_summary.json')
  const { data: reg } = useJson('06_regularization_summary.json')
  const [activeReg, setActiveReg] = useState('L2 (Ridge Regression)')

  const penalties = linreg?.penalties || {}
  const activePenalty = penalties[activeReg] || penalties['None (Ordinary Least Squares)'] || {}

  const stats = [
    { label: 'ROC-AUC', value: linreg?.roc_auc ? linreg.roc_auc.toFixed(4) : '—', icon: Gauge, sub: 'discrimination power', accent: 'accent-green' },
    { label: 'RMSE', value: linreg?.rmse ? linreg.rmse.toFixed(4) : '—', icon: Scale, sub: 'multivariate error' },
    { label: 'MSE', value: linreg?.mse ? linreg.mse.toFixed(4) : '—', icon: Target, sub: 'mean squared error' },
    { label: 'R²', value: linreg?.r2 ? linreg.r2.toFixed(4) : '—', icon: TrendingUp, sub: 'variance explained', accent: 'accent-purple' },
  ]

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 05</div>
        <h1><span className="gradient-text">Linear Regression Analysis</span></h1>
        <p className="subtitle">
          Predicting continuous click probabilities and placement metrics using academic performance, frequency encodings, and feature scores.
        </p>
      </div>

      <div className="page-main page-enter">

        {/* Regularization penalty dropdown */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <label style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Regularization Penalty:
          </label>
          <select
            value={activeReg}
            onChange={(e) => setActiveReg(e.target.value)}
            style={{
              padding: '0.4rem 0.8rem',
              borderRadius: 6,
              border: '1px solid var(--border)',
              background: 'var(--surface)',
              color: 'var(--text)',
              fontSize: '0.875rem',
              fontWeight: 500,
              cursor: 'pointer',
            }}
          >
            {REG_LABELS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            Active Model: <strong>{activeReg}</strong>
          </span>
        </div>

        {/* Section 1: Simple Linear Regression */}
        <section className="section">
          <h2 className="section-title">
            <span className="section-badge">01</span>
            1. Simple Linear Regression: Click Rate ~ {activePenalty.feature || 'Feature'}
          </h2>
          <p className="section-desc" style={{ marginBottom: '1rem' }}>
            Evaluating the direct linear relationship between overall predictor signal and placement click rate.
          </p>

          {/* Equation Box */}
          <div className="callout blue" style={{ marginBottom: '1.5rem', fontFamily: 'monospace', fontSize: '0.9rem', fontWeight: 600 }}>
            {activePenalty.equation || `Click Rate = β₀ + β₁ * ${activePenalty.feature || 'Feature'}`}
          </div>

          {/* Two-column layout: Left = Table, Right = Fitted Line Graph */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.1fr', gap: '1.75rem', alignItems: 'start', marginBottom: '1.5rem' }}>

            {/* Left Column: Model Performance Metrics Table */}
            <div>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>
                Model Performance Metrics
              </h3>
              <div className="table-wrap">
                <table className="data-table">
                  <tbody>
                    <tr>
                      <td><strong>Training MSE</strong></td>
                      <td>{activePenalty.train_mse != null ? activePenalty.train_mse.toFixed(5) : '—'}</td>
                    </tr>
                    <tr>
                      <td><strong>Testing MSE</strong></td>
                      <td>{activePenalty.test_mse != null ? activePenalty.test_mse.toFixed(5) : '—'}</td>
                    </tr>
                    <tr>
                      <td><strong>Training R² Score</strong></td>
                      <td>{activePenalty.train_r2 != null ? activePenalty.train_r2.toFixed(4) : '—'}</td>
                    </tr>
                    <tr className="highlight">
                      <td><strong>Testing R² Score</strong></td>
                      <td><strong>{activePenalty.test_r2 != null ? activePenalty.test_r2.toFixed(4) : '—'}</strong></td>
                    </tr>
                    <tr>
                      <td><strong>Sample Size</strong></td>
                      <td>{activePenalty.sample_size ? `${activePenalty.sample_size.toLocaleString()} records` : '808,579 records'}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Right Column: Fitted Regression Line Scatter Plot */}
            <div>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
                Fitted Regression Line
              </h3>
              <FigureCard
                src={activePenalty.figure || 'lr_fitted_ols.png'}
                title={`Simple Linear Regression: Click Rate vs ${activePenalty.feature || 'Feature'} (${activeReg.replace(/\s*\(.*\)/, '')})`}
                purpose="Scatter plot of actual observations vs the fitted regression line."
                observation={`Fit equation: y = ${activePenalty.intercept ?? '0.17'} + ${activePenalty.slope ?? '0.08'}*x`}
              />
            </div>
          </div>

          {/* Stat Cards */}
          <div className="stats-grid" style={{ marginTop: '1rem' }}>
            {stats.map((s) => {
              const Icon = s.icon
              return (
                <div key={s.label} className={`stat-card ${s.accent || ''}`}>
                  <div className="stat-label"><Icon size={14} style={{ verticalAlign: -2, marginRight: 6 }} />{s.label}</div>
                  <div className="stat-value">{s.value}</div>
                  <div className="stat-sub">{s.sub}</div>
                </div>
              )
            })}
          </div>
        </section>

        {/* Diagnostics Section */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">02</span>Model Diagnostics & Coefficients</h2>
          <div className="figure-grid">
            <FigureCard
              src="lr_01_coefficients.png"
              title="Top 20 linear regression coefficients"
              purpose="Magnitudes of predictor coefficients across engineered features."
              observation="Frequency-encoded identifiers and category codes exert the strongest linear influence."
              fullWidth
            />
            <FigureCard
              src="lr_02_prediction_distribution.png"
              title="Predicted probability distribution"
              purpose="Spread of raw linear predictions across [0, 1]."
              observation="Linear regression creates uncalibrated probabilities, motivating logistic classification."
              fullWidth
            />
          </div>
        </section>
      </div>
    </>
  )
}