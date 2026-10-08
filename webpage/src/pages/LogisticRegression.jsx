import { useState } from 'react'
import { Binary, Gauge, Target, Crosshair, Scale } from 'lucide-react'
import useJson from '../hooks/useJson'
import FigureCard from '../components/FigureCard'

const PENALTY_LABELS = ['L2 (Ridge Regularization)', 'L1 (Lasso Regularization)', 'None (Unscaled)']

export default function LogisticRegression() {
  const { data } = useJson('05_logistic_regression_summary.json')
  const [activePenalty, setActivePenalty] = useState('L2 (Ridge Regularization)')

  const testAccDisplay = data
    ? (data.test_accuracy > 1 ? data.test_accuracy.toFixed(2) : (data.test_accuracy * 100).toFixed(2))
    : '58.54'

  const trainAccDisplay = data?.train_accuracy
    ? (data.train_accuracy > 1 ? data.train_accuracy.toFixed(2) : (data.train_accuracy * 100).toFixed(2))
    : '58.57'

  const stats = [
    {
      label: 'Testing Accuracy',
      value: data ? `${testAccDisplay}%` : '58.54%',
      icon: Gauge,
      sub: `Training: ${trainAccDisplay}%`,
      accent: 'accent-green',
    },
    {
      label: 'Training Records',
      value: data ? data.train_records?.toLocaleString() ?? '808,579' : '808,579',
      icon: Target,
      sub: '80% Stratified Split',
    },
    {
      label: 'Testing Records',
      value: data ? data.test_records?.toLocaleString() ?? '202,145' : '202,145',
      icon: Crosshair,
      sub: '20% Test Evaluation',
      accent: 'accent-purple',
    },
    {
      label: 'Target Ratio',
      value: data?.target_ratio ?? '16.94% Click',
      icon: Scale,
      sub: 'Click / No Click',
    },
  ]

  const scaleData = data?.scaling_comparison || {}
  const scaleRows = Object.entries(scaleData).map(([method, m]) => ({ method, ...m }))

  const reportRows = data?.classification_report
    ? Object.entries(data.classification_report).map(([cls, v]) => ({ cls, ...v }))
    : []

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 06</div>
        <h1><span className="gradient-text">Logistic Regression</span></h1>
        <p className="subtitle">
          Binary classification predicting click vs no-click using preprocessed features,
          assessment scores, and regularization. Trained with balanced class weights.
        </p>
      </div>

      <div className="page-main page-enter">

        {/* Regularization selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <label style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Regularization Penalty:
          </label>
          <select
            value={activePenalty}
            onChange={(e) => setActivePenalty(e.target.value)}
            style={{
              padding: '0.4rem 0.8rem',
              borderRadius: 6,
              border: '1px solid var(--border)',
              background: 'var(--surface)',
              color: 'var(--text)',
              fontSize: '0.875rem',
              cursor: 'pointer',
            }}
          >
            {PENALTY_LABELS.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
            Active Penalty: {activePenalty.split(' ')[0]}
          </span>
        </div>

        {/* Stat cards */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">01</span>Performance</h2>
          <div className="stats-grid">
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

        {/* Feature scaling comparison */}
        {scaleRows.length > 0 && (
          <section className="section">
            <h2 className="section-title"><span className="section-badge">02</span>Feature Scaling Method Comparison</h2>
            <p className="section-desc">
              Evaluating how different feature scaling strategies impact logistic classification accuracy and discrimination.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'start' }}>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr><th>Scaling Method</th><th>ROC-AUC</th><th>Log-Loss</th></tr>
                  </thead>
                  <tbody>
                    {scaleRows.map((r) => (
                      <tr key={r.method} className={r.method === 'standard' || r.method === 'StandardScaler' ? 'highlight' : ''}>
                        <td><code>{r.method}</code></td>
                        <td>{r.roc_auc ? r.roc_auc.toFixed(4) : (r.train_accuracy ? `${r.train_accuracy}%` : '—')}</td>
                        <td>{r.log_loss ? r.log_loss.toFixed(4) : (r.test_accuracy ? `${r.test_accuracy}%` : '—')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="callout blue" style={{ marginTop: '1rem' }}>
                  <span className="callout-title">Key Insight: </span>
                  StandardScaler standardizes features to zero mean and unit variance,
                  preventing high-magnitude features from dominating gradient updates
                  and leading to the most stable convergence.
                </div>
              </div>
              <FigureCard
                src="logreg_04_scaling_comparison.png"
                title="Scaling Accuracy Comparison"
                purpose="Training vs testing metrics for each scaling method."
                observation="All methods produce comparable ROC-AUC (≈ 0.6468) and Log-Loss (≈ 0.6566) across engineered features."
              />
            </div>
          </section>
        )}

        {/* Classification report */}
        {reportRows.length > 0 && (
          <section className="section">
            <h2 className="section-title"><span className="section-badge">03</span>Evaluation Metrics & Classification Report</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'start' }}>
              <FigureCard
                src="logreg_02_confusion_matrix.png"
                title="Confusion Matrix"
                purpose="True/false positives and negatives at the 0.5 threshold."
                observation="Balanced class weights push recall up for the click class — more clicks are captured at the cost of some false positives."
              />
              <div>
                <div className="section-desc" style={{ marginBottom: '0.75rem' }}>
                  <strong>Detailed Classification Report</strong>
                </div>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>
                    </thead>
                    <tbody>
                      {reportRows.map((r) => (
                        <tr key={r.cls}>
                          <td><code>{r.cls}</code></td>
                          <td>{r.precision.toFixed(4)}</td>
                          <td>{r.recall.toFixed(4)}</td>
                          <td>{r.f1_score.toFixed(4)}</td>
                          <td>{r.support.toLocaleString()}</td>
                        </tr>
                      ))}
                      <tr className="highlight">
                        <td><strong>Test Accuracy</strong></td>
                        <td colSpan={4}><strong>{testAccDisplay}%</strong></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ROC + probability */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">04</span>Evaluation Figures</h2>
          <div className="figure-grid">
            <FigureCard
              src="logreg_01_roc_curve.png"
              title="ROC curve"
              purpose="True positive rate vs false positive rate across thresholds."
              observation="AUC = 0.6468 sits well above the random diagonal — the model ranks clicks above non-clicks consistently."
            />
            <FigureCard
              src="logreg_03_probability_distribution.png"
              title="Predicted probability by true class"
              purpose="Separation between the click and no-click probability distributions."
              observation="The two distributions overlap heavily — expected for CTR data — but the click class is shifted right, which is what drives the AUC."
              fullWidth
            />
          </div>
        </section>

        <div className="callout green">
          <span className="callout-title">Result: </span>
          Logistic regression matches the OLS baseline on ROC-AUC (≈ 0.6468) but produces
          <strong> properly calibrated probabilities</strong> — the key requirement for CTR
          ranking and bidding. Balanced class weights trade precision for recall, lifting F1
          to ≈ 0.35 versus ≈ 0.00 for the unweighted baseline.
        </div>
      </div>
    </>
  )
}