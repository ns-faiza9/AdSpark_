import { useState } from 'react'
import { Zap, Gauge, Target, Crosshair, Layers } from 'lucide-react'
import useJson from '../hooks/useJson'
import FigureCard from '../components/FigureCard'

const MODEL_LABELS = {
  'Random Forest': 'Bagging — Random Forest',
  'AdaBoost': 'AdaBoost',
  'Gradient Boosting': 'Gradient Boosting',
  'LightGBM': 'LightGBM',
  'XGBoost': 'XGBoost',
}

const ALGO_COLORS = {
  'Bagging': 'blue',
  'Boosting': 'green',
}

export default function Ensemble() {
  const { data } = useJson('08_ensemble_summary.json')
  const [activeModel, setActiveModel] = useState('Random Forest')

  const models = data?.models || {}
  const modelNames = Object.keys(models)
  const m = models[activeModel] || {}
  const bestModel = data?.best_model

  const stats = [
    {
      label: 'Testing Accuracy',
      value: m.test_accuracy ? `${m.test_accuracy}%` : '—',
      sub: `${m.test_records ? m.test_records.toLocaleString() : '—'} test records`,
      icon: Gauge,
      accent: 'accent-green',
    },
    {
      label: 'Training Accuracy',
      value: m.train_accuracy ? `${m.train_accuracy}%` : '—',
      sub: `${m.train_records ? m.train_records.toLocaleString() : '—'} training records`,
      icon: Target,
    },
    {
      label: 'Algorithm Family',
      value: m.algorithm_family || '—',
      sub: activeModel,
      icon: Zap,
      accent: 'accent-purple',
    },
    {
      label: 'Feature Count',
      value: m.n_features || '—',
      sub: 'Engineered predictors',
      icon: Layers,
    },
  ]

  const reportRows = m.classification_report
    ? Object.entries(m.classification_report).map(([cls, v]) => ({ cls, ...v }))
    : []

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 09</div>
        <h1><span className="gradient-text">Ensemble Learning</span></h1>
        <p className="subtitle">
          Evaluating tree-based and ensemble algorithms for CTR prediction.
          Comparing <strong>Bagging — Random Forest</strong>, <strong>AdaBoost</strong>,
          <strong>Gradient Boosting</strong>, <strong>LightGBM</strong>, and <strong>XGBoost</strong>.
        </p>
      </div>

      <div className="page-main page-enter">

        {/* Model selector tabs */}
        <section className="section">
          <div className="model-tabs" style={{ marginBottom: '1.5rem' }}>
            {modelNames.map((name) => {
              const label = MODEL_LABELS[name] || name
              return (
                <button
                  key={name}
                  className={`model-tab ${activeModel === name ? 'active' : ''}`}
                  onClick={() => setActiveModel(name)}
                >
                  {name === bestModel ? `★ ${label}` : label}
                </button>
              )
            })}
          </div>

          {/* Sub-label */}
          {m.algorithm_family && (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
              <em>{activeModel}</em> — <strong>{m.algorithm_family}</strong> ensemble
              {bestModel === activeModel && <span style={{ marginLeft: 8, color: '#10b981' }}>✓ Best model</span>}
            </p>
          )}

          {/* Stat cards */}
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

        {/* Classification report + confusion matrix */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">02</span>Evaluation Metrics</h2>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'start' }}>
            {/* Confusion matrix figure */}
            <FigureCard
              src="ens_05_confusion_matrix_best.png"
              title={`Confusion Matrix — ${bestModel ?? 'Best Model'}`}
              purpose="True/false positives and negatives for the best-performing model."
              observation="Balanced weights push recall up at the cost of precision — appropriate for click ranking."
            />

            {/* Classification report table */}
            <div>
              <div className="section-desc" style={{ marginBottom: '0.75rem' }}>
                <strong>Detailed Classification Report</strong>
              </div>
              {reportRows.length > 0 ? (
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
                        <td colSpan={4}><strong>{m.test_accuracy}%</strong></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>Loading…</p>
              )}

              {/* Model metrics summary */}
              <div className="callout blue" style={{ marginTop: '1rem' }}>
                <span className="callout-title">ROC-AUC: </span>
                <strong>{m.roc_auc?.toFixed(4) ?? '—'}</strong>
                {'  ·  '}
                <span className="callout-title">F1: </span>
                <strong>{m.f1?.toFixed(4) ?? '—'}</strong>
                {'  ·  '}
                <span className="callout-title">Precision / Recall: </span>
                <strong>{m.precision?.toFixed(3) ?? '—'} / {m.recall?.toFixed(3) ?? '—'}</strong>
              </div>
            </div>
          </div>
        </section>

        {/* Comparison figures */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">03</span>Model Comparison</h2>
          <div className="figure-grid">
            <FigureCard
              src="ens_01_roc_comparison.png"
              title="ROC curves — all models"
              purpose="True positive rate vs false positive rate for all three ensemble models."
              observation="All models outperform logistic regression — ensemble methods capture non-linear interactions the linear model misses."
            />
            <FigureCard
              src="ens_02_auc_comparison.png"
              title="AUC comparison"
              purpose="Side-by-side AUC for Random Forest, Gradient Boosting and AdaBoost."
              observation="Gradient Boosting typically edges out the others due to its additive correction approach."
            />
            <FigureCard
              src="ens_03_accuracy_comparison.png"
              title="Accuracy comparison (train vs test)"
              purpose="Training and testing accuracy for each model."
              observation="Random Forest shows the largest train-test gap — a sign of variance; boosting methods are more stable."
              fullWidth
            />
            <FigureCard
              src="ens_04_feature_importance_rf.png"
              title="Random Forest — feature importances (top 20)"
              purpose="Mean impurity decrease per feature across the 200 trees."
              observation="Frequency-encoded identifiers and the anonymous C-features dominate; banner position and hour_of_day are strong time-context signals."
              fullWidth
            />
          </div>
        </section>

        {/* Predictive features */}
        {m.top_features && (
          <section className="section">
            <h2 className="section-title"><span className="section-badge">04</span>Predictive Features ({m.n_features})</h2>
            <div className="chip-row">
              {Object.keys(m.top_features).map((f) => (
                <span key={f} className="chip">{f}</span>
              ))}
            </div>
          </section>
        )}
      </div>
    </>
  )
}
