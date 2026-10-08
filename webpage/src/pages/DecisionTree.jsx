import { useState } from 'react'
import { GitBranch, Gauge, Target, Crosshair, Layers } from 'lucide-react'
import useJson from '../hooks/useJson'
import FigureCard from '../components/FigureCard'

export default function DecisionTree() {
  const { data } = useJson('07_decision_tree_summary.json')
  const [selectedDepth, setSelectedDepth] = useState(null)

  const sweep = data?.depth_tuning || data?.depth_sweep || []
  const activeDepth = selectedDepth ?? data?.best_depth ?? data?.max_depth ?? 5
  const activeRow = sweep.find((r) => r.depth === activeDepth) || {}

  const stats = [
    {
      label: 'Testing Accuracy',
      value: data ? `${data.test_accuracy}%` : '63.88%',
      sub: `${data ? data.test_records?.toLocaleString() : '202,145'} test records`,
      icon: Gauge,
      accent: 'accent-green',
    },
    {
      label: 'Training Accuracy',
      value: data ? `${data.train_accuracy}%` : '64.21%',
      sub: `${data ? data.train_records?.toLocaleString() : '808,579'} training records`,
      icon: Target,
    },
    {
      label: 'ROC-AUC',
      value: data ? data.roc_auc.toFixed(4) : '0.6681',
      sub: 'discrimination power',
      icon: Crosshair,
      accent: 'accent-purple',
    },
    {
      label: 'Tree Nodes',
      value: data ? data.n_nodes ?? '—' : '—',
      sub: `${data?.n_leaves ?? '—'} leaves, depth ${data?.best_depth ?? data?.max_depth ?? 5}`,
      icon: Layers,
    },
  ]

  const reportRows = data?.classification_report
    ? Object.entries(data.classification_report).map(([cls, v]) => ({ cls, ...v }))
    : []

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 08</div>
        <h1><span className="gradient-text">Decision Tree</span></h1>
        <p className="subtitle">
          A <code>DecisionTreeClassifier</code> with balanced class weights trained on
          the engineered features. The depth sweep reveals the bias-variance trade-off.
        </p>
      </div>

      <div className="page-main page-enter">

        {/* Stat cards */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">01</span>Performance Overview</h2>
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

        {/* Classification report */}
        {reportRows.length > 0 && (
          <section className="section">
            <h2 className="section-title"><span className="section-badge">02</span>Classification Report (Depth = {data?.best_depth ?? data?.max_depth ?? 5})</h2>
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
                    <td colSpan={4}><strong>{data?.test_accuracy}%</strong></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Depth selector */}
        {sweep.length > 0 && (
          <section className="section">
            <h2 className="section-title"><span className="section-badge">03</span>Depth Sweep</h2>
            <p className="section-desc">
              Select a depth to see its metrics. The optimal model uses <strong>depth = {data?.best_depth ?? data?.max_depth ?? 5}</strong>.
            </p>

            {/* Tab selector */}
            <div className="model-tabs" style={{ marginBottom: '1rem' }}>
              {sweep.map((r) => (
                <button
                  key={r.depth}
                  className={`model-tab ${activeDepth === r.depth ? 'active' : ''}`}
                  onClick={() => setSelectedDepth(r.depth)}
                >
                  depth {r.depth}
                </button>
              ))}
            </div>

            {/* Active depth metrics */}
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr><th>Depth</th><th>Train Accuracy</th><th>Test Accuracy</th><th>ROC-AUC</th><th>F1</th></tr>
                </thead>
                <tbody>
                  {sweep.map((r) => (
                    <tr key={r.depth} className={r.depth === activeDepth ? 'highlight' : ''}>
                      <td>{r.depth}</td>
                      <td>{r.train_accuracy}%</td>
                      <td>{r.test_accuracy}%</td>
                      <td>{r.roc_auc.toFixed(4)}</td>
                      <td>{r.f1.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="figure-grid mt-3">
              <FigureCard
                src="dt_04_depth_vs_auc.png"
                title="Depth sweep — AUC and Accuracy"
                purpose="ROC-AUC and accuracy as a function of max_depth (1–10)."
                observation={`AUC balances around depth ${data?.best_depth ?? data?.max_depth ?? 5}; deeper trees risk memorizing training noise.`}
                fullWidth
              />
            </div>
          </section>
        )}

        {/* Tree visualisation */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">04</span>Model Diagnostics</h2>
          <div className="figure-grid">
            <FigureCard
              src="dt_01_tree_visualization.png"
              title="Tree structure (top branches)"
              purpose="First levels of the decision tree — shows the most important split decisions."
              observation="The root split is on the most discriminative feature (C18_enc); subsequent splits handle site and app context."
              fullWidth
            />
            <FigureCard
              src="dt_02_feature_importance.png"
              title="Top feature importances"
              purpose="Gini-based feature importances — which features the tree uses most."
              observation="Feature C18_enc accounts for over 50% of total split importance, followed by site_id_freq and app_id_freq."
            />
            <FigureCard
              src="dt_03_confusion_matrix.png"
              title={`Confusion matrix (depth=${data?.best_depth ?? data?.max_depth ?? 5})`}
              purpose="True/false positives and negatives at the 0.5 threshold."
              observation="Balanced class weights push the tree to recall more clicks at the cost of precision."
            />
          </div>

          {/* Predictive features chip list */}
          {data?.top_features && (
            <div style={{ marginTop: '1.5rem' }}>
              <div className="section-desc" style={{ marginBottom: '0.75rem' }}>
                <strong>Predictive features ({Object.keys(data.top_features).length})</strong>
              </div>
              <div className="chip-row">
                {Object.keys(data.top_features).map((f) => (
                  <span key={f} className="chip">{f}</span>
                ))}
              </div>
            </div>
          )}
        </section>
      </div>
    </>
  )
}
