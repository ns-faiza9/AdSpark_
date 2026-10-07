import { SlidersHorizontal, Gauge, Scale, Filter } from 'lucide-react'
import useJson from '../hooks/useJson'
import FigureCard from '../components/FigureCard'

export default function Regularization() {
  const { data } = useJson('06_regularization_summary.json')

  const regView = data?.regression_view || {}
  const clsView = data?.classification_view || {}

  const regRows = Object.entries(regView).map(([name, m]) => ({
    name, rmse: m.rmse, nonzero: m.n_nonzero_coefs,
  }))
  const clsRows = Object.entries(clsView).map(([name, m]) => ({
    name, auc: m.roc_auc, logLoss: m.log_loss, f1: m.f1, nonzero: m.n_nonzero_coefs,
  }))

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 07</div>
        <h1><span className="gradient-text">Regularization</span></h1>
        <p className="subtitle">
          Lasso (L1), Ridge (L2) and Elastic Net penalties applied to both the linear
          regression and logistic regression views of the problem.
        </p>
      </div>

      <div className="page-main page-enter">
        <section className="section">
          <h2 className="section-title"><span className="section-badge">01</span>Regression View — RMSE</h2>
          <p className="section-desc">
            Regularised linear models on the engineered features. Lasso and Elastic Net
            shrink some coefficients to exactly zero (feature selection).
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr><th>Model</th><th>RMSE</th><th>Non-zero coefficients</th></tr>
              </thead>
              <tbody>
                {regRows.map((r) => (
                  <tr key={r.name}>
                    <td><code>{r.name}</code></td>
                    <td>{r.rmse.toFixed(4)}</td>
                    <td>{r.nonzero} / {data?.total_features ?? 23}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="figure-grid mt-3">
            <FigureCard
              src="reg_02_rmse_comparison.png"
              title="RMSE comparison"
              purpose="All three penalties achieve nearly identical RMSE ≈ 0.367."
              observation="Ridge and Elastic Net are statistically indistinguishable here; Lasso removes 4 features at no cost in error — a sparser, simpler model."
            />
            <FigureCard
              src="reg_01_lasso_paths.png"
              title="Lasso coefficient paths"
              purpose="How each coefficient shrinks toward zero as the penalty α increases."
              observation="As α grows, coefficients collapse to zero one by one — the mechanism behind Lasso's automatic feature selection."
              fullWidth
            />
          </div>
        </section>

        <section className="section">
          <h2 className="section-title"><span className="section-badge">02</span>Classification View — Regularised Logistic Regression</h2>
          <p className="section-desc">
            L1, L2 and Elastic Net penalties on the logistic regression model.
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr><th>Model</th><th>ROC-AUC</th><th>Log Loss</th><th>F1</th><th>Non-zero coefficients</th></tr>
              </thead>
              <tbody>
                {clsRows.map((r) => (
                  <tr key={r.name}>
                    <td><code>{r.name}</code></td>
                    <td>{r.auc.toFixed(4)}</td>
                    <td>{r.logLoss.toFixed(4)}</td>
                    <td>{r.f1.toFixed(4)}</td>
                    <td>{r.nonzero} / {data?.total_features ?? 23}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="figure-grid mt-3">
            <FigureCard
              src="reg_03_auc_comparison.png"
              title="ROC-AUC comparison"
              purpose="Regularised logistic regression variants at C = 1.0."
              observation="All penalties converge to AUC ≈ 0.647 — with 23 features the regularisation strength is mild, so the models agree. Tuning C would show larger differences."
            />
          </div>
        </section>

        <section className="section">
          <div className="callout purple">
            <span className="callout-title">Conclusion: </span>
            Regularization keeps performance stable while shrinking the model —
            Lasso removes 4 of 23 features with no measurable loss in RMSE or AUC.
            For production, an Elastic Net with tuned <code>α</code> and <code>l1_ratio</code>
            would offer the best sparsity-vs-accuracy trade-off.
          </div>
        </section>
      </div>
    </>
  )
}