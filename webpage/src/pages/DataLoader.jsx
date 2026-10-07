import { Database, Rows3, Columns3, MousePointerClick, HardDrive } from 'lucide-react'
import useJson from '../hooks/useJson'

export default function DataLoader() {
  const { data } = useJson('01_data_loading_summary.json')

  const stats = [
    { label: 'Impressions', value: data ? data.rows.toLocaleString() : '—', icon: Rows3, sub: 'sampled rows' },
    { label: 'Features', value: data ? data.columns : '—', icon: Columns3, sub: '24 raw columns' },
    { label: 'Click Rate', value: data ? `${(data.click_rate * 100).toFixed(2)}%` : '—', icon: MousePointerClick, sub: `${data ? data.positive_clicks.toLocaleString() : '—'} clicks` },
    { label: 'Memory', value: data ? `${data.memory_mb} MB` : '—', icon: HardDrive, sub: 'in-memory footprint' },
  ]

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 02</div>
        <h1><span className="gradient-text">Data Loading</span></h1>
        <p className="subtitle">
          The Avazu CTR dataset is loaded from <code>Data/train.gz</code> (~40.4M impressions).
          The pipeline samples ~1M rows so every stage runs quickly while staying statistically sound.
        </p>
      </div>

      <div className="page-main page-enter">
        <section className="section">
          <h2 className="section-title"><span className="section-badge">01</span>Dataset Overview</h2>
          <p className="section-desc">
            The sample preserves the full 24-column schema of the original Avazu dataset —
            anonymised ad-impression context with a binary <code>click</code> target.
          </p>
          <div className="stats-grid">
            {stats.map((s) => {
              const Icon = s.icon
              return (
                <div key={s.label} className="stat-card">
                  <div className="stat-label"><Icon size={14} style={{ verticalAlign: -2, marginRight: 6 }} />{s.label}</div>
                  <div className="stat-value">{s.value}</div>
                  <div className="stat-sub">{s.sub}</div>
                </div>
              )
            })}
          </div>
        </section>

        <section className="section">
          <h2 className="section-title"><span className="section-badge">02</span>Column Schema</h2>
          <p className="section-desc">
            The dataset mixes numeric anonymised features (<code>C1</code>, <code>C14</code>–<code>C21</code>)
            with high-cardinality identifiers (<code>site_id</code>, <code>device_id</code>, …).
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr><th>Column</th><th>Type</th><th>Description</th></tr>
              </thead>
              <tbody>
                <tr><td><code>id</code></td><td>uint64</td><td>Unique ad impression identifier</td></tr>
                <tr className="highlight"><td><code>click</code></td><td>int64</td><td>Target — 1 if the ad was clicked, 0 otherwise</td></tr>
                <tr><td><code>hour</code></td><td>int64</td><td>Impression time, format YYMMDDHH (e.g. 14102100)</td></tr>
                <tr><td><code>C1</code>, <code>C14</code>–<code>C21</code></td><td>int64</td><td>Anonymised categorical features</td></tr>
                <tr><td><code>banner_pos</code></td><td>int64</td><td>Banner position on the page</td></tr>
                <tr><td><code>site_*</code></td><td>str</td><td>Publisher site id / domain / category</td></tr>
                <tr><td><code>app_*</code></td><td>str</td><td>Mobile app id / domain / category</td></tr>
                <tr><td><code>device_*</code></td><td>str / int</td><td>Device id / ip / model / type / connection type</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section className="section">
          <h2 className="section-title"><span className="section-badge">03</span>First Rows</h2>
          <p className="section-desc">A preview of the first impressions in the sample.</p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>click</th><th>hour</th><th>C1</th><th>banner_pos</th>
                  <th>site_id</th><th>site_category</th><th>app_category</th>
                  <th>device_type</th><th>device_conn_type</th><th>C14</th><th>C21</th>
                </tr>
              </thead>
              <tbody>
                {[
                  [1, 14102100, 1005, 1, 'd9750ee7', 'f028772b', '07d7df22', 1, 0, 17614, 33],
                  [1, 14102101, 1002, 0, 'ea81afe8', '50e219e0', '07d7df22', 0, 0, 14198, 23],
                  [0, 14102101, 1005, 0, '1fbe01fe', '28905ebd', '07d7df22', 1, 0, 15706, 79],
                  [0, 14102101, 1005, 1, '93de26ae', '335d28a8', '07d7df22', 1, 0, 20596, 157],
                  [1, 14102102, 1002, 0, 'c54454a2', '50e219e0', '07d7df22', 0, 0, 15698, 79],
                ].map((r, i) => (
                  <tr key={i}>
                    {r.map((v, j) => <td key={j}>{j === 0 ? (v ? '✓' : '✗') : v}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="section">
          <div className="callout blue">
            <span className="callout-title">Note: </span>
            The full dataset has <strong>~40.4M rows</strong> and no missing values.
            A deterministic 2.5% sample (~1,010,724 rows) is used for the analysis
            so the pipeline completes in minutes on a laptop.
          </div>
        </section>
      </div>
    </>
  )
}