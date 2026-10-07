import { Filter, Layers, Hash, Clock } from 'lucide-react'
import useJson from '../hooks/useJson'
import FigureCard from '../components/FigureCard'

export default function FeatureEngineering() {
  const { data } = useJson('03_feature_engineering_summary.json')

  const stats = [
    { label: 'Model Features', value: data ? data.n_features : '—', icon: Layers, sub: 'after engineering' },
    { label: 'Frequency Encoded', value: data ? data.high_cardinality_encoded.length : '—', icon: Hash, sub: 'high-cardinality cols' },
    { label: 'Label Encoded', value: data ? data.low_cardinality_encoded.length : '—', icon: Filter, sub: 'low-cardinality cols' },
    { label: 'Time Features', value: 2, icon: Clock, sub: 'hour_of_day, day_of_week' },
  ]

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 04</div>
        <h1><span className="gradient-text">Preprocessing</span></h1>
        <p className="subtitle">
          Missing-value strategy and feature transforms: the raw 24 columns become a
          compact, model-ready feature matrix.
        </p>
      </div>

      <div className="page-main page-enter">
        <section className="section">
          <h2 className="section-title"><span className="section-badge">01</span>Transformations Applied</h2>
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
          <h2 className="section-title"><span className="section-badge">02</span>Strategy</h2>
          <div className="callout blue">
            <span className="callout-title">Missing values: </span>
            The Avazu dataset contains no explicit NaN values, but <code>C20</code> uses
            <code> -1</code> as a sentinel for "unknown". These are kept as a distinct
            category — the label encoder treats them like any other value.
          </div>
          <div className="callout green">
            <span className="callout-title">High-cardinality columns: </span>
            <code>site_id</code>, <code>app_id</code>, <code>device_id</code>, <code>device_ip</code>,
            <code> device_model</code>, <code>site_domain</code> and <code>app_domain</code> have up to
            hundreds of thousands of unique values. Each is replaced by its <strong>frequency</strong> in
            the training sample — a compact, information-rich encoding.
          </div>
          <div className="callout purple">
            <span className="callout-title">Low-cardinality columns: </span>
            <code>C1</code>, <code>banner_pos</code>, <code>site_category</code>, <code>app_category</code>,
            <code> device_type</code>, <code>device_conn_type</code> and <code>C14</code>–<code>C21</code>
            are mapped to integer codes via label encoding.
          </div>
          <div className="callout red">
            <span className="callout-title">Dropped: </span>
            <code>id</code> (pure identifier), <code>hour</code> (replaced by time features) and the raw
            <code> device_id</code> / <code>device_ip</code> strings (replaced by their frequencies).
          </div>
        </section>

        <section className="section">
          <h2 className="section-title"><span className="section-badge">03</span>Engineered Features</h2>
          <div className="figure-grid">
            <FigureCard
              src="fe_01_frequency_encodings.png"
              title="Frequency-encoded feature distributions"
              purpose="Each high-cardinality column becomes a continuous frequency in [0, 1] — most values are rare."
              observation="The heavy-tailed distributions confirm that most identifiers appear only a handful of times; frequency encoding captures this without exploding dimensionality."
              fullWidth
            />
            <FigureCard
              src="fe_02_hour_of_day.png"
              title="Impressions by hour of day"
              purpose="Traffic volume peaks around midnight and drops in the early morning."
              observation="hour_of_day and day_of_week are kept as numeric features so models can exploit time-of-day effects."
            />
          </div>
        </section>
      </div>
    </>
  )
}