import { BarChart3, MousePointerClick, Layers, Hash, Copy, Clock, Filter, Grid } from 'lucide-react'
import useJson from '../hooks/useJson'
import FigureCard from '../components/FigureCard'

export default function EDA() {
  const { data } = useJson('02_eda_summary.json')

  const tasks = [
    { num: '01', title: 'Task 1: Load Data & Overview', desc: 'Inspect raw impression attributes, shape (1.01M x 24), and head.' },
    { num: '02', title: 'Task 2: Structure & Data Types', desc: 'Summary stats and data types across string IDs & numeric features.' },
    { num: '03', title: 'Task 3: Missing Values Analysis', desc: 'Missingness count, percentage, and imputation (0 missing found).' },
    { num: '04', title: 'Task 4: Duplicate Impressions Check', desc: 'Exact row duplicate check (0 duplicates found).' },
    { num: '05', title: 'Task 5: Target Distribution', desc: 'Class imbalance breakdown for click (16.94% baseline CTR).' },
    { num: '06', title: 'Task 6: Feature Distributions', desc: 'Histograms & KDE for hour_of_day, banner_pos, C14, C21.' },
    { num: '07', title: 'Task 7: Outlier Detection (Boxplots)', desc: 'Boxplots for banner_pos, C14, C17, C21 IQR spread.' },
    { num: '08', title: 'Task 8: Correlation Analysis', desc: 'Pearson correlation matrix and feature ranking with click target.' },
    { num: '09', title: 'Task 9: Interaction Scatter Plots', desc: 'Scatter interaction pairs hue-coded by click outcome.' },
    { num: '10', title: 'Task 10: Categorical Counts', desc: 'Frequency count bar plots for site_category, app_category, C15, C16.' },
    { num: '11', title: 'Task 11: Hourly CTR Analysis', desc: '24-hour Click-Through Rate fluctuation line chart.' },
    { num: '12', title: 'Task 12: Ad Context vs CTR', desc: 'CTR % breakdown across banner positions & connection types.' },
    { num: '13', title: 'Task 13: Temporal Multi-Day Trends', desc: 'Daily/hourly CTR trend line charts segmented by connection type.' },
    { num: '14', title: 'Task 14: Category Performance', desc: 'Impression volume vs CTR conversion rate dual bar charts.' },
    { num: '15', title: 'Task 15: Pairwise Feature Plot', desc: 'sns.pairplot matrix across key numerical features hue-coded by click.' },
  ]

  return (
    <>
      <div className="page-header">
        <div className="kicker">Stage 02 • Course Outcome CO2</div>
        <h1><span className="gradient-text">15-Task Exploratory Data Analysis</span></h1>
        <p class="subtitle">
          Standard 15-Task Exploratory Data Analysis specifically executed on the Avazu CTR dataset schema —
          covering uni-variate distributions, bi-variate relationships, multi-variate pairplots, and temporal click trends.
        </p>
      </div>

      <div className="page-main page-enter">
        {/* Task Index Overview */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">INDEX</span>15 Standard EDA Tasks</h2>
          <div className="quick-access-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
            {tasks.map((t) => (
              <div key={t.num} className="quick-card">
                <span className="card-index">{t.num}</span>
                <h3>{t.title}</h3>
                <p>{t.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Section: Task 1 - 5 */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">01</span>Tasks 1–5: Data Overview, Quality & Target</h2>
          <div className="figure-grid">
            <FigureCard
              src="eda_01_target_distribution.png"
              title="Task 5: Target Click Distribution"
              purpose="Uni-variate class distribution showing 16.94% positive click baseline."
              observation="Heavy class imbalance requires log-loss and ROC-AUC metrics instead of raw accuracy."
            />
            <FigureCard
              src="eda_02_missingness.png"
              title="Task 3: Missing Values Heatmap"
              purpose="Checking missingness count and percentage across all 24 attributes."
              observation="Clean dataset with zero missing values across all 1.01M impression records."
            />
            <FigureCard
              src="eda_10_numeric_distributions.png"
              title="Task 6: Feature Distributions"
              purpose="Histograms and KDE plots for hour_of_day, banner_pos, C14, and C21."
              observation="Extracted hour_of_day reveals peak impression volume during morning hours."
              fullWidth
            />
          </div>
        </section>

        {/* Section: Tasks 7 - 10 */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">02</span>Tasks 7–10: Outliers, Correlations & Interactions</h2>
          <div className="figure-grid">
            <FigureCard
              src="eda_13_outlier_boxplots.png"
              title="Task 7: Outlier Detection Boxplots"
              purpose="Boxplots for banner_pos, C14, C17, and C21 IQR spread."
              observation="C14 and C21 display significant long-tail value distributions."
              fullWidth
            />
            <FigureCard
              src="eda_11_correlation_heatmap.png"
              title="Task 8: Correlation Matrix"
              purpose="Pearson correlation heatmap across all numerical columns."
              observation="Linear correlations with click target are weak, confirming non-linear model requirement."
            />
            <FigureCard
              src="eda_15_correlation_with_target.png"
              title="Task 8b: Feature Correlation with Target"
              purpose="Ranked correlation values relative to target variable click."
              observation="C16 (+0.129) and C21 (+0.069) show top positive linear correlation with ad clicks."
            />
            <FigureCard
              src="eda_16_scatter_feature_click.png"
              title="Task 9: Interaction Scatter Plots"
              purpose="C14 vs C17 and banner_pos vs device_type hue-coded by click."
              observation="Non-linear clusters visible across categorical feature interaction pairs."
              fullWidth
            />
            <FigureCard
              src="eda_14_feature_count_plots.png"
              title="Task 10: Categorical Frequency Counts"
              purpose="Count bar plots for site_category, app_category, device_type, connection type."
              observation="Device type 1 (smartphones) and connection type 0 (cellular) dominate overall volume."
              fullWidth
            />
          </div>
        </section>

        {/* Section: Tasks 11 - 15 */}
        <section className="section">
          <h2 className="section-title"><span className="section-badge">03</span>Tasks 11–15: Context CTR, Temporal Trends & Pairplots</h2>
          <div className="figure-grid">
            <FigureCard
              src="eda_05_click_rate_hour.png"
              title="Task 11: Hourly CTR Analysis"
              purpose="24-hour Click-Through Rate line chart across hour_of_day."
              observation="Night hours (00:00 - 02:00) exhibit higher CTR (~18.8%) compared to afternoon drops (15.8%)."
              fullWidth
            />
            <FigureCard
              src="eda_04_click_rate_banner_pos.png"
              title="Task 12: Ad Context vs CTR"
              purpose="Grouped bar charts for banner position and device connection type."
              observation="Banner position 7 achieves peak 32.46% CTR, followed by position 1 (18.36%)."
            />
            <FigureCard
              src="eda_17_click_rate_dayofweek.png"
              title="Task 13: Multi-Day Temporal Trends"
              purpose="Daily CTR trends segmented by connection type."
              observation="Weekend engagement (Sunday 18.24%) exceeds mid-week Wednesday (15.60%)."
            />
            <FigureCard
              src="eda_08_click_rate_site_category.png"
              title="Task 14: Category Performance vs Volume"
              purpose="Impression volume vs CTR conversion rate dual axis chart."
              observation="High-volume categories stay near 16–18% CTR, while specialized categories reach 50%+."
              fullWidth
            />
            <FigureCard
              src="eda_18_pairplot.png"
              title="Task 15: Pairwise Feature Plot"
              purpose="sns.pairplot matrix across banner_pos, C1, C14, C17, C21 colored by click."
              observation="Heavy non-linear overlap confirms tree-based ensembles are needed for effective prediction."
              fullWidth
            />
          </div>
        </section>
      </div>
    </>
  )
}