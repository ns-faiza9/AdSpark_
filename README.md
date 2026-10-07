# AdSpark — CTR Prediction (Flask Web Application)

Machine Learning project that predicts ad Click-Through Rate (CTR) on the 1.01M sample Avazu dataset, served via a full **Python Flask Web Application**.

---

## Project Structure

| Folder / File | Purpose |
| ------------- | ------- |
| `main.py` | Entry point launcher to start the Flask web app server |
| `app.py` | Flask application defining all 10 page routes, static file handlers & API endpoints |
| `ctr_predictor.py` | Real-time CTR inference calculation engine |
| `templates/` | Jinja2 HTML templates for all dashboard sections, model stages & CTR Predictor Sandbox |
| `static/` | CSS styles (`style.css`), JavaScript engine (`main.js`), and assets |
| `analysis/` | Python data-science ML pipeline (EDA, feature engineering, models, figures, JSON summaries) |
| `analysis/output/` | Generated JSON summaries and output PNG figures served dynamically by Flask |
| `Data/` | Raw and processed Avazu ad impression datasets |

---

## Quick Start — How to Run

### Prerequisites

Make sure Python (>= 3.10) and Flask are installed:

```bash
pip install -r analysis/requirements.txt
```

### Starting the Flask Web Application

Run the application directly using `main.py` or `app.py`:

```bash
python main.py
```

Or using Flask CLI:

```bash
flask run
```

Then open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## Application Features & Navigation

- **00 Dashboard** (`/`) — Executive overview, key metrics, model comparison leaderboard (OLS, Ridge, Lasso, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, LightGBM, XGBoost).
- **01 Data Loading** (`/data-loading`) — Raw dataset inspection (1.01M rows, 24 features, memory optimization to 639.8 MB).
- **02 Exploratory EDA** (`/eda`) — Click rate breakdowns by hour of day, day of week, banner position, site category, device connection type, and correlation matrices.
- **03 Feature Engineering** (`/feature-engineering`) — Frequency encodings for 561k IP addresses and 151k device IDs, standard scaling, and temporal feature extraction.
- **04 Linear Regression** (`/linear-regression`) — Continuous baseline regression metrics, OLS, Ridge, Lasso coefficients, and residual distributions.
- **05 Logistic Regression** (`/logistic-regression`) — Binary classification with Sigmoid mapping, scaling comparison, log-loss, and ROC curve analysis.
- **06 Regularization** (`/regularization`) — L1 (Lasso) vs L2 (Ridge) penalty paths and hyperparameter tuning across regularizer strength \(C\).
- **07 Decision Tree** (`/decision-tree`) — Tree depth vs AUC tuning, tree visualization, Gini impurity feature importances.
- **08 Ensemble Methods** (`/ensemble`) — Benchmarks across Random Forest, AdaBoost, Gradient Boosting, LightGBM, and XGBoost (Best Model: **XGBoost** with ROC-AUC **0.7397**).
- **✨ CTR Predictor Sandbox** (`/predict`) — Interactive real-time ad CTR predictor tool with live inference calculation via `/api/predict`.