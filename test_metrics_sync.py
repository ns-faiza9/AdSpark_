import json
import ml_engine
import app

def test_all_metrics():
    print("=== 1. Testing ml_engine.get_executive_summary_cards() ===")
    cards = ml_engine.get_executive_summary_cards()
    c1 = cards['card_1']
    c2 = cards['card_2']
    c3 = cards['card_3']

    print("Card 1:", c1)
    print("Card 2:", c2)
    print("Card 3:", c3)

    assert c1['best_model'] == 'XGBoost', f"Expected XGBoost, got {c1['best_model']}"
    assert c1['best_auc'] == 0.7397, f"Expected 0.7397, got {c1['best_auc']}"
    assert c1['total_models'] == 10, f"Expected 10, got {c1['total_models']}"
    assert c2['best_silhouette'] == 0.0653, f"Expected 0.0653, got {c2['best_silhouette']}"
    assert c2['dbscan_noise_pct'] == '70.8%', f"Expected 70.8%, got {c2['dbscan_noise_pct']}"
    assert '9 Components' in c2['pca_variance'], f"Expected 9 Components, got {c2['pca_variance']}"
    assert c3['stratified_cv_auc'] == 0.5181, f"Expected 0.5181, got {c3['stratified_cv_auc']}"
    assert c3['nested_cv_auc'] == 0.5061, f"Expected 0.5061, got {c3['nested_cv_auc']}"
    assert c3['pr_auc'] == 0.1749, f"Expected 0.1749, got {c3['pr_auc']}"

    print("\n=== 2. Testing ml_engine.get_full_leaderboard() ===")
    lb = ml_engine.get_full_leaderboard()
    print(f"Leaderboard has {len(lb)} models.")
    for m in lb:
        print(f"{m['name']:<30} | AUC: {m['roc_auc']:<6} | Loss: {m['log_loss']:<6} | Acc: {m['accuracy']}%")

    assert len(lb) == 10, f"Expected 10 models, got {len(lb)}"
    assert lb[0]['name'] == 'XGBoost Classifier'
    assert lb[0]['roc_auc'] == 0.7397
    assert lb[1]['name'] == 'LightGBM Classifier'
    assert lb[1]['roc_auc'] == 0.7391
    assert any(m['name'] == 'Gradient Boosting (GBM)' and m['roc_auc'] == 0.7287 for m in lb)
    assert any(m['name'] == 'Random Forest Classifier' and m['roc_auc'] == 0.7225 for m in lb)
    assert any(m['name'] == 'Decision Tree (Pruned)' and m['roc_auc'] == 0.6681 for m in lb)
    assert any(m['name'] == 'Logistic Regression (L2)' and m['roc_auc'] == 0.6468 for m in lb)
    assert any(m['name'] == 'Logistic Regression (L1)' and m['roc_auc'] == 0.6468 for m in lb)

    print("\n=== 3. Testing Flask App Template Rendering ===")
    client = app.app.test_client()

    res_dash = client.get('/')
    assert res_dash.status_code == 200
    html_dash = res_dash.get_data(as_text=True)
    assert '0.7397' in html_dash
    assert '0.0653' in html_dash
    assert '70.8%' in html_dash
    assert '9 Components' in html_dash
    assert '0.5181' in html_dash
    assert '0.5061' in html_dash
    assert '0.1749' in html_dash

    res_lin = client.get('/linear-regression')
    assert res_lin.status_code == 200
    html_lin = res_lin.get_data(as_text=True)
    assert '0.0422' in html_lin
    assert '0.3671' in html_lin

    res_log = client.get('/logistic-regression')
    assert res_log.status_code == 200
    html_log = res_log.get_data(as_text=True)
    assert '58.54%' in html_log
    assert '0.6468' in html_log
    assert '0.6566' in html_log

    res_dt = client.get('/decision-tree')
    assert res_dt.status_code == 200
    html_dt = res_dt.get_data(as_text=True)
    assert '0.6681' in html_dt
    assert '63.88%' in html_dt

    res_km = client.get('/kmeans')
    assert res_km.status_code == 200
    html_km = res_km.get_data(as_text=True)
    assert '0.0653' in html_km

    res_dr = client.get('/dimensionality')
    assert res_dr.status_code == 200
    html_dr = res_dr.get_data(as_text=True)
    assert '9' in html_dr

    res_val = client.get('/validation')
    assert res_val.status_code == 200
    html_val = res_val.get_data(as_text=True)
    assert '0.5181' in html_val
    assert '0.5061' in html_val

    res_imb = client.get('/imbalanced-metrics')
    assert res_imb.status_code == 200
    html_imb = res_imb.get_data(as_text=True)
    assert '0.1749' in html_imb

    print("\n>>> ALL 100% OF ASSERTIONS AND METRICS MATCH PERFECTLY! <<<")

if __name__ == "__main__":
    test_all_metrics()
