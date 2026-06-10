# predictive-no-show-fvl
Predictive model for estimating patient no-shows in internal medicine outpatient consultations at Fundación Valle del Lili, using machine learning techniques to optimize appointment management and improve healthcare efficiency.

## Backend inference

The API now prioritizes `outputs/catboost/catboost_final.joblib` as the active inference artifact because it is the best-performing model reported in the repository metrics. If that artifact is unavailable, the backend falls back to `outputs/stacking/stacking_final.joblib` or `backend/models/model.pkl`.

Optional overrides:
- `PREDICTIVE_MODEL_PATH`
- `PREDICTIVE_METRICS_PATH`

## CatBoost

### Entrenar
```bash
python src/train.py --config .\configs\training_catboost.yml
```

### Validar + graficas
```bash
python src/validate.py --model outputs/catboost/catboost_final.joblib --config configs/training_catboost.yml
```

Salidas por defecto:
- outputs/catboost/catboost_final.joblib
- outputs/catboost/catboost_final_metrics.json (val/test en training)
- outputs/catboost/catboost_final_test_metrics.json (test en validate)
- outputs/catboost/catboost_final_confusion_matrix.png
- outputs/catboost/catboost_final_roc_curve.png
- outputs/catboost/catboost_final_pr_curve.png
