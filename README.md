# predictive-no-show-fvl
Predictive model for estimating patient no-shows in internal medicine outpatient consultations at Fundación Valle del Lili, using machine learning techniques to optimize appointment management and improve healthcare efficiency.

## Backend inference

The API now prioritizes `outputs/stacking/stacking_final.joblib` as the active inference artifact because it is the best-performing ensemble reported in the repository metrics. If that artifact is unavailable, the backend falls back to `backend/models/model.pkl`.

Optional overrides:
- `PREDICTIVE_MODEL_PATH`
- `PREDICTIVE_METRICS_PATH`

## Stacking Final (LightGBM + XGBoost + CatBoost)

### Entrenar
```bash
python src/train.py --config configs/training_stack.yml
```

### Validar + graficas
```bash
python src/validate.py --model outputs/stacking_final.joblib --config configs/training_stack.yml
```

Salidas por defecto:
- outputs/stacking_final.joblib
- outputs/stacking_final_metrics.json (val/test en training)
- outputs/stacking_final_test_metrics.json (test en validate)
- outputs/stacking_final_confusion_matrix.png
- outputs/stacking_final_roc_curve.png
- outputs/stacking_final_pr_curve.png
