# Documentacion del Proyecto

## Objetivo
Este repositorio construye un modelo de machine learning para predecir inasistencias (No-Show) en consultas de medicina interna. El objetivo es ayudar a la gestion de citas mediante un **modelo principal de CatBoost** que optimiza la predicción de no-asistencias, con soporte para modelos alternativos (LightGBM, XGBoost y Stacking).

## Estructura completa del repositorio
- `configs/`
  - `training_catboost.yml`: configuracion para el modelo principal de CatBoost.
  - `training_lightgbm.yml`: configuracion para LightGBM (alternativo).
  - `training_xgboost.yml`: configuracion para XGBoost (alternativo).
  - `training_stack.yml`: configuracion para el modelo de stacking.
- `data/`
  - `docs/`: documentacion y notas del dataset.
  - `methodology CRISP-DM/`: fases de la metodologia.
  - `processed/df_limpio.csv`: dataset limpio usado para el entrenamiento.
  - `raw/`: datos originales.
- `models/`
  - `catboost/` (Principal)
    - `model.py`: define el modelo CatBoost.
    - `config.yml`: hiperparametros de CatBoost.
  - `lightgbm/`
    - `model.py`: define el modelo individual LightGBM.
    - `config.yml`: hiperparametros de LightGBM.
  - `xgboost/`
    - `model.py`: define el modelo individual XGBoost.
    - `config.yml`: hiperparametros de XGBoost.
  - `stacking/`
    - `model.py`: define el stacking con LightGBM + XGBoost + CatBoost.
    - `config.yml`: hiperparametros del stacking.
- `notebooks/`
  - `01_Exploración.ipynb`
  - `02_Limpieza.ipynb`
- `outputs/`
  - `catboost/`: artefactos y graficas de CatBoost (MODELO PRINCIPAL).
  - `lightgbm/`: artefactos y graficas de LightGBM.
  - `xgboost/`: artefactos y graficas de XGBoost.
  - `stacking/`: artefactos y graficas del stacking.
- `src/`
  - `train.py`: entrena el modelo final y guarda artefactos.
  - `validate.py`: valida en test y genera graficas.
  - `preprocess.py`: utilidades de preprocesamiento.
  - `utils/data_loader.py`: carga y prepara datos.
- `documentation.md`: documentacion del flujo y ejecucion.
- `README.md`: resumen del proyecto.
- `requirements.txt`: dependencias.

## Ejecucion

### Modelo Principal: CatBoost

#### 1) Entrenar el modelo final de CatBoost
```bash
python src/train.py --config configs/training_catboost.yml
```

Genera:
- `outputs/catboost/catboost_final.joblib`
- `outputs/catboost/catboost_final_metrics.json`

#### 2) Validar y generar graficas
```bash
python src/validate.py --model outputs/catboost/catboost_final.joblib --config configs/training_catboost.yml --output outputs/catboost/catboost_final_test_metrics.json --plots-dir outputs/catboost --prefix catboost_final
```

Genera:
- `outputs/catboost/catboost_final_test_metrics.json`
- `outputs/catboost/catboost_final_confusion_matrix.png`
- `outputs/catboost/catboost_final_roc_curve.png`
- `outputs/catboost/catboost_final_pr_curve.png`

### Modelos Alternativos (LightGBM, XGBoost)

#### LightGBM (Alternativo)
Entrenar:
```bash
python src/train.py --config configs/training_lightgbm.yml
```
Validar:
```bash
python src/validate.py --model outputs/lightgbm/lightgbm_final.joblib --config configs/training_lightgbm.yml --output outputs/lightgbm/lightgbm_final_test_metrics.json --plots-dir outputs/lightgbm --prefix lightgbm_final
```

#### XGBoost (Alternativo)
Entrenar:
```bash
python src/train.py --config configs/training_xgboost.yml
```
Validar:
```bash
python src/validate.py --model outputs/xgboost/xgboost_final.joblib --config configs/training_xgboost.yml --output outputs/xgboost/xgboost_final_test_metrics.json --plots-dir outputs/xgboost --prefix xgboost_final
```

### Modelo Alternativo: Stacking

#### 1) Entrenar el stacking
```bash
python src/train.py --config configs/training_stack.yml
```

Genera:
- `outputs/stacking/stacking_final.joblib`
- `outputs/stacking/stacking_final_metrics.json`

#### 2) Validar y generar graficas
```bash
python src/validate.py --model outputs/stacking/stacking_final.joblib --config configs/training_stack.yml
```

Genera:
- `outputs/stacking/stacking_final_test_metrics.json`
- `outputs/stacking/stacking_final_confusion_matrix.png`
- `outputs/stacking/stacking_final_roc_curve.png`
- `outputs/stacking/stacking_final_pr_curve.png`

---

## Arquitectura de Modelos

### CatBoost (Modelo Principal)
**CatBoost** es el modelo principal para prediccion de no-show porque:
- Maneja muy bien variables categoricas sin requerer codificación manual.
- Reduce overfitting significativamente mediante su esquema de ordenamiento interno (leaf-wise ordering).
- Proporciona predicciones robustas y bien calibradas.
- Optimizado para datos desbalanceados con `auto_class_weights: Balanced`.

**Estrategias implementadas:**
- División 70/15/15 (train/val/test) estratificada
- StandardScaler para normalización de features numéricas
- RandomizedSearchCV para optimización de hiperparámetros (25 iteraciones, 5-fold CV)
- Threshold optimization en validación (300 puntos de grid)

### Stacking (Modelo Alternativo)
El stacking combina LightGBM, XGBoost y CatBoost mediante un meta-modelo:

1. **Base estimators**: LightGBM, XGBoost y CatBoost generan probabilidades.
2. **Meta-modelo**: Regresión logística aprende a ponderar las salidas de los modelos base.
3. **Beneficio**: Diversidad de sesgos inductivos reduce errores correlacionados.

### Por que se usan LightGBM, XGBoost y CatBoost
- **LightGBM**: Muy eficiente con datos tabulares, captura relaciones no lineales.
- **XGBoost**: Robusto ante ruido y desequilibrio, proporciona estabilidad.
- **CatBoost**: Manejo óptimo de variables categóricas y reduction de overfitting.

## Notas de evaluacion
- Split 70/15/15 (train/val/test) estratificado.
- Threshold optimizado en validacion para maximizar `f1_no_show`.
- Metricas reportadas: accuracy, balanced_accuracy, f1_macro, f1_weighted, f1_no_show, roc_auc, pr_auc, cohen_kappa.

