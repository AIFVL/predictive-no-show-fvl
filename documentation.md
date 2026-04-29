# Documentacion del Proyecto

## Objetivo
Este repositorio construye un modelo de machine learning para predecir inasistencias (No-Show) en consultas de medicina interna. El objetivo es ayudar a la gestion de citas mediante un modelo principal de stacking que combina LightGBM, XGBoost y CatBoost.

## Estructura completa del repositorio
- `configs/`
  - `training_stack.yml`: configuracion para el modelo de stacking.
  - `training_lightgbm.yml`: configuracion para LightGBM.
  - `training_xgboost.yml`: configuracion para XGBoost.
  - `training_catboost.yml`: configuracion para CatBoost.
- `data/`
  - `docs/`: documentacion y notas del dataset.
  - `methodology CRISP-DM/`: fases de la metodologia.
  - `processed/df_limpio.csv`: dataset limpio usado para el entrenamiento.
  - `raw/`: datos originales.
- `models/`
  - `lightgbm/`
    - `model.py`: define el modelo individual LightGBM.
    - `config.yml`: hiperparametros de LightGBM.
  - `xgboost/`
    - `model.py`: define el modelo individual XGBoost.
    - `config.yml`: hiperparametros de XGBoost.
  - `catboost/`
    - `model.py`: define el modelo individual CatBoost.
    - `config.yml`: hiperparametros de CatBoost.
  - `stacking/`
    - `model.py`: define el stacking con LightGBM + XGBoost + CatBoost.
    - `config.yml`: hiperparametros del stacking.
- `notebooks/`
  - `01_Exploración.ipynb`
  - `02_Limpieza.ipynb`
- `outputs/`
  - `lightgbm/`: artefactos y graficas de LightGBM.
  - `xgboost/`: artefactos y graficas de XGBoost.
  - `catboost/`: artefactos y graficas de CatBoost.
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

### Modelos individuales

#### LightGBM
Entrenar:
```bash
python src/train.py --config configs/training_lightgbm.yml
```
Validar:
```bash
python src/validate.py --model outputs/lightgbm/lightgbm_final.joblib --config configs/training_lightgbm.yml --output outputs/lightgbm/lightgbm_final_test_metrics.json --plots-dir outputs/lightgbm --prefix lightgbm_final
```

#### XGBoost
Entrenar:
```bash
python src/train.py --config configs/training_xgboost.yml
```
Validar:
```bash
python src/validate.py --model outputs/xgboost/xgboost_final.joblib --config configs/training_xgboost.yml --output outputs/xgboost/xgboost_final_test_metrics.json --plots-dir outputs/xgboost --prefix xgboost_final
```

#### CatBoost
Entrenar:
```bash
python src/train.py --config configs/training_catboost.yml
```
Validar:
```bash
python src/validate.py --model outputs/catboost/catboost_final.joblib --config configs/training_catboost.yml --output outputs/catboost/catboost_final_test_metrics.json --plots-dir outputs/catboost --prefix catboost_final
```

### Stacking

#### 1) Entrenar el stacking final
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

## Como funciona el stacking
El modelo principal es un ensemble por stacking: combina las predicciones de tres modelos base y entrena un modelo final (meta-modelo) que aprende a ponderar esas salidas. El flujo es:

1. LightGBM, XGBoost y CatBoost generan probabilidades de no-show.
2. Esas probabilidades se usan como nuevas features para el meta-modelo.
3. El meta-modelo aprende la mejor combinacion para mejorar la generalizacion.

## Por que se usan LightGBM, XGBoost y CatBoost
- **LightGBM**: muy eficiente con datos tabulares, captura relaciones no lineales con buen rendimiento en tiempo y memoria.
- **XGBoost**: robusto ante ruido y desequilibrio, suele aportar estabilidad y buen poder predictivo.
- **CatBoost**: maneja bien variables categoricas y reduce overfitting con su esquema de ordenamiento interno.

La combinacion aporta diversidad de modelos (diferentes sesgos inductivos), lo que ayuda a reducir errores correlacionados.

## Notas de evaluacion
- Split 70/15/15 (train/val/test) estratificado.
- Threshold optimizado en validacion para maximizar `f1_no_show`.
- Metricas reportadas: accuracy, balanced_accuracy, f1_macro, f1_weighted, f1_no_show, roc_auc, pr_auc, cohen_kappa.

