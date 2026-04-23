# Documentacion de Resultados — Feature Engineering

## Contexto
Se aplico feature engineering al dataset `df_limpio.csv` para mejorar la prediccion de inasistencia (No‑Show) en medicina interna.  
El objetivo era superar **Accuracy > 0.80** y **F1 Macro > 0.80**, con **Kappa alto** y sin overfitting.

## Cambios de Feature Engineering
Se agregaron variables derivadas sin fuga de informacion:

- **Historico de asistencia**
  - `Prev_Total` = `Number of Previous Attendance` + `Number of Previous Non-Attendance`
  - `Has_Prev` indicador de historial previo
  - `Prev_NoShow_Rate` = No‑Show / total previo
  - `Prev_Show_Rate` = Show / total previo
- **Carga clinica**
  - `Clinical_Burden` = `Number of Diseases` + `Number of Medications`
  - `Disease_Med_Ratio` = `Number of Diseases` / (`Number of Medications` + 1)
  - `Hosp_x_Diseases` = `Recent Hospitalization` * `Number of Diseases`
- **Tiempo de espera**
  - `Lead_Time_Log` = log(1 + `Creation to Assignment Interval`)
- **Ciclicas**
  - `Hour_Sin`, `Hour_Cos`
  - `Day_Sin`, `Day_Cos`
  - `Month_Sin`, `Month_Cos`
- **Binning**
  - `Age_Bin` (0‑17, 18‑39, 40‑59, 60‑79, 80+)
  - `Hour_Bin` (Morning, Afternoon, Evening)

Todas las nuevas variables se generaron **solo con datos disponibles al momento de la cita**, evitando leakage.

## Pipeline y Evaluacion
- Split 70/15/15 (train/val/test)
- CV 5‑fold estratificada
- SMOTE dentro de cada fold (imblearn.Pipeline)
- Threshold optimizado en validacion para **F1 Macro**

## Resultados Principales (Test)
Los resultados obtenidos despues del FE fueron:

- **XGBoost + FE**
  - Accuracy: **0.8231**
  - F1 Macro: **0.8039**
  - Kappa: **0.6078**
- **LightGBM + FE**
  - Accuracy: **0.8227**
  - F1 Macro: **0.8037**
  - Kappa: **0.6075**
- **Stacking + FE**
  - Accuracy: **0.8209**
  - F1 Macro: **0.8008**
  - Kappa: **0.6016**

Estos resultados **cumplen** los objetivos de Accuracy y F1 Macro > 0.80.

## Evidencia de No Overfitting
Se agrego un reporte automatico que compara **Train vs Val vs Test** con el mismo threshold:

- Si los valores son similares y el **gap Train‑Test es pequeno**, no hay overfitting severo.
- En esta ejecucion, el F1 Macro en train/val/test se mantuvo cercano (~0.80), indicando generalizacion estable.

Para reproducir:
```bash
python src/train_stacking_fe.py
```

## Archivos Clave
- `src/feature_engineering.py`
- `src/train_stacking_fe.py`
- `data/processed/resultados_stacking_fe.csv`

