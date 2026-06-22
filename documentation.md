# Documentacion del Proyecto

## ⚠️ Requisito Importante: Dataset

**La aplicación requiere un dataset para funcionar correctamente.** El dataset utilizado en este proyecto es información institucional de **Fundación Valle del Lili** y contiene datos sensibles de pacientes. Por motivos de **confidencialidad y protección de datos personales**, no se incluye en esta entrega.

### Implicaciones

**Sin un dataset en `data/raw/`, la aplicación no podrá:**
- ✗ Entrenar o reentrenar modelos
- ✗ Inicializar la base de datos
- ✗ Ejecutar predicciones
- ✗ Cargar datos de citas

La aplicación mostrará errores al intentar:
1. Ejecutar `docker-compose up` (fallarán los scripts de inicialización)
2. Correr `python src/train.py` (no encontrará dataset)
3. Acceder a endpoints de predicción (sin datos en BD)

### Solución

Para consultar sobre las funcionalidades restantes, **contacta al equipo de desarrollo**:

josealejandromc4@gmail.com

samuelalvarez2221@gmail.com

samuel.ibarra1227@gmail.com

---

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
- `docker-compose.yml`: orquestación de contenedores (backend, frontend).
- `backend/Dockerfile`: contenedor backend (Python 3.13 + FastAPI).
- `backend/.dockerignore`: archivos a excluir del build del backend.
- `frontend/Dockerfile`: contenedor frontend (Node 18 + Nginx).
- `frontend/nginx.conf`: configuración de Nginx para el frontend.
- `frontend/.dockerignore`: archivos a excluir del build del frontend.
- `.dockerignore`: archivos a excluir de todos los builds.

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

---

## Dockerización del Proyecto

El proyecto incluye soporte completo para Docker Compose con contenedores separados para backend, frontend y almacenamiento de datos.

### Arquitectura Docker

```
┌─────────────────────────────────────────┐
│      Docker Network (predictive)        │
│  ┌───────────────┐  ┌──────────────┐   │
│  │ Backend       │  │  Frontend    │   │
│  │ :8000         │  │  :5173       │   │
│  │ FastAPI       │  │  React+Nginx │   │
│  │ + Scheduler   │  │              │   │
│  │ + CatBoost    │  │              │   │
│  └───────────────┘  └──────────────┘   │
│        ▲                    ▲            │
│        │                    │            │
│  ┌─────┴────────────────────┴─────┐   │
│  │  Volúmenes Compartidos         │   │
│  │  - data/                        │   │
│  │  - outputs/                     │   │
│  │  - backend/appointments.db      │   │
│  └────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Componentes

**Backend (`backend/Dockerfile`)**
- Base: Python 3.13-slim
- Framework: FastAPI + Uvicorn
- Features: Modelo ML, Scheduler, APIs REST
- Puerto: 8000
- Volúmenes: data/, outputs/, db

**Frontend (`frontend/Dockerfile`)**
- Build: Node 18 + Vite
- Serving: Nginx alpine
- Features: React app + reverse proxy a backend
- Puerto: 5173 (HTTP)
- Volúmenes: ninguno (build estático)

**Network**
- Driver: bridge
- Nombre: `predictive-network`
- Permite comunicación entre contenedores por hostname

### Ejecución Rápida

#### Build y start

```bash
docker-compose up --build
```

#### Ejecución sin rebuild

```bash
docker-compose up
```

#### Detener

```bash
docker-compose down
```

#### Detener y limpiar volúmenes

```bash
docker-compose down -v
```

### Logs y Debugging

```bash
# Todos los servicios
docker-compose logs -f

# Solo backend
docker-compose logs -f backend

# Solo frontend
docker-compose logs -f frontend

# Últimas 100 líneas
docker-compose logs --tail=100
```

### Ejecutar comandos en contenedor

```bash
# Bash en backend
docker-compose exec backend bash

# Bash en frontend
docker-compose exec frontend sh

# Ejecutar comando específico
docker-compose exec backend python src/train.py --config configs/training_catboost.yml
```

### Health Checks

Both servicios incluyen health checks automáticos:

```bash
# Ver estado
docker-compose ps

# El output muestra: backend (healthy), frontend (healthy)
```

### Volúmenes Persistentes

- `data/`: dataset raw y procesado
- `outputs/`: modelos entrenados y métricas
- `backend/appointments.db`: base de datos SQLite

Estos volúmenes **sobreviven** a `docker-compose down` y se reutilizan en el próximo `up`.

### Desarrollo Local

Para desarrollo sin Docker:

```bash
# Terminal 1: Backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

### Producción

Para deployment en servidor:

1. Clonar repositorio
2. Instalar Docker + Docker Compose
3. Configurar variables de entorno (si es necesario)
4. `docker-compose -f docker-compose.yml up -d`
5. Usar reverse proxy (Nginx/Traefik) para SSL

### Troubleshooting

**Contenedor no inicia**
```bash
docker-compose logs backend  # Ver error específico
```

**Puerto ya en uso**
```bash
# Cambiar en docker-compose.yml
# ports:
#   - "8001:8000"  # Host:Container
```

**Reconstruir sin cache**
```bash
docker-compose build --no-cache
```

**Limpiar todo**
```bash
docker-compose down -v
docker system prune -a
```

