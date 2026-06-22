# Instrucciones de la aplicacion Predictive No-Show FVL

Esta guia resume como instalar, entrenar, validar y ejecutar la aplicacion completa. El modelo principal actual es **CatBoost**; LightGBM, XGBoost y Stacking quedan como modelos alternativos o de comparacion.

---

## ⚠️ Nota Importante: Dataset Privado

**La aplicación requiere un dataset para funcionar completamente.** El dataset original proviene de **Fundación Valle del Lili** y contiene información confidencial de pacientes. Por razones de privacidad y cumplimiento con regulaciones de protección de datos, **el dataset real no se incluye en esta entrega.**

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

## Requisitos

- Python 3.10 o superior.
- Node.js y npm.
- Windows PowerShell, Git Bash o una terminal equivalente.
- Dataset limpio en `data/processed/df_limpio.csv`.
- Para regenerar el dataset desde Excel: `data/raw/database_non-shows.xlsx`.

## Instalacion

Desde la raiz del repositorio:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
```

El `requirements.txt` principal instala las dependencias de entrenamiento, incluyendo CatBoost. El `backend/requirements.txt` instala las dependencias de la API FastAPI, tambien con CatBoost para poder cargar `outputs/catboost/catboost_final.joblib`.

Instalar el frontend:

```powershell
cd frontend
npm install
cd ..
```

## Modelo principal: CatBoost

### Entrenar CatBoost

Desde la raiz:

```powershell
python src/train.py --config configs/training_catboost.yml
```

Este comando genera o actualiza:

- `outputs/catboost/catboost_final.joblib`
- `outputs/catboost/catboost_final_metrics.json`

La API usa este artefacto por defecto. Si existe `outputs/catboost/catboost_final.joblib`, el backend lo carga antes que cualquier modelo alternativo.

### Validar CatBoost y generar graficas

```powershell
python src/validate.py --model outputs/catboost/catboost_final.joblib --config configs/training_catboost.yml --output outputs/catboost/catboost_final_test_metrics.json --plots-dir outputs/catboost --prefix catboost_final
```

Este comando genera:

- `outputs/catboost/catboost_final_test_metrics.json`
- `outputs/catboost/catboost_final_confusion_matrix.png`
- `outputs/catboost/catboost_final_roc_curve.png`
- `outputs/catboost/catboost_final_pr_curve.png`

### Metricas actuales de CatBoost

Las metricas guardadas en `outputs/catboost/catboost_final_metrics.json` son:

| Metrica test | Valor | Porcentaje |
|---|---:|---:|
| Accuracy | 0.8143 | 81.43% |
| Balanced accuracy | 0.8032 | 80.32% |
| F1 macro | 0.7981 | 79.81% |
| F1 weighted | 0.8157 | 81.57% |
| F1 no-show | 0.7410 | 74.10% |
| Precision no-show | 0.7166 | 71.66% |
| Recall no-show | 0.7671 | 76.71% |
| ROC-AUC | 0.8940 | 89.40% |
| PR-AUC | 0.8450 | 84.50% |

Threshold optimizado: `0.5411`.

### Mejoras porcentuales de CatBoost

Comparado contra LightGBM en test:

| Metrica | Mejora relativa |
|---|---:|
| Accuracy | +0.71% |
| Balanced accuracy | +0.86% |
| F1 macro | +0.81% |
| F1 weighted | +0.71% |
| F1 no-show | +1.17% |
| Precision no-show | +0.98% |
| Recall no-show | +1.37% |
| ROC-AUC | +0.79% |
| PR-AUC | +1.11% |

Comparado contra XGBoost en test:

| Metrica | Cambio relativo |
|---|---:|
| Accuracy | -0.04% |
| Balanced accuracy | +1.38% |
| F1 macro | +0.51% |
| F1 weighted | +0.20% |
| F1 no-show | +1.66% |
| Precision no-show | -2.97% |
| Recall no-show | +6.62% |
| ROC-AUC | +0.47% |
| PR-AUC | +1.04% |

Comparado contra Stacking en test:

| Metrica | Cambio relativo |
|---|---:|
| Accuracy | -0.87% |
| Balanced accuracy | +0.59% |
| F1 macro | -0.37% |
| F1 weighted | -0.60% |
| F1 no-show | +0.50% |
| Precision no-show | -4.65% |
| Recall no-show | +6.01% |
| ROC-AUC | -0.02% |
| PR-AUC | -0.13% |

Conclusion operativa: CatBoost es el modelo principal porque mejora el **recall de no-show** frente a LightGBM, XGBoost y Stacking. Esto ayuda a detectar mas pacientes con riesgo de inasistencia. Stacking conserva una ligera ventaja en accuracy, precision y PR-AUC, pero CatBoost prioriza mejor la deteccion de no-shows.

## Modelos alternativos

Entrenar LightGBM:

```powershell
python src/train.py --config configs/training_lightgbm.yml
```

Validar LightGBM:

```powershell
python src/validate.py --model outputs/lightgbm/lightgbm_final.joblib --config configs/training_lightgbm.yml --output outputs/lightgbm/lightgbm_final_test_metrics.json --plots-dir outputs/lightgbm --prefix lightgbm_final
```

Entrenar XGBoost:

```powershell
python src/train.py --config configs/training_xgboost.yml
```

Validar XGBoost:

```powershell
python src/validate.py --model outputs/xgboost/xgboost_final.joblib --config configs/training_xgboost.yml --output outputs/xgboost/xgboost_final_test_metrics.json --plots-dir outputs/xgboost --prefix xgboost_final
```

Entrenar Stacking:

```powershell
python src/train.py --config configs/training_stack.yml
```

Validar Stacking:

```powershell
python src/validate.py --model outputs/stacking/stacking_final.joblib --config configs/training_stack.yml --output outputs/stacking/stacking_final_test_metrics.json --plots-dir outputs/stacking --prefix stacking_final
```

## Ejecutar backend

Desde la raiz:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Comprobar que la API esta viva:

```powershell
curl http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{"status":"ok"}
```

Notas:

- `backend/app/services/prediction.py` carga primero `outputs/catboost/catboost_final.joblib`.
- Si no existe CatBoost, intenta usar `outputs/stacking/stacking_final.joblib`.
- Si tampoco existe, usa el fallback legacy `backend/models/model.pkl`.
- El scheduler de reentrenamiento usa APScheduler. Si la dependencia falta, la API ya no se cae; el scheduler queda deshabilitado y las citas siguen funcionando.

## Ejecutar con Docker (Recomendado para producción)

### Requisitos
- Docker Desktop (https://www.docker.com/products/docker-desktop)
- Docker Compose (incluido)

### Uso rápido

Desde la raiz:

```powershell
docker-compose up --build
```

Esto:
- Construye e inicia backend en http://localhost:8000
- Construye e inicia frontend en http://localhost:5173
- Crea red privada entre servicios

### Comandos útiles

```powershell
# Detener
docker-compose down

# Ver logs
docker-compose logs -f

# Reconstruir sin cache
docker-compose build --no-cache

# Ejecutar solo backend
docker-compose up backend

# Ejecutar solo frontend
docker-compose up frontend
```

### Ventajas
- Ambiente consistente (dev/staging/prod)
- Sin dependencias locales
- Componentes aislados y escalables
- Facilita deployment

## Ejecutar frontend

En otra terminal:

```powershell
cd frontend
npm run dev
```

La aplicacion queda en:

```text
http://localhost:5173
```

Por defecto el frontend llama al backend en:

```text
http://localhost:8000
```

Si necesitas cambiar la URL de la API, crea `frontend/.env`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Build del frontend

```powershell
cd frontend
npm run build
```

Salida esperada:

- `frontend/dist/index.html`
- `frontend/dist/assets/...`

Para previsualizar el build:

```powershell
cd frontend
npm run preview
```

## Flujo completo recomendado

Desde la raiz del proyecto:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
python src/train.py --config configs/training_catboost.yml
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

En otra terminal:

```powershell
cd frontend
npm install
npm run dev
```

Abrir:

```text
http://localhost:5173
```

## Endpoints principales

### Salud

```powershell
curl http://127.0.0.1:8000/health
```

### Citas

Listar citas en la ventana por defecto de 8 dias:

```powershell
curl "http://127.0.0.1:8000/appointments/"
```

Listar citas proximos 15 dias:

```powershell
curl "http://127.0.0.1:8000/appointments/?days=15"
```

Listar todas las citas sin filtro de ventana:

```powershell
curl "http://127.0.0.1:8000/appointments/?days=0"
```

Buscar citas por medico:

```powershell
curl "http://127.0.0.1:8000/appointments/MED-014?days=15"
```

Crear cita:

```powershell
curl -X POST "http://127.0.0.1:8000/appointments/" -H "Content-Type: application/json" -d "{\"medic_id\":\"MED-014\",\"patient_id\":\"PAC-001\",\"hour\":10,\"day\":5,\"month\":6}"
```

Consultar detalle:

```powershell
curl "http://127.0.0.1:8000/appointments/info/1"
```

Eliminar cita:

```powershell
curl -X DELETE "http://127.0.0.1:8000/appointments/1"
```

Registrar resultado como Asistida:

```powershell
curl -X PATCH "http://127.0.0.1:8000/appointments/type/1?appointment_type=0"
```

Registrar resultado como No asistio:

```powershell
curl -X PATCH "http://127.0.0.1:8000/appointments/type/1?appointment_type=1"
```

Estados internos:

- `0`: Asistida.
- `1`: No asistio.
- `2`: En espera.

### Predicciones

Prediccion individual de una cita:

```powershell
curl "http://127.0.0.1:8000/predictions/appointment/1"
```

Predicciones de citas en espera en los proximos 8 dias:

```powershell
curl "http://127.0.0.1:8000/predictions/waiting?days=8"
```

Predicciones de citas en espera para un medico y ventana de 15 dias:

```powershell
curl "http://127.0.0.1:8000/predictions/waiting?medic_id=MED-014&days=15"
```

Informacion del modelo activo:

```powershell
curl "http://127.0.0.1:8000/model-info"
```

### Scheduler

Estado del scheduler:

```powershell
curl "http://127.0.0.1:8000/scheduler/status"
```

Iniciar scheduler:

```powershell
curl -X POST "http://127.0.0.1:8000/scheduler/start"
```

Detener scheduler:

```powershell
curl -X POST "http://127.0.0.1:8000/scheduler/stop"
```

Cambiar intervalo a 30 minutos:

```powershell
curl -X PUT "http://127.0.0.1:8000/scheduler/interval?minutes=30"
```

Ejecutar reentrenamiento manual de CatBoost:

```powershell
curl -X POST "http://127.0.0.1:8000/scheduler/manual-retrain"
```

Recargar el modelo desde disco (sin reiniciar):

```powershell
curl -X POST "http://127.0.0.1:8000/scheduler/reload-model"
```

Nota importante: el scheduler actual ejecuta `backend/models/train_pipeline.py`, que ahora entrena CatBoost y actualiza `outputs/catboost/catboost_final.joblib`. Después de cada reentrenamiento automático exitoso, el modelo se recarga dinámicamente sin necesidad de reiniciar el servidor. También puedes regenerar el modelo principal manualmente con `python src/train.py --config configs/training_catboost.yml`.

## Ventanas de fechas en la app

El calendario filtra citas por una ventana hacia adelante desde la fecha actual:

- `days=8`: proximos 8 dias.
- `days=15`: proximos 15 dias.
- `days=30`: proximo mes.
- `days=0`: sin filtro por ventana.

La app usa el calendario 2026 para construir fechas desde `day`, `month` y `hour`.

## Estados visuales del calendario

- Asistira: azul.
- No asistira: rojo.
- Asistida: verde.
- No asistio: naranja.
- Sin prediccion: gris.

## Reentrenamiento y actualizacion de datos

Cuando una cita en espera se cierra como `Asistida` o `No asistio`, el backend llama a `backend/app/services/training_service.py`. El flujo esperado es:

1. Buscar el ultimo registro historico del paciente.
2. Crear una nueva fila en `data/raw/database_non-shows.xlsx`.
3. Actualizar contadores de asistencia o inasistencia.
4. Ejecutar reentrenamiento de CatBoost desde `backend/models/train_pipeline.py`.

### Recarga dinámica del modelo

Después de cada reentrenamiento automático exitoso, el modelo se recarga dinámicamente:

1. El scheduler ejecuta `train_pipeline.py`
2. Se generan nuevos artefactos en `/app/outputs/catboost/catboost_final.joblib` y `/app/backend/models/model.pkl`
3. Se llama automáticamente a `reload_model()`
4. Las variables globales se resetean y se cargan desde disco
5. **Próximas predicciones usan el nuevo modelo sin reiniciar el servidor**

Para regenerar CatBoost desde el pipeline configurable, ejecutar manualmente:

```powershell
python src/train.py --config configs/training_catboost.yml
```

Luego usar el endpoint de recarga (o esperar al siguiente ciclo del scheduler):

```powershell
curl -X POST "http://127.0.0.1:8000/scheduler/reload-model"
```

O simplemente reiniciar el backend para que cargue el nuevo `outputs/catboost/catboost_final.joblib`.

## Problemas comunes

### Error: `Failed to fetch`

Normalmente significa que el frontend no pudo llegar al backend.

Revisar:

```powershell
curl http://127.0.0.1:8000/health
```

Si no responde, iniciar Uvicorn:

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### Error: `No module named 'catboost'`

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r backend/requirements.txt
```

### Error: `No module named 'apscheduler'`

Instalar dependencias del backend:

```powershell
python -m pip install -r backend/requirements.txt
```

### No existe `outputs/catboost/catboost_final.joblib`

Entrenar el modelo principal:

```powershell
python src/train.py --config configs/training_catboost.yml
```

### Cambie el modelo y la API sigue usando el anterior

Reiniciar Uvicorn. El servicio carga los artefactos al iniciar.

## Archivos clave

- `configs/training_catboost.yml`: configuracion del entrenamiento principal.
- `models/catboost/config.yml`: hiperparametros base de CatBoost.
- `src/train.py`: entrenamiento de modelos finales.
- `src/validate.py`: validacion y graficas.
- `outputs/catboost/catboost_final.joblib`: modelo principal usado por la API.
- `outputs/catboost/catboost_final_metrics.json`: metricas de CatBoost.
- `backend/app/services/prediction.py`: carga del modelo y predicciones.
- `backend/app/routes/appointment_routes.py`: endpoints de citas.
- `backend/app/routes/prediction_routes.py`: endpoints de prediccion.
- `frontend/src/App.jsx`: interfaz principal.
- `frontend/packages/shared/src/appointmentStatus.js`: estados y colores.
