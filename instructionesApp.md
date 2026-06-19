# Resumen de lo hecho (backend)

- Estructura propuesta
  - backend/
    - app/ (FastAPI aún por crear)
    - models/ (contiene train_pipeline.py)
    - tests/, scripts/, data/ etc.
  - data/raw/database_non-shows.xlsx debe existir
  - data/processed/ contiene df_limpio.csv tras limpieza

- train_pipeline.py (backend/models/train_pipeline.py)
  - Carga raw Excel desde data/raw/
  - Normaliza nombres y tipos de columna (replica 02_Limpieza.ipynb)
  - Aplica reglas de validación y elimina duplicados
  - Quita outliers (Creation to Assignment Interval > 365)
  - Selecciona features numéricas y categóricas según notebook
  - Construye pipeline sklearn:
    - ColumnTransformer: OneHotEncoder (categoricals) + StandardScaler (num)
    - Imputación/llenado simple donde procede
    - Modelo por defecto: DecisionTreeClassifier (parámetros ya tuneados en notebook)
  - Entrena, valida (train/test split) e imprime métricas (accuracy, classification_report)
  - Serializa el pipeline en backend/models/model.pkl
  - Guarda métricas en backend/models/metrics.json y df_limpio.csv en data/processed/

- Dependencias
  - Se sugirió backend/requirements.txt (o backend/requirement.txt en tu repo)
  - Comando de instalación (desde la raíz del repo, Windows):
    - Si el fichero es backend\requirements.txt:
      python -m pip install -r backend\requirements.txt
    - Si el fichero es backend\requirement.txt:
      python -m pip install -r backend\requirement.txt
    - Alternativa: renombrar backend\requirement.txt → backend\requirements.txt y luego instalar

- Cómo generar el .pkl (ya listo)
  - Ejecutar desde la raíz:
    python backend\models\train_pipeline.py
  - Verificar que data/raw/database_non-shows.xlsx existe antes de ejecutar
  - Tras ejecución: revisar backend/models/model.pkl y backend/models/metrics.json

- Sugerencias previas ya dadas (opcionalmente implementadas)
  - Crear backend/app/deps.py para centralizar imports (conveniencia)
  - Crear FastAPI app (backend/app/main.py, schemas, services/prediction.py) que cargue model.pkl con joblib y exponga endpoint /predict
  - Persistencia: usar SQLite/Postgres y ORM (SQLModel/SQLAlchemy) para citas/pacientes

---

# Instrucciones principales (resumen de uso)

Estas instrucciones explican cómo ejecutar el frontend y backend localmente y cómo usar los endpoints implementados hasta ahora.

**Requisitos previos**
- Tener Python 3.10+ instalado.
- Node.js + npm para el frontend.
- (Opcional) Activar / pausar OneDrive si los archivos de DB se encuentran en una carpeta sincronizada.

**Ejecutar backend (API)**
- Instalar dependencias (desde la raíz):
  ```powershell
  python -m pip install -r backend/requirements.txt
  ```
- Iniciar el servidor (desde la raíz del repo):
  ```powershell
  python -m uvicorn backend.app.main:app --reload --port 8000
  ```
- Nota: `init_db()` crea las tablas y contiene una comprobación ligera que añade la columna `medic_id` si falta (no borra datos).

**Recrear la base de datos (opcional, borra datos existentes)**
- Si quieres empezar desde cero borra `backend/appointments.db` (asegúrate de parar el servidor primero). Luego reinicia Uvicorn y la DB se recreará.

**Ejecutar frontend**
- Ir al directorio `frontend` y instalar dependencias:
  ```bash
  cd frontend
  npm install
  npm run dev
  ```
- El frontend asume que el backend está en `http://localhost:8000`.
- FullCalendar CSS se carga vía CDN en `frontend/index.html` (por compatibilidad con dependencias que no exportan el CSS directamente).

**Endpoints disponibles (resumen)**
- `GET /health` → estado (dev).  
- `GET /appointments/` → lista todas las citas.  
- `POST /appointments/` → crear cita. Payload JSON: `{ "medic_id": "m1", "patient_id": "p1", "hour":10, "day":5, "month":6 }`.
- `GET /appointments/{medic_id}` → lista citas del médico con id `medic_id`.
- `GET /appointments/info/{appointment_id}` → devuelve información completa de una sola cita (usado por el modal de detalles).
- `PUT /appointments/{appointment_id}` → actualizar cita (incluye `medic_id`).
- `PATCH /appointments/{appointment_id}/type` → actualizar estado de cita (0=Asistida, 1=No asistió, 2=Pendiente). **Nota: Al cambiar a 0 o 1, se dispara automáticamente la actualización de datos de entrenamiento y reentrenamiento del modelo**.
- `DELETE /appointments/{appointment_id}` → eliminar cita.

Ejemplos (curl):
```bash
# Crear cita
curl -X POST http://127.0.0.1:8000/appointments/ -H 'Content-Type: application/json' -d '{"medic_id":"m1","patient_id":"p1","hour":10,"day":5,"month":6}'

# Listar todas
curl http://127.0.0.1:8000/appointments/

# Buscar por médico
curl http://127.0.0.1:8000/appointments/m1

# Info de una cita
curl http://127.0.0.1:8000/appointments/info/1
```

**Comportamiento del frontend**
- El calendario (FullCalendar) muestra las citas convirtiendo `hour`, `day`, `month` a una fecha en 2026 por defecto.  
- Añadir cita: hay un botón y un formulario inline que envía `POST /appointments/` y refresca la vista.  
- Buscar por Médico ID: barra en la cabecera; al buscar llama `GET /appointments/{medic_id}` y muestra sólo esas citas.  
- Detalle de cita: al hacer click en un evento el frontend llama `GET /appointments/info/{appointment_id}` y muestra un modal con toda la información.

**Notas de diseño y estilos**
- FullCalendar se inicializa en `frontend/src/App.jsx` y su CSS principal se carga desde CDN en `frontend/index.html`.  
- El proyecto incluye clases Tailwind en el JSX; si Tailwind no está configurado en tu entorno, hay estilos inline de fallback para botones y el modal.

---

# Flujo de Actualización de Datos de Entrenamiento y Reentrenamiento

## Descripción del proceso

Cuando cambias el estado de una cita a **"Asistida"** (0) o **"No asistió"** (1):

1. **Registro de datos:** Se crea un nuevo registro en `data/raw/database_non-shows.xlsx` con:
   - Los datos clínicos del paciente (edad, enfermedades, medicamentos, etc.) del último registro histórico
   - La información de la cita actual (hora, día, mes)
   - El contador `Number of Previous Attendance` o `Number of Previous Non-Attendance` incrementado en 1
   - El estado de asistencia registrado

2. **Reentrenamiento automático:** El modelo se reentrena automáticamente usando:
   - El Excel actualizado con el nuevo registro
   - Todos los pasos de limpieza y normalización
   - El pipeline sklearn con SMOTE, StandardScaler, OneHotEncoder
   - LightGBM como modelo base

3. **Predicciones futuras:** En la siguiente predicción para ese paciente:
   - El modelo usará los datos históricos más recientes
   - Los contadores de asistencias/no-asistencias estarán actualizados
   - Las predicciones serán más precisas basadas en el comportamiento real

## Arquitectura de la implementación

**Nuevo servicio:** `backend/app/services/training_service.py`

Funciones principales:
- `get_latest_patient_record()` - Obtiene el último registro del paciente desde el Excel
- `append_training_record()` - Añade un nuevo registro con datos actualizados
- `retrain_model()` - Ejecuta el pipeline de entrenamiento
- `handle_appointment_status_change()` - Orquesta todo el proceso

**Cambios en endpoints:**
- El endpoint `PATCH /appointments/{appointment_id}/type` ahora dispara automáticamente el proceso de entrenamiento cuando el estado cambia a 0 o 1

## Ejemplo de uso

```bash
# 1. Cambiar estado de cita a "Asistida" (0)
curl -X PATCH http://127.0.0.1:8000/appointments/1/type \
  -H 'Content-Type: application/json' \
  -d '0'

# Detrás de escenas:
# - Se guarda un nuevo registro en data/raw/database_non-shows.xlsx
# - Se incrementa "Number of Previous Attendance" del paciente
# - Se ejecuta train_pipeline.py automáticamente
# - El modelo se actualiza con los nuevos datos

# 2. Verificar que el reentrenamiento funcionó
# Revisa los logs del servidor:
# - "Training update: Cita actualizada: paciente p1, estado Asistida. Registro guardado y modelo reentrenado."
```

## Dependencias adicionales requeridas

Asegúrate de que `backend/requirements.txt` incluya:
```
openpyxl>=3.0.0  # Para escribir/actualizar Excel
```

Si no está, instálalo manualmente:
```powershell
python -m pip install openpyxl
```

**Siguientes mejoras sugeridas**
- Añadir validación en frontend para los campos del formulario (horas/fechas).  
- Formatear `created_at` en el modal (usar `Intl.DateTimeFormat`).  
- Añadir edición/eliminación desde el modal.  
- Persistencia avanzada: usar migraciones reales (Alembic) en lugar de alteraciones ad-hoc.

Si quieres, actualizo este fichero con más ejemplos o añado una sección "Despliegue" para producción.
