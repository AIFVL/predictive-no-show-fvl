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
