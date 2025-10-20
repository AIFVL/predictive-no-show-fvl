# 2. Comprensión de los Datos (Data Understanding)

## 2.1 Recolección Inicial de Datos

La Fundación Valle del Lili proporcionó un conjunto de datos histórico de citas médicas, que incluye variables sociodemográficas, clínicas y de comportamiento.
Los datos provienen de **múltiples servidores y sistemas transaccionales**, lo que implica retos de integración y calidad.

- **Origen de los datos:**  
  - Servidor de citas externas (transacciones de agendamiento)
  - Servidor de historia clínica (variables clínicas)
  - Servidor de gestión administrativa (variables sociodemográficas)

- **Variables principales:**  
  - Sociodemográficas: Edad, sexo, tipo de seguro
  - Clínicas: Diagnósticos, hospitalizaciones, medicamentos
  - Historial de comportamiento: Asistencias e inasistencias previas
  - Temporales: Hora, día, mes, intervalo de asignación

## 2.2 Descripción de los Datos

- **Cantidad de registros:** [Completar con el número real]
- **Cantidad de variables:** 13 principales
- **Formato:** Archivo Excel (.xlsx)
- **Diccionario de datos:** Disponible en la carpeta `data/docs/`

## 2.3 Exploración Inicial y EDA

Se realizó un análisis exploratorio (EDA) para:

- Identificar la distribución de cada variable
- Detectar valores nulos, duplicados y atípicos (outliers)
- Analizar la correlación entre variables y la variable objetivo (no-show)
- Visualizar la dispersión y patrones de los datos

## 2.4 Calidad e Integridad de los Datos

- **Problemas detectados:**
  - **Datos faltantes:** Algunos registros carecen de información clínica o sociodemográfica por la integración de fuentes distintas.
  - **Sobre-escritura y pérdida de datos:** Al integrar variables de diferentes sistemas, algunos datos pueden haberse sobreescrito o perdido, especialmente en pacientes con múltiples citas o cambios administrativos.
  - **Inconsistencias:** Diferencias en la codificación de variables categóricas (por ejemplo, tipo de seguro, sexo).
  - **Duplicados:** Registros repetidos por errores de transacción.
  - **Outliers:** Valores extremos en variables como edad, número de hospitalizaciones, medicamentos diarios.

- **Tratamiento realizado:**
  - Limpieza de registros inválidos y duplicados
  - Imputación o eliminación de valores nulos según el caso
  - Homogeneización de variables categóricas
  - Detección y análisis de outliers para decidir su tratamiento

## 2.5 Valor y Limitaciones de los Datos

- **Valor:**  
  - Permiten construir un modelo predictivo robusto, con información relevante para identificar patrones de inasistencia.
  - La integración de variables clínicas y sociodemográficas enriquece el análisis y la capacidad de predicción.

- **Limitaciones:**  
  - Posible pérdida de información por integración de sistemas
  - Desbalance de clases (más asistencias que inasistencias)
  - Calidad variable según la fuente original
  - Falta de variables externas (por ejemplo, clima, transporte)

## 2.6 Conclusiones y Siguientes Pasos

- Los datos disponibles son adecuados para el desarrollo del modelo, pero requieren un proceso riguroso de limpieza y validación.
- Es fundamental documentar todas las decisiones tomadas en la limpieza y tratamiento de datos para asegurar la reproducibilidad y transparencia.
- Se recomienda mantener comunicación con el área de sistemas de la Fundación para resolver dudas sobre la integración y origen de variables.