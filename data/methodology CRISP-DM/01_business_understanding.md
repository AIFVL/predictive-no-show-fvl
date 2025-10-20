# 1. Comprensión del Negocio (Business Understanding)

## 📋 Introducción a la Metodología CRISP-DM

La metodología seleccionada para guiar el desarrollo del proyecto de Data Mining es el **CRoss-Industry Standard Process for Data Mining (CRISP-DM)**. Este estándar industrial proporciona un marco estructurado que divide el proyecto en seis fases interconectadas y cíclicas: Comprensión del Negocio, Comprensión de los Datos, Preparación de los Datos, Modelado, Evaluación y Despliegue.

### ¿Por qué utilizamos CRISP-DM en este proyecto?

Este proyecto de **predicción de no-shows médicos** (inasistencias a citas médicas) requiere un enfoque estructurado y sistemático que garantice:

- **Alineación con objetivos de negocio**: Entender primero el problema real antes de saltar a la implementación técnica.
- **Proceso iterativo**: Permite refinar modelos basándose en resultados y feedback continuo.
- **Reproducibilidad**: Documentación clara de cada fase para facilitar la replicación y mejora.
- **Gestión de riesgos**: Identificación temprana de problemas y limitaciones.
- **Comunicación efectiva**: Marco común para todos los stakeholders del proyecto.

### Beneficios de CRISP-DM

- **Estructura clara**: 6 fases bien definidas que guían todo el proyecto.
- **Flexibilidad**: Adaptable a diferentes contextos y dominios (salud, finanzas, retail, etc.).
- **No dependiente de herramientas**: Independiente de tecnologías específicas.
- **Basado en mejores prácticas**: Décadas de experiencia condensadas en un framework probado.
- **Enfoque orientado a resultados**: Prioriza el valor de negocio sobre la complejidad técnica.
- **Reducción de costos**: Evita trabajo innecesario al validar cada fase antes de avanzar.
- **Mayor tasa de éxito**: Proyectos mejor planificados tienen mayor probabilidad de implementación exitosa.
- **ROI medible**: Permite establecer métricas claras de éxito desde el inicio.
- **Comunicación mejorada**: Lenguaje común entre técnicos y stakeholders de negocio.
- **Documentación completa**: Facilita transferencia de conocimiento y mantenimiento.
- **Escalabilidad**: Framework que crece con la madurez analítica de la organización.

---

## 4.1 Fase 1: Comprensión del Negocio (Business Understanding)

El objetivo primordial de esta fase inicial es alinear rigurosamente los objetivos del proyecto de minería de datos con las metas y exigencias del negocio de la Fundación Valle del Lili (FVL), garantizando que el resultado genere un valor tangible para la institución.

### 4.1.1 Determinación de los Objetivos de Negocio

El proyecto se centra en mitigar el impacto negativo de las inasistencias (no-show) a las consultas externas de medicina interna, un problema que afecta la eficiencia, la calidad asistencial y el acceso oportuno a la salud.

**Principales objetivos de negocio:**

- **Optimización Operacional:** Reducir la tasa de inasistencias para maximizar el uso de los cupos de atención médica, optimizando la asignación de recursos humanos y físicos, y reduciendo las pérdidas económicas derivadas de los espacios de agenda vacíos.
- **Mejora de la Calidad Asistencial:** Garantizar la continuidad en el seguimiento de los pacientes, especialmente aquellos con patologías crónicas, al disminuir los retrasos en la atención que resultan del no-show.
- **Mejora de la Experiencia del Paciente y Acceso:** Implementar un sistema predictivo que permita la gestión proactiva de la agenda, liberando cupos con suficiente antelación para ser reasignados a otros pacientes en lista de espera.
- **Innovación Basada en Datos:** Consolidar una herramienta de análisis avanzado para la toma de decisiones en la gestión de citas, marcando un avance en la transformación digital de los servicios de la FVL.

### 4.1.2 Evaluación de la Situación Actual

La gestión actual de las inasistencias se basa en procesos reactivos o recordatorios masivos, sin la capacidad de identificar la probabilidad individual de no-show.

- **Procesos Actuales:** Los recordatorios de citas son gestionados de forma generalizada (e.g., llamadas telefónicas automatizadas o mensajes de texto) para todos los pacientes, independientemente de su riesgo histórico o contextual.
- **Limitaciones:** La falta de un mecanismo predictivo impide que las intervenciones (ej. sobre-agendamiento, llamadas de confirmación) se dirijan únicamente a los pacientes con alta probabilidad de inasistencia, resultando en un uso ineficiente de los recursos administrativos. El sistema actual no anticipa el problema, sino que reacciona a él una vez que el cupo ya se ha perdido.

### 4.1.3 Determinación de los Objetivos de Minería de Datos

Los objetivos de minería de datos se definen para traducir la necesidad del negocio en un problema técnico resoluble mediante Machine Learning.

**Objetivo General de Minería de Datos:**

Construir, entrenar y validar un modelo de clasificación binaria utilizando técnicas de aprendizaje automático que pueda predecir la probabilidad de inasistencia (No-Show) de un paciente a una consulta agendada en la especialidad de medicina interna.

**Objetivos Específicos de Minería de Datos:**

- **Exploración y Preprocesamiento:** Recolectar datos históricos de citas, características sociodemográficas y registros clínicos para construir un dataset robusto.
- **Ingeniería de Características:** Identificar y transformar las variables con mayor poder predictivo (ej. historial de no-shows, tiempo de antelación de la cita, tipo de aseguradora).
- **Modelado y Comparación:** Implementar y evaluar diversos algoritmos de Machine Learning (como Random Forest y XGBoost) para seleccionar el modelo con el rendimiento predictivo óptimo.
- **Validación:** Evaluar el modelo seleccionado con métricas de rendimiento robustas (AUC-ROC, F1-Score) para garantizar su precisión y utilidad clínica.

### 4.1.4 Definición de Criterios de Éxito

Los criterios de éxito se dividen en técnicos y de negocio para asegurar la relevancia y aplicabilidad del proyecto.

| Criterio                      | Métrica   | Descripción                                                                 |
|-------------------------------|-----------|----------------------------------------------------------------------------|
| **Técnico Principal**         | AUC-ROC   | El modelo debe alcanzar un valor mínimo de 0.85, indicando excelente capacidad de discriminación entre pacientes asistentes e inasistentes. |
| **Técnico Secundario**        | F1-Score  | Se debe obtener un valor aceptable que balancee la Precisión y el Recall, crucial para clasificar correctamente los casos minoritarios (no-shows). |
| **Negocio**                   | Reducción del Tasa de No-Show | Tras la implementación de una estrategia de intervención dirigida (basada en las predicciones del modelo), se espera una reducción de la tasa de inasistencia de al menos un 10% en los grupos de alto riesgo. |

---


