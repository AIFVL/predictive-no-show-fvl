/** Stored workflow states in the database (appointment_type). */
export const APPOINTMENT_TYPE = {
  ASISTIDA: 0,
  NO_ASISTIO: 1,
  EN_ESPERA: 2,
}

/** Unified display states shown in calendar, filters and detail views. */
export const DISPLAY_STATUS = {
  ASISTIDA: 'asistida',
  NO_ASISTIO: 'no_asistio',
  ASISTIRA: 'asistira',
  NO_ASISTIRA: 'no_asistira',
  SIN_PREDICCION: 'sin_prediccion',
}

export const DISPLAY_STATUS_LABELS = {
  [DISPLAY_STATUS.ASISTIDA]: 'Asistida',
  [DISPLAY_STATUS.NO_ASISTIO]: 'No asistió',
  [DISPLAY_STATUS.ASISTIRA]: 'Asistirá',
  [DISPLAY_STATUS.NO_ASISTIRA]: 'No asistirá',
  [DISPLAY_STATUS.SIN_PREDICCION]: 'Sin predicción',
}

export const DISPLAY_STATUS_COLORS = {
  [DISPLAY_STATUS.ASISTIDA]: '#0f766e',
  [DISPLAY_STATUS.NO_ASISTIO]: '#ea580c',
  [DISPLAY_STATUS.ASISTIRA]: '#2563eb',
  [DISPLAY_STATUS.NO_ASISTIRA]: '#dc2626',
  [DISPLAY_STATUS.SIN_PREDICCION]: '#64748b',
}

export const STATUS_FILTER_OPTIONS = [
  { value: 'all', label: 'Todos los estados' },
  { value: DISPLAY_STATUS.ASISTIRA, label: 'Asistirá' },
  { value: DISPLAY_STATUS.NO_ASISTIRA, label: 'No asistirá' },
  { value: DISPLAY_STATUS.ASISTIDA, label: 'Asistida' },
  { value: DISPLAY_STATUS.NO_ASISTIO, label: 'No asistió' },
  { value: DISPLAY_STATUS.SIN_PREDICCION, label: 'Sin predicción' },
]

export const getFinalPredictionLabel = (prediction) => (
  prediction?.final_label ?? prediction?.predicted_label ?? prediction?.model_label
)

/** Resolve the four business statuses (+ sin predicción for pending citas). */
export const getDisplayStatus = (appointment, prediction = null) => {
  const type = Number(appointment?.appointment_type)

  if (type === APPOINTMENT_TYPE.ASISTIDA) return DISPLAY_STATUS.ASISTIDA
  if (type === APPOINTMENT_TYPE.NO_ASISTIO) return DISPLAY_STATUS.NO_ASISTIO

  const finalLabel = getFinalPredictionLabel(prediction)
  if (finalLabel === 1) return DISPLAY_STATUS.NO_ASISTIRA
  if (finalLabel === 0) return DISPLAY_STATUS.ASISTIRA
  return DISPLAY_STATUS.SIN_PREDICCION
}

export const getDisplayStatusLabel = (appointment, prediction = null) => (
  DISPLAY_STATUS_LABELS[getDisplayStatus(appointment, prediction)] ?? 'Sin estado'
)

export const getDisplayStatusColor = (appointment, prediction = null) => (
  DISPLAY_STATUS_COLORS[getDisplayStatus(appointment, prediction)] ?? DISPLAY_STATUS_COLORS[DISPLAY_STATUS.SIN_PREDICCION]
)

export const matchesStatusFilter = (appointment, prediction, statusFilter) => {
  if (statusFilter === 'all') return true
  return getDisplayStatus(appointment, prediction) === statusFilter
}

/** Outcome options when closing a scheduled cita (type 2 → 0 or 1). */
export const OUTCOME_STATUS_OPTIONS = [
  { value: APPOINTMENT_TYPE.ASISTIDA, label: 'Asistida' },
  { value: APPOINTMENT_TYPE.NO_ASISTIO, label: 'No asistió' },
  { value: APPOINTMENT_TYPE.EN_ESPERA, label: 'En espera' },
]
