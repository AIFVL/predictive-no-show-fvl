export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export const CALENDAR_YEAR = 2026

export const MONTH_LABELS = [
  'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
  'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
]

export const DEFAULT_DATE_WINDOW = 8

export const DATE_WINDOW_OPTIONS = [
  { value: 8, label: 'Próximos 8 días' },
  { value: 15, label: 'Próximos 15 días' },
  { value: 30, label: 'Próximo mes' },
]

export const RISK_COPY = {
  high: { label: 'Riesgo alto', tone: 'critical' },
  medium: { label: 'Seguimiento', tone: 'warning' },
  low: { label: 'Asistencia probable', tone: 'positive' },
  none: { label: 'Sin analítica', tone: 'neutral' },
}

export const emptyAppointmentForm = {
  medic_id: '',
  patient_id: '',
  hour: 9,
  day: 1,
  month: 1,
  search: '',
}
