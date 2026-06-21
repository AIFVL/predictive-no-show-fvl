import { API_BASE_URL, DEFAULT_DATE_WINDOW } from './constants.js'

const buildQuery = ({ medicId = '', days = DEFAULT_DATE_WINDOW, referenceDate = '' } = {}) => {
  const params = new URLSearchParams()
  if (medicId) params.set('medic_id', medicId)
  if (days != null) params.set('days', String(days))
  if (referenceDate) params.set('reference_date', referenceDate)
  const query = params.toString()
  return query ? `?${query}` : ''
}

export const appointmentsEndpoint = (medicId = '', days = DEFAULT_DATE_WINDOW) => {
  const query = buildQuery({ days })
  return medicId
    ? `${API_BASE_URL}/appointments/${encodeURIComponent(medicId)}${query}`
    : `${API_BASE_URL}/appointments/${query}`
}

export const predictionsWaitingEndpoint = (medicId = '', days = DEFAULT_DATE_WINDOW, referenceDate = '') => (
  `${API_BASE_URL}/predictions/waiting${buildQuery({ medicId, days, referenceDate })}`
)

export const appointmentInfoEndpoint = (appointmentId) => (
  `${API_BASE_URL}/appointments/info/${appointmentId}`
)

export const appointmentPredictionEndpoint = (appointmentId) => (
  `${API_BASE_URL}/predictions/appointment/${appointmentId}`
)

export const updateAppointmentTypeEndpoint = (appointmentId, appointmentType) => (
  `${API_BASE_URL}/appointments/type/${appointmentId}?appointment_type=${encodeURIComponent(appointmentType)}`
)
