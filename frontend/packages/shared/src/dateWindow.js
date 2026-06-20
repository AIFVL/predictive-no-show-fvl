import { CALENDAR_YEAR } from './constants.js'

export const buildAppointmentDate = (appointment) => (
  new Date(CALENDAR_YEAR, (appointment?.month || 1) - 1, appointment?.day || 1, appointment?.hour || 0)
)

export const isAppointmentTodayOrPast = (appointment) => {
  if (!appointment) return false
  const appointmentDate = new Date(CALENDAR_YEAR, (appointment.month || 1) - 1, appointment.day || 1)
  const today = new Date()
  const currentDate = new Date(today.getFullYear(), today.getMonth(), today.getDate())
  return appointmentDate.getTime() <= currentDate.getTime()
}

export const formatDateLabel = (appointment, monthLabels) => {
  if (!appointment) return 'Sin fecha'
  return `${appointment.day} de ${monthLabels[(appointment.month || 1) - 1]} de ${CALENDAR_YEAR}`
}

export const formatTimeLabel = (hour) => `${String(hour ?? 0).padStart(2, '0')}:00`

export const isWithinForwardWindow = (appointment, days, reference = new Date()) => {
  if (!days) return true
  const ref = new Date(reference.getFullYear(), reference.getMonth(), reference.getDate())
  const end = new Date(ref)
  end.setDate(end.getDate() + Number(days))
  const appt = new Date(CALENDAR_YEAR, (appointment.month || 1) - 1, appointment.day || 1)
  return appt >= ref && appt <= end
}
