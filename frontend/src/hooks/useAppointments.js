import { useCallback, useEffect, useState } from 'react'
import {
  API_BASE_URL,
  DEFAULT_DATE_WINDOW,
  appointmentsEndpoint,
  appointmentInfoEndpoint,
  appointmentPredictionEndpoint,
  predictionsWaitingEndpoint,
  updateAppointmentTypeEndpoint,
} from '@fvl/shared'

export function useAppointments(initialDays = DEFAULT_DATE_WINDOW) {
  const [appointments, setAppointments] = useState([])
  const [predictionsMap, setPredictionsMap] = useState({})
  const [summary, setSummary] = useState(null)
  const [activeMedicFilter, setActiveMedicFilter] = useState('')
  const [dateWindow, setDateWindow] = useState(initialDays)
  const [predictionReferenceDate, setPredictionReferenceDate] = useState('')
  const [loading, setLoading] = useState(false)

  const fetchPredictionsWaiting = useCallback(async (
    medicId = '',
    days = dateWindow,
    referenceDate = predictionReferenceDate,
  ) => {
    try {
      const response = await fetch(predictionsWaitingEndpoint(medicId, days, referenceDate))
      if (!response.ok) {
        console.warn('No fue posible obtener predicciones para citas en espera.')
        setSummary(null)
        return
      }

      const data = await response.json()
      const nextMap = {}
      for (const item of (data.per_appointment || [])) {
        if (item?.appointment_id != null) {
          nextMap[String(item.appointment_id)] = item
        }
      }

      setPredictionsMap(nextMap)
      setSummary(data)
    } catch (error) {
      console.error('Error loading waiting predictions', error)
      setSummary(null)
    }
  }, [dateWindow, predictionReferenceDate])

  const fetchAppointments = useCallback(async (
    medicId = '',
    days = dateWindow,
    referenceDate = predictionReferenceDate,
  ) => {
    setLoading(true)
    try {
      const response = await fetch(appointmentsEndpoint(medicId, 0))
      if (!response.ok) {
        throw new Error('No fue posible cargar las citas.')
      }

      const data = await response.json()
      setAppointments(Array.isArray(data) ? data : [])
      setActiveMedicFilter(medicId)
      await fetchPredictionsWaiting(medicId, days, referenceDate)
    } catch (error) {
      console.error('Error loading appointments', error)
      alert(`Error consultando citas: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }, [dateWindow, fetchPredictionsWaiting, predictionReferenceDate])

  useEffect(() => {
    fetchAppointments(activeMedicFilter, dateWindow, predictionReferenceDate)
  }, [dateWindow, predictionReferenceDate]) // eslint-disable-line react-hooks/exhaustive-deps

  const fetchAppointmentDetail = useCallback(async (appointmentId) => {
    const [appointmentResponse, predictionResponse] = await Promise.all([
      fetch(appointmentInfoEndpoint(appointmentId)),
      fetch(appointmentPredictionEndpoint(appointmentId)),
    ])

    if (!appointmentResponse.ok) {
      const message = await appointmentResponse.text()
      throw new Error(message || 'No fue posible cargar el detalle.')
    }

    const appointment = await appointmentResponse.json()
    let prediction = null
    if (predictionResponse.ok) {
      prediction = await predictionResponse.json()
      setPredictionsMap((current) => ({
        ...current,
        [String(appointmentId)]: prediction,
      }))
    }

    return { appointment, prediction }
  }, [])

  const createAppointment = useCallback(async (payload) => {
    const response = await fetch(`${API_BASE_URL}/appointments/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || 'No fue posible crear la cita.')
    }

    await fetchAppointments(activeMedicFilter, dateWindow, predictionReferenceDate)
  }, [activeMedicFilter, dateWindow, fetchAppointments, predictionReferenceDate])

  const deleteAppointment = useCallback(async (appointmentId) => {
    const response = await fetch(`${API_BASE_URL}/appointments/${appointmentId}`, { method: 'DELETE' })
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || 'No fue posible eliminar la cita.')
    }
    await fetchAppointments(activeMedicFilter, dateWindow, predictionReferenceDate)
  }, [activeMedicFilter, dateWindow, fetchAppointments, predictionReferenceDate])

  const updateAppointmentType = useCallback(async (appointmentId, newType) => {
    let previousAppointments = []
    let previousPredictionsMap = {}

    setAppointments((current) => {
      previousAppointments = current
      return current.map((appointment) => (
        String(appointment.id) === String(appointmentId)
          ? { ...appointment, appointment_type: Number(newType) }
          : appointment
      ))
    })

    if (Number(newType) !== 2) {
      setPredictionsMap((current) => {
        previousPredictionsMap = current
        const next = { ...current }
        delete next[String(appointmentId)]
        return next
      })
    }

    const response = await fetch(updateAppointmentTypeEndpoint(appointmentId, newType), { method: 'PATCH' })
    if (!response.ok) {
      setAppointments(previousAppointments)
      if (Number(newType) !== 2) setPredictionsMap(previousPredictionsMap)
      const message = await response.text()
      throw new Error(message || 'No fue posible actualizar el estado.')
    }

    const updated = await response.json()
    setAppointments((current) => current.map((appointment) => (
      String(appointment.id) === String(updated.id) ? updated : appointment
    )))
    fetchAppointments(activeMedicFilter, dateWindow, predictionReferenceDate).catch((error) => {
      console.error('Error refreshing appointments after status update', error)
    })
    return updated
  }, [activeMedicFilter, dateWindow, fetchAppointments, predictionReferenceDate])

  return {
    appointments,
    predictionsMap,
    summary,
    activeMedicFilter,
    dateWindow,
    setDateWindow,
    predictionReferenceDate,
    setPredictionReferenceDate,
    loading,
    fetchAppointments,
    fetchAppointmentDetail,
    createAppointment,
    deleteAppointment,
    updateAppointmentType,
  }
}
