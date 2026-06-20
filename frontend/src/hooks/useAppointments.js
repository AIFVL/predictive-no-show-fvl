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
  const [loading, setLoading] = useState(false)

  const fetchPredictionsWaiting = useCallback(async (medicId = '', days = dateWindow) => {
    try {
      const response = await fetch(predictionsWaitingEndpoint(medicId, days))
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
  }, [dateWindow])

  const fetchAppointments = useCallback(async (medicId = '', days = dateWindow) => {
    setLoading(true)
    try {
      const response = await fetch(appointmentsEndpoint(medicId, days))
      if (!response.ok) {
        throw new Error('No fue posible cargar las citas.')
      }

      const data = await response.json()
      setAppointments(Array.isArray(data) ? data : [])
      setActiveMedicFilter(medicId)
      await fetchPredictionsWaiting(medicId, days)
    } catch (error) {
      console.error('Error loading appointments', error)
      alert(`Error consultando citas: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }, [dateWindow, fetchPredictionsWaiting])

  useEffect(() => {
    fetchAppointments('', dateWindow)
  }, [dateWindow]) // eslint-disable-line react-hooks/exhaustive-deps

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

    await fetchAppointments(activeMedicFilter, dateWindow)
  }, [activeMedicFilter, dateWindow, fetchAppointments])

  const deleteAppointment = useCallback(async (appointmentId) => {
    const response = await fetch(`${API_BASE_URL}/appointments/${appointmentId}`, { method: 'DELETE' })
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || 'No fue posible eliminar la cita.')
    }
    await fetchAppointments(activeMedicFilter, dateWindow)
  }, [activeMedicFilter, dateWindow, fetchAppointments])

  const updateAppointmentType = useCallback(async (appointmentId, newType) => {
    const response = await fetch(updateAppointmentTypeEndpoint(appointmentId, newType), { method: 'PATCH' })
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || 'No fue posible actualizar el estado.')
    }
    const updated = await response.json()
    await fetchAppointments(activeMedicFilter, dateWindow)
    return updated
  }, [activeMedicFilter, dateWindow, fetchAppointments])

  return {
    appointments,
    predictionsMap,
    summary,
    activeMedicFilter,
    dateWindow,
    setDateWindow,
    loading,
    fetchAppointments,
    fetchAppointmentDetail,
    createAppointment,
    deleteAppointment,
    updateAppointmentType,
  }
}
