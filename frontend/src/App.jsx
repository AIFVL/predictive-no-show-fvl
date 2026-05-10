import { useEffect, useMemo, useRef, useState } from 'react'
import FullCalendar from '@fullcalendar/react'
import dayGridPlugin from '@fullcalendar/daygrid'
import timeGridPlugin from '@fullcalendar/timegrid'
import interactionPlugin from '@fullcalendar/interaction'
import esLocale from '@fullcalendar/core/locales/es'
import logoFvl from '../logoFVL.png'

const API_BASE_URL = 'http://localhost:8000'
const CALENDAR_YEAR = 2026

const TYPE_LABELS = {
  0: 'Asistida',
  1: 'No asistió',
  2: 'En espera',
}

const TYPE_COLORS = {
  0: '#0f766e',
  1: '#b42318',
  2: '#2563eb',
}

const MONTH_LABELS = [
  'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
  'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre',
]

const RISK_COPY = {
  high: { label: 'Riesgo alto', tone: 'critical' },
  medium: { label: 'Seguimiento', tone: 'warning' },
  low: { label: 'Asistencia probable', tone: 'positive' },
  none: { label: 'Sin analítica', tone: 'neutral' },
}

const MANUAL_VERIFICATION_RULES = {
  non_attendance: {
    title: 'Checklist manual de inasistencia',
    minScore: 4,
    items: [
      {
        id: 'NA1',
        label: 'Historial de inasistencias alto',
        condition: 'Previous Non-Attendance >= 2',
        rationale: 'Fuerte predictor de comportamiento futuro',
        weight: 2,
      },
      {
        id: 'NA2',
        label: 'Alta tasa de no-show',
        condition: 'Prev_NoShow_Rate > 0.5',
        rationale: 'Indica tendencia dominante a faltar',
        weight: 2,
      },
      {
        id: 'NA3',
        label: 'Baja experiencia con citas',
        condition: 'Prev_Total <= 2',
        rationale: 'Paciente sin hábito en el sistema',
        weight: 1,
      },
      {
        id: 'NA4',
        label: 'Última cita fue no-show',
        condition: 'Last_Attendance = No-Show',
        rationale: 'Predictor reciente muy fuerte',
        weight: 2,
      },
      {
        id: 'NA5',
        label: 'Intervalo largo de asignación',
        condition: 'Creation-Assignment > 7 días',
        rationale: 'Mayor probabilidad de olvido',
        weight: 1,
      },
      {
        id: 'NA6',
        label: 'Cita con poca anticipación',
        condition: 'Creation-Assignment <= 2 días',
        rationale: 'Conflictos de agenda o falta de preparación',
        weight: 1,
      },
      {
        id: 'NA7',
        label: 'Paciente joven + bajo compromiso',
        condition: 'Age < 30 AND Prev_Attendance <= 1',
        rationale: 'Menor adherencia al control médico',
        weight: 1,
      },
      {
        id: 'NA8',
        label: 'Baja carga clínica + bajo compromiso',
        condition: 'Diseases <= 1 AND Prev_Attendance <= 1',
        rationale: 'Menor percepción de necesidad médica',
        weight: 1,
      },
    ],
  },
  attendance: {
    title: 'Checklist manual de asistencia',
    minScore: 4,
    items: [
      {
        id: 'A1',
        label: 'Historial alto de asistencia',
        condition: 'Previous Attendance >= 3',
        rationale: 'Comportamiento consistente positivo',
        weight: 2,
      },
      {
        id: 'A2',
        label: 'Baja tasa de no-show',
        condition: 'Prev_NoShow_Rate <= 0.2',
        rationale: 'Indica alta adherencia',
        weight: 2,
      },
      {
        id: 'A3',
        label: 'Última cita fue asistida',
        condition: 'Last_Attendance = Show',
        rationale: 'Fuerte predictor reciente',
        weight: 2,
      },
      {
        id: 'A4',
        label: 'Alta experiencia con citas',
        condition: 'Prev_Total >= 4',
        rationale: 'Paciente acostumbrado al sistema',
        weight: 1,
      },
      {
        id: 'A5',
        label: 'Intervalo moderado',
        condition: '3 <= Creation-Assignment <= 7 días',
        rationale: 'Balance entre preparación y olvido',
        weight: 1,
      },
      {
        id: 'A6',
        label: 'Paciente con mayor carga clínica',
        condition: 'Diseases >= 2',
        rationale: 'Mayor percepción de necesidad médica',
        weight: 1,
      },
      {
        id: 'A7',
        label: 'Mayor consumo de medicamentos',
        condition: 'Medications >= 2',
        rationale: 'Mayor compromiso terapéutico',
        weight: 1,
      },
      {
        id: 'A8',
        label: 'Baja carga clínica + bajo compromiso',
        condition: 'Diseases <= 1 AND Prev_Attendance <= 1',
        rationale: 'Mayor adherencia al seguimiento médico',
        weight: 1,
      },
    ],
  },
}

const emptyForm = {
  medic_id: '',
  patient_id: '',
  hour: 9,
  day: 1,
  month: 1,
  search: '',
}

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value)))

const formatPercent = (value) => {
  if (value == null || Number.isNaN(Number(value))) return 'Sin dato'
  return `${(Number(value) * 100).toFixed(1)}%`
}

const formatDateLabel = (appointment) => {
  if (!appointment) return 'Sin fecha'
  return `${appointment.day} de ${MONTH_LABELS[(appointment.month || 1) - 1]} de ${CALENDAR_YEAR}`
}

const formatTimeLabel = (hour) => `${String(hour ?? 0).padStart(2, '0')}:00`

const getProbAttendFromPrediction = (prediction) => {
  if (!prediction) return null
  if (prediction.model_analysis?.probability_attend != null) return clamp01(prediction.model_analysis.probability_attend)
  if (prediction.prob_attend != null) return clamp01(prediction.prob_attend)
  if (prediction.prob_no_show != null) return clamp01(1 - Number(prediction.prob_no_show))
  if (prediction.probability != null) return clamp01(1 - Number(prediction.probability))
  return null
}

const getProbNoShowFromPrediction = (prediction) => {
  if (!prediction) return null
  if (prediction.model_analysis?.probability_no_show != null) return clamp01(prediction.model_analysis.probability_no_show)
  if (prediction.prob_no_show != null) return clamp01(prediction.prob_no_show)
  if (prediction.probability != null) return clamp01(prediction.probability)
  if (prediction.prob_attend != null) return clamp01(1 - Number(prediction.prob_attend))
  return null
}

const getPredictionProbabilitySummary = (prediction) => {
  const finalLabel = prediction?.final_label ?? prediction?.predicted_label ?? prediction?.model_label
  const probAttend = getProbAttendFromPrediction(prediction)
  const probNoShow = getProbNoShowFromPrediction(prediction)

  if (!prediction || (probAttend == null && probNoShow == null)) {
    return {
      label: 'Sin probabilidad',
      probability: null,
      probAttend,
      probNoShow,
    }
  }

  if (finalLabel === 1) {
    return {
      label: 'No asistirá',
      probability: probNoShow,
      probAttend,
      probNoShow,
    }
  }

  return {
    label: 'Asistirá',
    probability: probAttend,
    probAttend,
    probNoShow,
  }
}

const getManualGroupMaxScore = (groupKey) => (
  MANUAL_VERIFICATION_RULES[groupKey].items.reduce((total, item) => total + item.weight, 0)
)

const getAdjustedProbabilitySummary = (prediction, manualAttendanceScore, manualNonAttendanceScore) => {
  const base = getPredictionProbabilitySummary(prediction)
  const hasManualChecks = manualAttendanceScore > 0 || manualNonAttendanceScore > 0

  if (!prediction || !hasManualChecks) {
    return {
      ...base,
      adjusted: false,
      modelProbAttend: base.probAttend,
      modelProbNoShow: base.probNoShow,
    }
  }

  const attendanceMax = getManualGroupMaxScore('attendance')
  const nonAttendanceMax = getManualGroupMaxScore('non_attendance')
  const attendanceStrength = attendanceMax > 0 ? manualAttendanceScore / attendanceMax : 0
  const nonAttendanceStrength = nonAttendanceMax > 0 ? manualNonAttendanceScore / nonAttendanceMax : 0
  const totalManualStrength = attendanceStrength + nonAttendanceStrength

  const modelProbNoShow = base.probNoShow ?? 0.5
  const manualProbNoShow = totalManualStrength > 0
    ? nonAttendanceStrength / totalManualStrength
    : modelProbNoShow
  const manualWeight = Math.min(0.45, 0.15 + (0.3 * Math.max(attendanceStrength, nonAttendanceStrength)))
  const probNoShow = clamp01((modelProbNoShow * (1 - manualWeight)) + (manualProbNoShow * manualWeight))
  const probAttend = clamp01(1 - probNoShow)
  const label = probNoShow > probAttend ? 'No asistirá' : 'Asistirá'

  return {
    label,
    probability: label === 'No asistirá' ? probNoShow : probAttend,
    probAttend,
    probNoShow,
    adjusted: true,
    modelProbAttend: base.probAttend,
    modelProbNoShow: base.probNoShow,
    manualWeight,
  }
}

const formatFactorValue = (value) => {
  if (value == null || value === '') return 'Sin dato'
  if (typeof value === 'number' && Number.isFinite(value)) return Number(value).toFixed(2)
  return String(value)
}

const getRiskLevel = (prediction) => {
  const finalLabel = prediction?.final_label ?? prediction?.predicted_label ?? prediction?.model_label
  const verificationStatus = prediction?.verification_status
  const probAttend = getProbAttendFromPrediction(prediction)

  if (!prediction) return 'none'
  if (finalLabel === 1 || verificationStatus === 'confirmed_no_show') return 'high'
  if (verificationStatus?.includes('contradictory')) return 'medium'
  if (probAttend == null) return 'none'
  if (probAttend < 0.65) return 'high'
  if (probAttend < 0.82) return 'medium'
  return 'low'
}

const getRiskSummary = (prediction) => {
  const level = getRiskLevel(prediction)
  const base = RISK_COPY[level]

  if (level === 'high') {
    return {
      ...base,
      message: prediction?.verification_status === 'confirmed_no_show'
        ? 'La doble verificación confirma alta probabilidad de inasistencia.'
        : 'Conviene confirmar la cita y priorizar seguimiento telefónico.',
    }
  }

  if (level === 'medium') {
    return {
      ...base,
      message: 'Hay señales mixtas; revisar antecedentes antes de la consulta.',
    }
  }

  if (level === 'low') {
    return {
      ...base,
      message: 'La combinación histórica sugiere asistencia esperada.',
    }
  }

  return {
    ...base,
    message: 'Aún no hay información analítica suficiente para clasificarla.',
  }
}

const getPredictionHeadline = (prediction) => {
  if (!prediction) {
    return {
      short: 'Sin predicción',
      long: 'Esta cita en espera aún no tiene predicción disponible.',
    }
  }

  const probAttend = getProbAttendFromPrediction(prediction)
  const finalLabel = prediction?.final_label ?? prediction?.predicted_label ?? prediction?.model_label

  if (probAttend == null) {
    return {
      short: 'Sin probabilidad',
      long: 'Hay registro de predicción, pero no se recibió una probabilidad interpretable.',
    }
  }

  if (finalLabel === 1) {
    return {
      short: 'No asistirá',
      long: 'Predicción final para esta cita programada: no asistirá.',
    }
  }

  return {
    short: 'Asistirá',
    long: 'Predicción final para esta cita programada: asistirá.',
  }
}

const getPredictionSourceLabel = (prediction) => {
  if (!prediction) return 'Sin fuente de predicción'
  if (prediction.feature_source === 'matched_dataset_row') return 'Fuente: cruce directo con dataset'
  if (prediction.feature_source === 'fallback_reference_profile') return 'Fuente: perfil de referencia del dataset'
  return 'Fuente: predicción disponible'
}

const getToneClass = (tone) => {
  if (tone === 'critical') return 'tone-pill tone-pill-critical'
  if (tone === 'warning') return 'tone-pill tone-pill-warning'
  if (tone === 'positive') return 'tone-pill tone-pill-positive'
  return 'tone-pill tone-pill-neutral'
}

const buildManualStateFromPrediction = (prediction) => {
  const state = {}
  const verificationRules = prediction?.verification?.rules || {}

  for (const groupKey of Object.keys(MANUAL_VERIFICATION_RULES)) {
    state[groupKey] = {}
    const autoChecks = verificationRules[groupKey]?.checks || {}
    for (const item of MANUAL_VERIFICATION_RULES[groupKey].items) {
      const autoTriggered = autoChecks[item.id]?.triggered
      state[groupKey][item.id] = autoTriggered === true
    }
  }

  return state
}

const scoreManualGroup = (groupKey, groupState) => {
  const config = MANUAL_VERIFICATION_RULES[groupKey]
  return config.items.reduce((total, item) => (
    groupState?.[item.id] ? total + item.weight : total
  ), 0)
}

const validateAppointmentForm = (nextForm) => {
  const errors = {}
  const medicId = String(nextForm.medic_id ?? '').trim()
  const patientId = String(nextForm.patient_id ?? '').trim()
  const hour = Number(nextForm.hour)
  const day = Number(nextForm.day)
  const month = Number(nextForm.month)

  if (!medicId) errors.medic_id = 'Ingresa el identificador del médico.'
  if (!patientId) errors.patient_id = 'Ingresa el identificador del paciente.'
  if (!Number.isFinite(hour) || hour < 0 || hour > 23) errors.hour = 'La hora debe estar entre 0 y 23.'
  if (!Number.isFinite(day) || day < 1 || day > 31) errors.day = 'El día debe estar entre 1 y 31.'
  if (!Number.isFinite(month) || month < 1 || month > 12) errors.month = 'El mes debe estar entre 1 y 12.'

  return errors
}

function App() {
  const [appointments, setAppointments] = useState([])
  const [predictionsMap, setPredictionsMap] = useState({})
  const [detailPrediction, setDetailPrediction] = useState(null)
  const [manualVerification, setManualVerification] = useState({
    non_attendance: {},
    attendance: {},
  })
  const [summary, setSummary] = useState(null)
  const [showForm, setShowForm] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [showSelectType, setShowSelectType] = useState(false)
  const [selectedAppt, setSelectedAppt] = useState(null)
  const [detailsTab, setDetailsTab] = useState('info')
  const [statusFilter, setStatusFilter] = useState('all')
  const [riskFilter, setRiskFilter] = useState('all')
  const [activeMedicFilter, setActiveMedicFilter] = useState('')
  const [form, setForm] = useState(emptyForm)
  const [formErrors, setFormErrors] = useState({})
  const calendarRef = useRef(null)

  const fetchPredictionsWaiting = async (medicId) => {
    try {
      const query = medicId ? `?medic_id=${encodeURIComponent(medicId)}` : ''
      const response = await fetch(`${API_BASE_URL}/predictions/waiting${query}`)
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
  }

  const fetchAppointments = async (medicId = '') => {
    try {
      const endpoint = medicId
        ? `${API_BASE_URL}/appointments/${encodeURIComponent(medicId)}`
        : `${API_BASE_URL}/appointments`
      const response = await fetch(endpoint)
      if (!response.ok) {
        throw new Error('No fue posible cargar las citas.')
      }

      const data = await response.json()
      const normalized = Array.isArray(data) ? data : []
      setAppointments(normalized)
      setActiveMedicFilter(medicId)
      await fetchPredictionsWaiting(medicId)
    } catch (error) {
      console.error('Error loading appointments', error)
      alert(`Error consultando citas: ${error.message}`)
    }
  }

  useEffect(() => {
    fetchAppointments()
  }, [])

  const handleSearch = async () => {
    const medicId = String(form.search ?? '').trim()
    await fetchAppointments(medicId)
  }

  const resetSearch = async () => {
    setForm((current) => ({ ...current, search: '' }))
    await fetchAppointments('')
  }

  const handleChange = (event) => {
    const { name, value } = event.target
    setForm((current) => ({
      ...current,
      [name]: ['hour', 'day', 'month'].includes(name) ? Number(value) : value,
    }))

    if (formErrors[name]) {
      setFormErrors((current) => {
        const copy = { ...current }
        delete copy[name]
        return copy
      })
    }
  }

  const handleSubmit = async (event) => {
    event.preventDefault()
    const normalized = {
      ...form,
      medic_id: String(form.medic_id ?? '').trim(),
      patient_id: String(form.patient_id ?? '').trim(),
    }

    const errors = validateAppointmentForm(normalized)
    setFormErrors(errors)
    if (Object.keys(errors).length > 0) return

    try {
      const response = await fetch(`${API_BASE_URL}/appointments/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(normalized),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'No fue posible crear la cita.')
      }

      await fetchAppointments(activeMedicFilter)
      setShowForm(false)
      setForm((current) => ({ ...emptyForm, search: current.search }))
      setFormErrors({})
    } catch (error) {
      console.error('Error creating appointment', error)
      alert(`Error creando la cita: ${error.message}`)
    }
  }

  const fetchAppointmentInfo = async (appointmentId) => {
    try {
      const [appointmentResponse, predictionResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/appointments/info/${appointmentId}`),
        fetch(`${API_BASE_URL}/predictions/appointment/${appointmentId}`),
      ])
      if (!appointmentResponse.ok) {
        const message = await appointmentResponse.text()
        throw new Error(message || 'No fue posible cargar el detalle.')
      }

      const data = await appointmentResponse.json()
      setSelectedAppt(data)
      if (predictionResponse.ok) {
        const predictionData = await predictionResponse.json()
        setDetailPrediction(predictionData)
        setManualVerification(buildManualStateFromPrediction(predictionData))
        setPredictionsMap((current) => ({
          ...current,
          [String(appointmentId)]: predictionData,
        }))
      } else {
        setDetailPrediction(null)
        setManualVerification(buildManualStateFromPrediction(null))
      }
      setShowSelectType(false)
      setDetailsTab('info')
      setShowDetails(true)
    } catch (error) {
      console.error('Error loading appointment detail', error)
      alert(`Error consultando la cita: ${error.message}`)
    }
  }

  const handleDelete = async (appointmentId) => {
    if (!window.confirm('¿Deseas eliminar esta cita del calendario?')) return

    try {
      const response = await fetch(`${API_BASE_URL}/appointments/${appointmentId}`, { method: 'DELETE' })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'No fue posible eliminar la cita.')
      }

      await fetchAppointments(activeMedicFilter)
      setShowDetails(false)
      setSelectedAppt(null)
    } catch (error) {
      console.error('Error deleting appointment', error)
      alert(`Error eliminando la cita: ${error.message}`)
    }
  }

  const handleChangeAppointmentType = async (appointmentId, newType) => {
    try {
      const response = await fetch(
        `${API_BASE_URL}/appointments/type/${appointmentId}?appointment_type=${encodeURIComponent(newType)}`,
        { method: 'PATCH' },
      )

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'No fue posible actualizar el estado.')
      }

      const updated = await response.json()
      setSelectedAppt(updated)
      await fetchAppointments(activeMedicFilter)
    } catch (error) {
      console.error('Error updating appointment type', error)
      alert(`Error actualizando la cita: ${error.message}`)
    }
  }

  const handleDateClick = (info) => {
    const api = calendarRef.current?.getApi()
    if (!api) return
    if (api.view?.type === 'dayGridMonth') {
      api.changeView('timeGridDay')
    }
    api.gotoDate(info.date)
  }

  const isAppointmentTodayOrPast = (appointment) => {
    if (!appointment) return false
    const appointmentDate = new Date(CALENDAR_YEAR, (appointment.month || 1) - 1, appointment.day || 1)
    const today = new Date()
    const currentDate = new Date(today.getFullYear(), today.getMonth(), today.getDate())
    return appointmentDate.getTime() <= currentDate.getTime()
  }

  const enrichedAppointments = useMemo(() => {
    return appointments.map((appointment) => {
      const prediction = predictionsMap[String(appointment.id)] || null
      const risk = getRiskSummary(prediction)
      const start = new Date(CALENDAR_YEAR, (appointment.month || 1) - 1, appointment.day || 1, appointment.hour || 0)

      return {
        ...appointment,
        prediction,
        risk,
        start,
        dateLabel: formatDateLabel(appointment),
        timeLabel: formatTimeLabel(appointment.hour),
      }
    })
  }, [appointments, predictionsMap])

  const filteredAppointments = useMemo(() => {
    return enrichedAppointments.filter((appointment) => {
      if (statusFilter !== 'all' && String(appointment.appointment_type) !== statusFilter) return false
      if (riskFilter !== 'all' && getRiskLevel(appointment.prediction) !== riskFilter) return false
      return true
    })
  }, [enrichedAppointments, riskFilter, statusFilter])

  const events = useMemo(() => {
    return filteredAppointments.map((appointment) => {
      const predictionHeadline = getPredictionHeadline(appointment.prediction)
      const finalLabel = appointment.prediction?.final_label ?? appointment.prediction?.predicted_label ?? appointment.prediction?.model_label

      let baseColor = TYPE_COLORS[appointment.appointment_type] || TYPE_COLORS[2]
      if (appointment.appointment_type === 2) {
        if (finalLabel === 1) {
          baseColor = '#dc2626'
        } else if (finalLabel === 0) {
          baseColor = '#2563eb'
        } else {
          baseColor = '#94a3b8'
        }
      }

      return {
        id: String(appointment.id),
        title: appointment.appointment_type === 2
          ? `${appointment.timeLabel} · ${appointment.patient_id} · ${predictionHeadline.short}`
          : `${appointment.timeLabel} · ${appointment.patient_id}`,
        start: appointment.start.toISOString(),
        backgroundColor: baseColor,
        borderColor: baseColor,
        textColor: '#ffffff',
        extendedProps: appointment,
      }
    })
  }, [filteredAppointments])

  const stats = useMemo(() => {
    const waiting = enrichedAppointments.filter((appointment) => appointment.appointment_type === 2)
    const highRisk = waiting.filter((appointment) => getRiskLevel(appointment.prediction) === 'high')
    const followUp = waiting.filter((appointment) => getRiskLevel(appointment.prediction) === 'medium')
    const resolved = enrichedAppointments.filter((appointment) => appointment.appointment_type !== 2)

    return [
      {
        label: 'Citas programadas',
        value: enrichedAppointments.length,
        caption: activeMedicFilter ? `Médico ${activeMedicFilter}` : 'Vista general institucional',
      },
      {
        label: 'En espera',
        value: waiting.length,
        caption: `${summary?.analyzed ?? 0} con analítica disponible`,
      },
      {
        label: 'Riesgo alto',
        value: highRisk.length,
        caption: 'Prioridad para confirmación de asistencia',
      },
      {
        label: 'Citas resueltas',
        value: resolved.length,
        caption: `${followUp.length} citas con seguimiento recomendado`,
      },
    ]
  }, [activeMedicFilter, enrichedAppointments, summary])

  const priorityList = useMemo(() => {
    return [...filteredAppointments]
      .filter((appointment) => appointment.appointment_type === 2)
      .sort((left, right) => {
        const severity = { high: 0, medium: 1, low: 2, none: 3 }
        const byRisk = severity[getRiskLevel(left.prediction)] - severity[getRiskLevel(right.prediction)]
        if (byRisk !== 0) return byRisk
        return left.start.getTime() - right.start.getTime()
      })
      .slice(0, 6)
  }, [filteredAppointments])

  const selectedPrediction = detailPrediction ?? (selectedAppt ? predictionsMap[String(selectedAppt.id)] : null)
  const selectedRisk = getRiskSummary(selectedPrediction)
  const selectedPredictionHeadline = getPredictionHeadline(selectedPrediction)
  const selectedModelAnalysis = selectedPrediction?.model_analysis ?? selectedPrediction?.shap_analysis ?? null
  const selectedTopFactors = selectedModelAnalysis?.top_factors?.slice(0, 3) || []
  const canChangeState = selectedAppt && selectedAppt.appointment_type === 2 && isAppointmentTodayOrPast(selectedAppt)
  const isSelectedWaiting = selectedAppt?.appointment_type === 2
  const manualAttendanceScore = scoreManualGroup('attendance', manualVerification.attendance)
  const manualNonAttendanceScore = scoreManualGroup('non_attendance', manualVerification.non_attendance)
  const selectedProbability = getAdjustedProbabilitySummary(
    selectedPrediction,
    manualAttendanceScore,
    manualNonAttendanceScore,
  )

  const manualVerificationSummary = useMemo(() => {
    const attendanceMin = MANUAL_VERIFICATION_RULES.attendance.minScore
    const nonAttendanceMin = MANUAL_VERIFICATION_RULES.non_attendance.minScore

    if (manualAttendanceScore >= attendanceMin && manualNonAttendanceScore < nonAttendanceMin) {
      return 'La revisión manual favorece asistencia.'
    }
    if (manualNonAttendanceScore >= nonAttendanceMin && manualAttendanceScore < attendanceMin) {
      return 'La revisión manual favorece inasistencia.'
    }
    if (manualAttendanceScore >= attendanceMin && manualNonAttendanceScore >= nonAttendanceMin) {
      return 'La revisión manual tiene señales contradictorias.'
    }
    return 'La revisión manual aún no confirma asistencia ni inasistencia.'
  }, [manualAttendanceScore, manualNonAttendanceScore])

  return (
    <div className="app-shell">
      <div className="app-backdrop" />
      <main className="app-layout">
        <section className="hero-card">
          <div className="hero-topbar">
            <div className="brand-lockup">
              <img className="hero-logo" src={logoFvl} alt="Fundación Valle del Lili" />
              <div className="brand-copy">
                <p className="eyebrow">Institución</p>
                <p className="brand-name">Fundación Valle del Lili</p>
              </div>
            </div>
          </div>

          <div className="hero-brand">
            <div className="hero-title-block">
              <h1>Prediccion de inasistencia de citas para medicina interna</h1>
              <p className="hero-copy">
                Este calendario concentra la programación 2026, identifica citas en espera, prioriza pacientes con mayor riesgo de inasistencia y facilita una gestión más oportuna para el seguimiento de medicina interna.
              </p>
            </div>
          </div>

          <div className="hero-actions">
            <div className="search-box">
              <label htmlFor="search">Buscar por médico</label>
              <div className="search-row">
                <input
                  id="search"
                  name="search"
                  value={form.search || ''}
                  onChange={handleChange}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      handleSearch()
                    }
                  }}
                  placeholder="Ej. MED-014"
                />
                <button className="primary-button" type="button" onClick={handleSearch}>
                  Filtrar agenda
                </button>
              </div>
              <button className="ghost-button" type="button" onClick={resetSearch}>
                Restablecer vista institucional
              </button>
            </div>

            <div className="legend-card">
              <p className="legend-title">Cómo leer el calendario</p>
              <div className="legend-grid">
                <span><i className="legend-dot legend-dot-blue" /> Asistirá</span>
                <span><i className="legend-dot legend-dot-red" /> No asistirá</span>
                <span><i className="legend-dot legend-dot-green" /> Asistida</span>
                <span><i className="legend-dot legend-dot-gray" /> Sin predicción</span>
              </div>
            </div>
          </div>
        </section>

        <section className="stats-grid">
          {stats.map((item) => (
            <article className="stat-card" key={item.label}>
              <p>{item.label}</p>
              <strong>{item.value}</strong>
              <span>{item.caption}</span>
            </article>
          ))}
        </section>

        <section className="workspace-grid">
          <div className="calendar-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Panel operativo</p>
                <h2>Calendario asistencial interactivo</h2>
              </div>

              <div className="filter-row">
                <select className="compact-filter" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                  <option value="all">Todos los estados</option>
                  <option value="2">En espera</option>
                  <option value="0">Asistidas</option>
                  <option value="1">No asistieron</option>
                </select>
                <select className="compact-filter" value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
                  <option value="all">Todos los riesgos</option>
                  <option value="high">Riesgo alto</option>
                  <option value="medium">Seguimiento</option>
                  <option value="low">Asistencia probable</option>
                  <option value="none">Sin analítica</option>
                </select>
                <button className="primary-button" type="button" onClick={() => setShowForm((value) => !value)}>
                  {showForm ? 'Cerrar formulario' : 'Registrar cita'}
                </button>
              </div>
            </div>

            {showForm && (
              <form className="appointment-form" onSubmit={handleSubmit}>
                <div className="form-grid">
                  <label>
                    <span>Médico ID</span>
                    <input name="medic_id" value={form.medic_id} onChange={handleChange} />
                    {formErrors.medic_id && <small>{formErrors.medic_id}</small>}
                  </label>
                  <label>
                    <span>Paciente ID</span>
                    <input name="patient_id" value={form.patient_id} onChange={handleChange} />
                    {formErrors.patient_id && <small>{formErrors.patient_id}</small>}
                  </label>
                  <label>
                    <span>Hora</span>
                    <input name="hour" type="number" min="0" max="23" value={form.hour} onChange={handleChange} />
                    {formErrors.hour && <small>{formErrors.hour}</small>}
                  </label>
                  <label>
                    <span>Día</span>
                    <input name="day" type="number" min="1" max="31" value={form.day} onChange={handleChange} />
                    {formErrors.day && <small>{formErrors.day}</small>}
                  </label>
                  <label>
                    <span>Mes</span>
                    <input name="month" type="number" min="1" max="12" value={form.month} onChange={handleChange} />
                    {formErrors.month && <small>{formErrors.month}</small>}
                  </label>
                </div>
                <div className="form-actions">
                  <button className="primary-button" type="submit">Guardar cita</button>
                  <button className="ghost-button" type="button" onClick={() => setShowForm(false)}>Cancelar</button>
                </div>
              </form>
            )}

            <div className="calendar-card">
              <FullCalendar
                ref={calendarRef}
                plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin]}
                initialView="dayGridMonth"
                locale={esLocale}
                headerToolbar={{ left: 'prev,next today', center: 'title', right: 'dayGridMonth,timeGridWeek,timeGridDay' }}
                events={events}
                height="auto"
                dateClick={handleDateClick}
                eventClick={(info) => fetchAppointmentInfo(info.event.id)}
                eventTimeFormat={{ hour: '2-digit', minute: '2-digit', meridiem: false }}
                dayMaxEvents={3}
                buttonText={{ today: 'Hoy', month: 'Mes', week: 'Semana', day: 'Día' }}
              />
            </div>
          </div>

          <aside className="insights-panel">
            <div className="insight-card">
              <p className="eyebrow">Lectura del calendario</p>
              <h3>¿Qué gestiona este tablero?</h3>
              <p>
                Las citas se registran inicialmente como <strong>En espera</strong>. A partir del historial clínico
                y del modelo predictivo, el panel clasifica si una cita programada <strong>asistirá</strong> o
                <strong> no asistirá</strong> para orientar llamadas, recordatorios y cierre posterior en asistida o
                no asistió. Si una cita no encuentra datos del paciente en el dataset procesado, se mostrará como
                <strong> Sin predicción</strong>.
              </p>
            </div>

            <div className="insight-card">
              <div className="section-head">
                <h3>Prioridades del día</h3>
                <span>{priorityList.length} casos</span>
              </div>
              <div className="priority-list">
                {priorityList.length === 0 && <p className="empty-state">No hay citas en espera con el filtro actual.</p>}
                {priorityList.map((appointment) => (
                  <button
                    className="priority-item"
                    key={appointment.id}
                    type="button"
                    onClick={() => fetchAppointmentInfo(appointment.id)}
                  >
                    <div>
                      <strong>{appointment.patient_id}</strong>
                      <span>{appointment.dateLabel} · {appointment.timeLabel}</span>
                      <span>{getPredictionHeadline(appointment.prediction).long}</span>
                      <span>{getPredictionSourceLabel(appointment.prediction)}</span>
                    </div>
                    <span className={getToneClass(appointment.risk.tone)}>{appointment.risk.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="insight-card">
              <div className="section-head">
                <h3>Resumen predictivo</h3>
                <span>Modelo activo</span>
              </div>
              <div className="model-summary">
                <strong>{summary?.analyzed ?? 0} citas analizadas</strong>
                <p>Las predicciones se aplican sobre citas programadas que siguen en estado En espera.</p>
                <p>Si una cita ya figura como Asistida o No asistió, el sistema la toma como resultado real y no como predicción.</p>
                <p>Si no hay cruce con el dataset clínico procesado, se mostrará Sin predicción.</p>
              </div>
            </div>
          </aside>
        </section>

        {showDetails && selectedAppt && (
          <div
            className="modal-overlay"
            onClick={() => {
              setShowDetails(false)
              setShowSelectType(false)
              setDetailsTab('info')
              setDetailPrediction(null)
            }}
          >
            <section className="modal-card" onClick={(event) => event.stopPropagation()}>
              <div className="modal-top">
                <div>
                  <p className="eyebrow">Detalle de cita</p>
                  <h2>Paciente {selectedAppt.patient_id}</h2>
                  <p>{formatDateLabel(selectedAppt)} · {formatTimeLabel(selectedAppt.hour)}</p>
                </div>
                <span className={getToneClass(isSelectedWaiting ? selectedRisk.tone : 'neutral')}>
                  {isSelectedWaiting ? selectedRisk.label : TYPE_LABELS[selectedAppt.appointment_type] ?? 'Sin estado'}
                </span>
              </div>

              <div className="tab-row">
                <button type="button" className={detailsTab === 'info' ? 'tab-button active' : 'tab-button'} onClick={() => setDetailsTab('info')}>
                  Información
                </button>
                <button
                  type="button"
                  className={detailsTab === 'verification' ? 'tab-button active' : 'tab-button'}
                  onClick={() => setDetailsTab('verification')}
                >
                  Doble verificación
                </button>
              </div>

              {detailsTab === 'info' && (
                <div className="details-grid">
                  <div className="detail-box">
                    <span>Médico</span>
                    <strong>{selectedAppt.medic_id}</strong>
                  </div>
                  <div className="detail-box">
                    <span>Estado</span>
                    <strong>{TYPE_LABELS[selectedAppt.appointment_type] ?? 'Sin estado'}</strong>
                  </div>
                  <div className="detail-box">
                    <span>Creada</span>
                    <strong>{selectedAppt.created_at ? new Date(selectedAppt.created_at).toLocaleString('es-CO') : 'Sin fecha'}</strong>
                  </div>
                  {isSelectedWaiting ? (
                    <>
                      <div className="detail-box">
                        <span>Predicción de la cita</span>
                        <strong>{selectedPredictionHeadline.short}</strong>
                      </div>
                      <div className="detail-box probability-box detail-box-wide">
                        <span>{selectedProbability.adjusted ? 'Probabilidad ajustada por checklist' : 'Probabilidad automática'}</span>
                        <div className="probability-headline">
                          <strong>{selectedProbability.label}</strong>
                          <strong>{formatPercent(selectedProbability.probability)}</strong>
                        </div>
                        <div className="probability-meter" aria-hidden="true">
                          <i style={{ width: `${Math.round((selectedProbability.probability ?? 0) * 100)}%` }} />
                        </div>
                        <div className="probability-pair">
                          <span>Asistirá: {formatPercent(selectedProbability.probAttend)}</span>
                          <span>No asistirá: {formatPercent(selectedProbability.probNoShow)}</span>
                        </div>
                        {selectedProbability.adjusted && (
                          <div className="probability-pair">
                            <span>Modelo base asistencia: {formatPercent(selectedProbability.modelProbAttend)}</span>
                            <span>Modelo base inasistencia: {formatPercent(selectedProbability.modelProbNoShow)}</span>
                          </div>
                        )}
                        {selectedModelAnalysis && (
                          <div className="model-analysis">
                            <span>{selectedModelAnalysis.method_label || 'Análisis del modelo'}</span>
                            {selectedTopFactors.length > 0 && (
                              <div className="factor-list">
                                {selectedTopFactors.map((factor) => (
                                  <span key={factor.feature}>
                                    {factor.label}: {formatFactorValue(factor.value)}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                      <div className="detail-box">
                        <span>Origen de la predicción</span>
                        <strong>{getPredictionSourceLabel(selectedPrediction)}</strong>
                      </div>
                      <div className="detail-box detail-box-wide">
                        <span>Recomendación operativa</span>
                        <strong>{selectedRisk.message}</strong>
                      </div>
                    </>
                  ) : (
                    <div className="detail-box detail-box-wide">
                      <span>Resultado de la cita</span>
                      <strong>
                        {selectedAppt.appointment_type === 0
                          ? 'La cita ya fue registrada como asistida; no requiere predicción.'
                          : selectedAppt.appointment_type === 1
                            ? 'La cita ya fue registrada como no asistió; no requiere predicción.'
                            : 'La cita no tiene resultado final registrado.'}
                      </strong>
                    </div>
                  )}
                </div>
              )}

              {detailsTab === 'verification' && (
                <div className="verification-grid">
                  <div className="verification-card verification-card-highlight">
                    <div className="section-head">
                      <h3>Resumen manual</h3>
                      <span>{manualVerificationSummary}</span>
                    </div>
                    <p className="verification-copy">
                      Usa estos checks manuales para validar la cita antes de marcarla como asistida o no asistió.
                    </p>
                    <div className="verification-probability-panel">
                      <div className="probability-headline">
                        <strong>{selectedProbability.label}</strong>
                        <strong>{formatPercent(selectedProbability.probability)}</strong>
                      </div>
                      <div className="probability-meter" aria-hidden="true">
                        <i style={{ width: `${Math.round((selectedProbability.probability ?? 0) * 100)}%` }} />
                      </div>
                      <div className="probability-pair">
                        <span>Asistirá: {formatPercent(selectedProbability.probAttend)}</span>
                        <span>No asistirá: {formatPercent(selectedProbability.probNoShow)}</span>
                      </div>
                    </div>
                    <div className="manual-summary-grid">
                      <div className="manual-summary-box">
                        <strong>Asistencia</strong>
                        <span>{manualAttendanceScore}/{MANUAL_VERIFICATION_RULES.attendance.minScore} para confirmar</span>
                      </div>
                      <div className="manual-summary-box">
                        <strong>Inasistencia</strong>
                        <span>{manualNonAttendanceScore}/{MANUAL_VERIFICATION_RULES.non_attendance.minScore} para confirmar</span>
                      </div>
                    </div>
                  </div>

                  {selectedPrediction?.verification && (
                    <div className="verification-card">
                      <div className="section-head">
                        <h3>Validación automática del modelo</h3>
                        <span>{selectedPrediction?.verification_status || 'Sin estado'}</span>
                      </div>
                      <p className="verification-copy">
                        Esta capa automática evalúa reglas sobre la información analítica disponible para respaldar o contradecir la predicción del modelo.
                      </p>
                    </div>
                  )}

                  {['attendance', 'non_attendance'].map((groupKey) => {
                    const config = MANUAL_VERIFICATION_RULES[groupKey]
                    const autoGroup = selectedPrediction?.verification?.rules?.[groupKey]

                    return (
                      <div className="verification-card" key={groupKey}>
                        <div className="section-head">
                          <h3>{config.title}</h3>
                          <span>
                            Puntaje manual {scoreManualGroup(groupKey, manualVerification[groupKey])} · mínimo {config.minScore}
                          </span>
                        </div>
                        <div className="manual-checklist">
                          {config.items.map((item) => {
                            const autoCheck = autoGroup?.checks?.[item.id]
                            return (
                              <label className="manual-check-row" key={item.id}>
                                <input
                                  type="checkbox"
                                  checked={!!manualVerification[groupKey]?.[item.id]}
                                  onChange={(event) => {
                                    const checked = event.target.checked
                                    setManualVerification((current) => ({
                                      ...current,
                                      [groupKey]: {
                                        ...current[groupKey],
                                        [item.id]: checked,
                                      },
                                    }))
                                  }}
                                />
                                <div className="manual-check-content">
                                  <div className="manual-check-head">
                                    <strong>{item.label}</strong>
                                    <span>Peso {item.weight}</span>
                                  </div>
                                  <span className="manual-check-condition">{item.condition}</span>
                                  <span className="manual-check-rationale">{item.rationale}</span>
                                  <span className="manual-check-auto">
                                    Automático: {autoCheck?.triggered === true ? 'Cumple' : autoCheck?.triggered === false ? 'No cumple' : 'Sin dato'}
                                  </span>
                                </div>
                              </label>
                            )
                          })}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}

              <div className="modal-actions">
                <button
                  className="primary-button"
                  type="button"
                  disabled={!canChangeState}
                  onClick={() => {
                    if (canChangeState) setShowSelectType((value) => !value)
                  }}
                >
                  Cambiar estado
                </button>
                {showSelectType && canChangeState && (
                  <select
                    className="status-select"
                    defaultValue={String(selectedAppt.appointment_type ?? '')}
                    onChange={(event) => {
                      handleChangeAppointmentType(selectedAppt.id, Number(event.target.value))
                      setShowSelectType(false)
                    }}
                  >
                    <option value="0">Asistida</option>
                    <option value="1">No asistió</option>
                    <option value="2">En espera</option>
                  </select>
                )}
                <button className="danger-button" type="button" onClick={() => handleDelete(selectedAppt.id)}>
                  Eliminar
                </button>
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => {
                    setShowDetails(false)
                    setDetailPrediction(null)
                  }}
                >
                  Cerrar
                </button>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
