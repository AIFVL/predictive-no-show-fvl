import { useMemo, useRef, useState } from 'react'
import FullCalendar from '@fullcalendar/react'
import dayGridPlugin from '@fullcalendar/daygrid'
import timeGridPlugin from '@fullcalendar/timegrid'
import interactionPlugin from '@fullcalendar/interaction'
import esLocale from '@fullcalendar/core/locales/es'
import logoFvl from '../logoFVL.png'

import {
  APPOINTMENT_TYPE,
  DATE_WINDOW_OPTIONS,
  DISPLAY_STATUS,
  MONTH_LABELS,
  OUTCOME_STATUS_OPTIONS,
  STATUS_FILTER_OPTIONS,
  buildAppointmentDate,
  formatDateLabel,
  formatTimeLabel,
  getDisplayStatus,
  getDisplayStatusColor,
  getDisplayStatusLabel,
  isAppointmentTodayOrPast,
  matchesStatusFilter,
  emptyAppointmentForm,
} from '@fvl/shared'

import { useAppointments } from './hooks/useAppointments'
import {
  formatFactorValue,
  formatPercent,
  formatShapValue,
  getAdjustedProbabilitySummary,
  getPredictionHeadline,
  getPredictionSourceLabel,
  getRiskLevel,
  getRiskSummary,
  getToneClass,
  getManualVerificationSummary,
} from './lib/predictions'
import {
  MANUAL_VERIFICATION_RULES,
  buildManualStateFromPrediction,
  scoreManualGroup,
} from './lib/verification'

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
  const {
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
  } = useAppointments()

  const [detailPrediction, setDetailPrediction] = useState(null)
  const [manualVerification, setManualVerification] = useState({ non_attendance: {}, attendance: {} })
  const [showForm, setShowForm] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [showSelectType, setShowSelectType] = useState(false)
  const [selectedAppt, setSelectedAppt] = useState(null)
  const [detailsTab, setDetailsTab] = useState('info')
  const [statusFilter, setStatusFilter] = useState('all')
  const [riskFilter, setRiskFilter] = useState('all')
  const [form, setForm] = useState(emptyAppointmentForm)
  const [formErrors, setFormErrors] = useState({})
  const calendarRef = useRef(null)

  const handleSearch = async () => {
    const medicId = String(form.search ?? '').trim()
    await fetchAppointments(medicId, dateWindow)
  }

  const resetSearch = async () => {
    setForm((current) => ({ ...current, search: '' }))
    await fetchAppointments('', dateWindow)
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
      await createAppointment(normalized)
      setShowForm(false)
      setForm((current) => ({ ...emptyAppointmentForm, search: current.search }))
      setFormErrors({})
    } catch (error) {
      console.error('Error creating appointment', error)
      alert(`Error creando la cita: ${error.message}`)
    }
  }

  const openAppointmentDetail = async (appointmentId) => {
    try {
      const { appointment, prediction } = await fetchAppointmentDetail(appointmentId)
      setSelectedAppt(appointment)
      setDetailPrediction(prediction)
      setManualVerification(buildManualStateFromPrediction(prediction))
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
      await deleteAppointment(appointmentId)
      setShowDetails(false)
      setSelectedAppt(null)
    } catch (error) {
      console.error('Error deleting appointment', error)
      alert(`Error eliminando la cita: ${error.message}`)
    }
  }

  const handleChangeAppointmentType = async (appointmentId, newType) => {
    try {
      const updated = await updateAppointmentType(appointmentId, newType)
      setSelectedAppt(updated)
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

  const enrichedAppointments = useMemo(() => (
    appointments.map((appointment) => {
      const prediction = predictionsMap[String(appointment.id)] || null
      const displayStatus = getDisplayStatus(appointment, prediction)
      return {
        ...appointment,
        prediction,
        displayStatus,
        displayStatusLabel: getDisplayStatusLabel(appointment, prediction),
        risk: getRiskSummary(prediction),
        start: buildAppointmentDate(appointment),
        dateLabel: formatDateLabel(appointment, MONTH_LABELS),
        timeLabel: formatTimeLabel(appointment.hour),
      }
    })
  ), [appointments, predictionsMap])

  const filteredAppointments = useMemo(() => (
    enrichedAppointments.filter((appointment) => {
      if (!matchesStatusFilter(appointment, appointment.prediction, statusFilter)) return false
      if (riskFilter !== 'all' && getRiskLevel(appointment.prediction) !== riskFilter) return false
      return true
    })
  ), [enrichedAppointments, riskFilter, statusFilter])

  const events = useMemo(() => (
    filteredAppointments.map((appointment) => {
      const baseColor = getDisplayStatusColor(appointment, appointment.prediction)
      const isScheduled = appointment.appointment_type === APPOINTMENT_TYPE.EN_ESPERA

      return {
        id: String(appointment.id),
        title: isScheduled
          ? `${appointment.timeLabel} · ${appointment.patient_id} · ${appointment.displayStatusLabel}`
          : `${appointment.timeLabel} · ${appointment.patient_id} · ${appointment.displayStatusLabel}`,
        start: appointment.start.toISOString(),
        backgroundColor: baseColor,
        borderColor: baseColor,
        textColor: '#ffffff',
        extendedProps: appointment,
      }
    })
  ), [filteredAppointments])

  const stats = useMemo(() => {
    const scheduled = enrichedAppointments.filter((a) => a.appointment_type === APPOINTMENT_TYPE.EN_ESPERA)
    const willAttend = enrichedAppointments.filter((a) => a.displayStatus === DISPLAY_STATUS.ASISTIRA)
    const willNotAttend = enrichedAppointments.filter((a) => a.displayStatus === DISPLAY_STATUS.NO_ASISTIRA)
    const resolved = enrichedAppointments.filter((a) => a.appointment_type !== APPOINTMENT_TYPE.EN_ESPERA)

    return [
      {
        label: 'Citas en ventana',
        value: enrichedAppointments.length,
        caption: activeMedicFilter ? `Médico ${activeMedicFilter}` : `Próximos ${dateWindow} días`,
      },
      {
        label: 'Programadas',
        value: scheduled.length,
        caption: `${summary?.analyzed ?? 0} con predicción activa`,
      },
      {
        label: 'Predicción: no asistirá',
        value: willNotAttend.length,
        caption: 'Prioridad para confirmación',
      },
      {
        label: 'Resultados cerrados',
        value: resolved.length,
        caption: `${willAttend.length} con predicción asistirá`,
      },
    ]
  }, [activeMedicFilter, dateWindow, enrichedAppointments, summary])

  const priorityList = useMemo(() => (
    [...filteredAppointments]
      .filter((appointment) => appointment.displayStatus === DISPLAY_STATUS.NO_ASISTIRA)
      .sort((left, right) => left.start.getTime() - right.start.getTime())
      .slice(0, 6)
  ), [filteredAppointments])

  const selectedPrediction = detailPrediction ?? (selectedAppt ? predictionsMap[String(selectedAppt.id)] : null)
  const selectedRisk = getRiskSummary(selectedPrediction)
  const selectedPredictionHeadline = getPredictionHeadline(selectedPrediction)
  const selectedModelAnalysis = selectedPrediction?.model_analysis ?? selectedPrediction?.shap_analysis ?? null
  const selectedTopFactors = selectedModelAnalysis?.top_factors?.slice(0, 3) || []
  const canChangeState = selectedAppt
    && selectedAppt.appointment_type === APPOINTMENT_TYPE.EN_ESPERA
    && isAppointmentTodayOrPast(selectedAppt)
  const isSelectedScheduled = selectedAppt?.appointment_type === APPOINTMENT_TYPE.EN_ESPERA
  const selectedDisplayStatus = selectedAppt
    ? getDisplayStatusLabel(selectedAppt, selectedPrediction)
    : 'Sin estado'
  const manualAttendanceScore = scoreManualGroup('attendance', manualVerification.attendance)
  const manualNonAttendanceScore = scoreManualGroup('non_attendance', manualVerification.non_attendance)
  const selectedProbability = getAdjustedProbabilitySummary(
    selectedPrediction,
    manualAttendanceScore,
    manualNonAttendanceScore,
  )
  const manualVerificationSummary = getManualVerificationSummary(
    manualAttendanceScore,
    manualNonAttendanceScore,
  )

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
              <h1>Predicción de inasistencia de citas para medicina interna</h1>
              <p className="hero-copy">
                El tablero carga citas desde hoy en una ventana configurable (8, 15 o 30 días),
                predice inasistencias solo en el backend y clasifica cada cita como
                <strong> Asistirá</strong>, <strong>No asistirá</strong>, <strong>Asistida</strong> o <strong>No asistió</strong>.
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
              <p className="legend-title">Estados de las citas</p>
              <div className="legend-grid">
                <span><i className="legend-dot legend-dot-blue" /> Asistirá</span>
                <span><i className="legend-dot legend-dot-red" /> No asistirá</span>
                <span><i className="legend-dot legend-dot-green" /> Asistida</span>
                <span><i className="legend-dot legend-dot-orange" /> No asistió</span>
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
                <select
                  className="compact-filter"
                  value={dateWindow}
                  onChange={(event) => setDateWindow(Number(event.target.value))}
                >
                  {DATE_WINDOW_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <select className="compact-filter" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                  {STATUS_FILTER_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
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

            {loading && <p className="loading-banner">Cargando citas y predicciones…</p>}

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
                eventClick={(info) => openAppointmentDetail(info.event.id)}
                eventTimeFormat={{ hour: '2-digit', minute: '2-digit', meridiem: false }}
                dayMaxEvents={3}
                buttonText={{ today: 'Hoy', month: 'Mes', week: 'Semana', day: 'Día' }}
              />
            </div>
          </div>

          <aside className="insights-panel">
            <div className="insight-card">
              <p className="eyebrow">Lectura del calendario</p>
              <h3>Estados operativos</h3>
              <p>
                Las citas nuevas quedan <strong>En espera</strong> internamente. El modelo predice
                <strong> Asistirá</strong> o <strong>No asistirá</strong>. Tras la consulta se cierran como
                <strong> Asistida</strong> o <strong>No asistió</strong>.
              </p>
            </div>

            <div className="insight-card">
              <div className="section-head">
                <h3>Prioridades del día</h3>
                <span>{priorityList.length} casos</span>
              </div>
              <div className="priority-list">
                {priorityList.length === 0 && <p className="empty-state">No hay citas con predicción “No asistirá” en la ventana actual.</p>}
                {priorityList.map((appointment) => (
                  <button
                    className="priority-item"
                    key={appointment.id}
                    type="button"
                    onClick={() => openAppointmentDetail(appointment.id)}
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
                <span>Ventana {dateWindow} días</span>
              </div>
              <div className="model-summary">
                <strong>{summary?.analyzed ?? 0} citas analizadas</strong>
                <p>El backend ejecuta las predicciones en lote solo para citas en espera dentro de la ventana seleccionada.</p>
                <p>Las citas ya cerradas como Asistida o No asistió no se vuelven a predecir.</p>
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
                  <p>{formatDateLabel(selectedAppt, MONTH_LABELS)} · {formatTimeLabel(selectedAppt.hour)}</p>
                </div>
                <span className={getToneClass(isSelectedScheduled ? selectedRisk.tone : 'neutral')}>
                  {selectedDisplayStatus}
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
                    <strong>{selectedDisplayStatus}</strong>
                  </div>
                  <div className="detail-box">
                    <span>Creada</span>
                    <strong>{selectedAppt.created_at ? new Date(selectedAppt.created_at).toLocaleString('es-CO') : 'Sin fecha'}</strong>
                  </div>
                  {isSelectedScheduled ? (
                    <>
                      <div className="detail-box">
                        <span>Predicción de la cita</span>
                        <strong>{selectedPredictionHeadline.short}</strong>
                      </div>
                      <div className="detail-box probability-box detail-box-wide">
                        <span>{selectedProbability.adjusted ? 'Probabilidad ajustada por doble verificación' : 'Probabilidad automática'}</span>
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
                        {selectedAppt.appointment_type === APPOINTMENT_TYPE.ASISTIDA
                          ? 'La cita fue registrada como Asistida.'
                          : selectedAppt.appointment_type === APPOINTMENT_TYPE.NO_ASISTIO
                            ? 'La cita fue registrada como No asistió.'
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
                      <h3>Resumen automático</h3>
                      <span>{manualVerificationSummary}</span>
                    </div>
                    <div className="verification-probability-panel">
                      <div className="probability-headline">
                        <strong>{selectedProbability.label}</strong>
                        <strong>{formatPercent(selectedProbability.probability)}</strong>
                      </div>
                      <div className="probability-meter" aria-hidden="true">
                        <i style={{ width: `${Math.round((selectedProbability.probability ?? 0) * 100)}%` }} />
                      </div>
                    </div>
                  </div>

                  {['attendance', 'non_attendance'].map((groupKey) => {
                    const config = MANUAL_VERIFICATION_RULES[groupKey]
                    const autoGroup = selectedPrediction?.verification?.rules?.[groupKey]

                    return (
                      <div className="verification-card" key={groupKey}>
                        <div className="section-head">
                          <h3>{config.title}</h3>
                          <span>
                            Puntaje automático {scoreManualGroup(groupKey, manualVerification[groupKey])} · mínimo {config.minScore}
                          </span>
                        </div>
                        <div className="manual-checklist">
                          {config.items.map((item) => {
                            const autoCheck = autoGroup?.checks?.[item.id]
                            return (
                              <label className="manual-check-row" key={item.id}>
                                <input type="checkbox" checked={!!manualVerification[groupKey]?.[item.id]} disabled readOnly />
                                <div className="manual-check-content">
                                  <div className="manual-check-head">
                                    <strong>{item.label}</strong>
                                    <span>Peso {formatPercent(autoCheck?.shap_weight)}</span>
                                  </div>
                                  <span className="manual-check-condition">{item.condition}</span>
                                  <span className="manual-check-auto">
                                    Resultado: {autoCheck?.triggered === true ? 'Cumple' : autoCheck?.triggered === false ? 'No cumple' : 'Sin dato'} · Valor SHAP {formatShapValue(autoCheck?.shap_value)}
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
                  Registrar resultado
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
                    {OUTCOME_STATUS_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
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
