import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import FullCalendar from '@fullcalendar/react'
import dayGridPlugin from '@fullcalendar/daygrid'
import timeGridPlugin from '@fullcalendar/timegrid'
import interactionPlugin from '@fullcalendar/interaction'
// CSS loaded from CDN in index.html

const TYPE_LABELS = {
  0: 'Asistida',
  1: 'No show',
  2: 'En espera'
}

// Color helpers for probability-based event styling
const clamp01 = (x) => Math.max(0, Math.min(1, x))

const hexToRgb = (hex) => {
  const h = String(hex || '').replace('#', '')
  const full = h.length === 3 ? h.split('').map(ch => ch + ch).join('') : h
  const n = parseInt(full, 16)
  if (!Number.isFinite(n)) return null
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 }
}

const rgbToHex = ({ r, g, b }) => {
  const to2 = (v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')
  return `#${to2(r)}${to2(g)}${to2(b)}`
}

const lerp = (a, b, t) => a + (b - a) * t

const lerpHex = (aHex, bHex, t) => {
  const a = hexToRgb(aHex)
  const b = hexToRgb(bHex)
  if (!a || !b) return bHex
  return rgbToHex({
    r: lerp(a.r, b.r, t),
    g: lerp(a.g, b.g, t),
    b: lerp(a.b, b.b, t),
  })
}

// Map probability of attendance to a smooth red->yellow->green palette.
// 0.0 => red, 0.5 => yellow, 1.0 => green.
const colorFromProbAttend = (probAttend) => {
  const p = clamp01(Number(probAttend))
  const RED = '#dc2626'
  const YELLOW = '#f59e0b'
  const GREEN = '#10b981'
  if (p <= 0.5) return lerpHex(RED, YELLOW, p / 0.5)
  return lerpHex(YELLOW, GREEN, (p - 0.5) / 0.5)
}

const relativeLuminance = ({ r, g, b }) => {
  const toLinear = (c) => {
    const s = c / 255
    return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4)
  }
  const R = toLinear(r)
  const G = toLinear(g)
  const B = toLinear(b)
  return 0.2126 * R + 0.7152 * G + 0.0722 * B
}

const contrastRatio = (l1, l2) => {
  const [L1, L2] = l1 >= l2 ? [l1, l2] : [l2, l1]
  return (L1 + 0.05) / (L2 + 0.05)
}

const bestTextColorForBg = (bgHex) => {
  const rgb = hexToRgb(bgHex)
  if (!rgb) return '#ffffff'
  const bgLum = relativeLuminance(rgb)
  const whiteLum = 1
  const blackLum = 0

  const cWhite = contrastRatio(whiteLum, bgLum)
  const cBlack = contrastRatio(bgLum, blackLum)
  // Use near-black for better look than pure black.
  return cBlack >= cWhite ? '#111827' : '#ffffff'
}

function App() {
  // Top-level component state
  // `appointments` holds the list of appointments fetched from backend
  const [appointments, setAppointments] = useState([])
  // `predictionsMap` stores prediction results keyed by appointment id
  const [predictionsMap, setPredictionsMap] = useState({})
  const [showForm, setShowForm] = useState(false)
  // `form` stores controlled inputs for the add-appointment form
  const [form, setForm] = useState({ medic_id: '', patient_id: '', hour: 9, day: 1, month: 1, search: '' })

  // Query backend for predictions for appointments in 'En espera'
  // Builds a map { appointment_id: predictionObject } for quick lookup
  const fetchPredictionsWaiting = useCallback(async (medicId) => {
    try {
      const url = medicId ? `http://localhost:8000/predictions/waiting?medic_id=${encodeURIComponent(medicId)}` : `http://localhost:8000/predictions/waiting`
      const res = await fetch(url)
      if (!res.ok) {
        console.warn('Failed to fetch predictions for waiting appointments')
        return
      }
      const data = await res.json()
      const map = {}
      // `per_appointment` contains items with appointment_id, prob_attend, prob_no_show, etc.
      for (const item of (data.per_appointment || [])) {
        if (item && item.appointment_id != null) map[String(item.appointment_id)] = item
      }
      setPredictionsMap(map)
    } catch (err) {
      console.error('Failed to fetch waiting predictions', err)
    }
  }, [])

  // Fetch all appointments from the backend API and refresh predictions map
  const fetchAppointments = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/appointments')
      const data = await res.json()
      const appts = Array.isArray(data) ? data : []
      setAppointments(appts)
      // After loading appointments, refresh prediction data for waiting appointments
      fetchPredictionsWaiting()
    } catch (err) {
      console.error('Failed to fetch appointments', err)
    }
  }, [fetchPredictionsWaiting])

  useEffect(() => { fetchAppointments() }, [fetchAppointments])

  // Toggle the add-appointment form visibility
  const toggleForm = () => setShowForm(v => !v)

  // Generic controlled input handler for the add form
  const handleChange = (e) => {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: name === 'hour' || name === 'day' || name === 'month' ? Number(value) : value }))
  }



  // Search appointments by medic id; refreshes the appointments list
  const handleSearch = async () => {
    const medicId = form.search && String(form.search).trim()
    if (!medicId) {
      fetchAppointments()
      return
    }
    try {
      const res = await fetch(`http://localhost:8000/appointments/${encodeURIComponent(medicId)}`)
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Failed to fetch by medic')
      }
      const data = await res.json()
      setAppointments(Array.isArray(data) ? data : [])
      // Keep prediction map in sync with the filtered appointment list.
      fetchPredictionsWaiting(medicId)
    } catch (err) {
      console.error('Search failed', err)
      alert('Error searching appointments: ' + err.message)
    }
  }

  // Submit handler for the add-appointment form
  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      const res = await fetch('http://localhost:8000/appointments/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Failed to create appointment')
      }
      // refresh and close form
      await fetchAppointments()
      setShowForm(false)
      setForm({ medic_id: '', patient_id: '', hour: 9, day: 1, month: 1 })
    } catch (err) {
      console.error('Failed to create appointment', err)
      alert('Error creating appointment: ' + err.message)
    }
  }

  // Build FullCalendar events from appointments state. Memoized for performance.
  const events = useMemo(() => {
    const year = 2026
    return appointments.map(appt => {
      const start = new Date(year, (appt.month || 1) - 1, appt.day || 1, appt.hour || 0)
      // color the event according to appointment type
      // 0: Asistida (green), 1: No show (red), 2: En espera (blue)
      const colorMap = {
        0: '#10b981', // Asistida -> green
        1: '#dc2626', // No show -> red
        2: '#2563eb'  // En espera -> blue/default
      }
      let color = colorMap[appt.appointment_type] || '#2563eb'

      // If appointment is waiting and we have a prediction, color by probability of attendance.
      if (appt.appointment_type === 2) {
        const p = predictionsMap[String(appt.id)]
        const probAttend =
          p?.prob_attend != null
            ? Number(p.prob_attend)
            : p?.prob_no_show != null
              ? 1 - Number(p.prob_no_show)
              : p?.probability != null
                ? 1 - Number(p.probability)
                : null

        if (probAttend != null && Number.isFinite(probAttend)) {
          color = colorFromProbAttend(probAttend)
        }
      }

      const textColor = bestTextColorForBg(color)

      return {
        id: String(appt.id),
        title: `Paciente ${appt.patient_id}` + (appt.appointment_type !== undefined ? ` — ${TYPE_LABELS[appt.appointment_type] ?? appt.appointment_type}` : ''),
        start: start.toISOString(),
        allDay: false,
        extendedProps: appt,
        color,
        textColor,
      }
    })
  }, [appointments, predictionsMap])

  const [selectedAppt, setSelectedAppt] = useState(null)
  const [showDetails, setShowDetails] = useState(false)
  const [showSelectType, setShowSelectType] = useState(false)
  const calendarRef = useRef(null)

  // Load detailed appointment info from backend and open modal
  const fetchAppointmentInfo = async (appointmentId) => {
    try {
      const res = await fetch(`http://localhost:8000/appointments/info/${appointmentId}`)
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Failed to fetch appointment info')
      }
      const data = await res.json()
      setSelectedAppt(data)
      setShowDetails(true)
    } catch (err) {
      console.error('Failed to fetch appointment info', err)
      alert('Error fetching appointment info: ' + err.message)
    }
  }

  // Delete an appointment by id and refresh list
  const handleDelete = async (appointmentId) => {
    if (!window.confirm('¿Estás seguro de que deseas eliminar esta cita?')) return
    try {
      const res = await fetch(`http://localhost:8000/appointments/${appointmentId}`, { method: 'DELETE' })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Failed to delete appointment')
      }
      await fetchAppointments()
      setShowDetails(false)
    } catch (err) {
      console.error('Failed to delete appointment', err)
      alert('Error deleting appointment: ' + err.message)
    }
  }
  
  // Change only the appointment_type for an appointment (PATCH)
  // Note: UI blocks this action unless the appointment is currently 'En espera'
  const handleChangeAppointmentType = async (appointmentId, newType) => {
    try{
      // backend expects appointment_type as query parameter
      const res = await fetch(`http://localhost:8000/appointments/type/${appointmentId}?appointment_type=${encodeURIComponent(newType)}`, {
        method: 'PATCH'
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || 'Failed to update appointment type')
      }
      const updatedAppt = await res.json()
      setSelectedAppt(updatedAppt)
      await fetchAppointments()
    } catch (err) {
      console.error('Failed to update appointment type', err)
      alert('Error updating appointment type: ' + err.message)
    }
  }

  // Handle clicks on calendar days: when user clicks a day in month view,
  // switch to the single-day view and navigate to that date.
  const handleDateClick = (info) => {
    try {
      const api = calendarRef.current?.getApi()
      if (!api) return
      const currentView = api.view?.type
      // If currently in the month-like view, change to day view first
      if (currentView === 'dayGridMonth' || currentView === 'dayGridYear') {
        api.changeView('timeGridDay')
      }
      // Navigate to the clicked date
      api.gotoDate(info.date)
    } catch (err) {
      console.error('Error handling date click', err)
    }
  }

  // Helper: determine whether an appointment datetime has already passed.
  // Uses the current year so comparisons match the UI calendar year.
  const isAppointmentInPast = (appt) => {
    if (!appt) return false
    const year = new Date().getFullYear()
    // fallback values if fields missing
    const month = (appt.month || 1) - 1
    const day = appt.day || 1
    const hour = appt.hour || 0
    const apptDate = new Date(year, month, day, hour)
    return apptDate.getTime() < Date.now()
  }



  return (
    <div className="min-h-screen bg-slate-50 p-6">
      <div className="max-w-4xl mx-auto">
        <header className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-extrabold text-slate-800">Calendario de Citas</h1>
            <p className="text-sm text-slate-500">Ver y crear citas médicas — todas las citas mostradas en 2026</p>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <input
              placeholder="Buscar por Médico ID"
              value={form.search || ''}
              name="search"
              onChange={(e) => setForm(prev => ({ ...prev, search: e.target.value }))}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleSearch() } }}
              style={{ padding: '8px 10px', borderRadius: 6, border: '1px solid #d1d5db' }}
            />
            <button onClick={() => handleSearch()} style={{ padding: '8px 12px', background: '#1e40af', color: '#fff', borderRadius: 6, border: 'none', cursor: 'pointer' }}>Buscar</button>
            <button onClick={() => { setForm(prev => ({ ...prev, search: '' })); fetchAppointments() }} style={{ padding: '8px 12px', background: '#64748b', color: '#fff', borderRadius: 6, border: 'none', cursor: 'pointer' }}>Mostrar todo</button>
          </div>
        </header>

        {showForm && (
          <form onSubmit={handleSubmit} className="mb-6 bg-white border rounded-lg shadow-sm" style={{ padding: 16 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, alignItems: 'end' }}>
              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#475569', marginBottom: 6 }}>Médico ID</label>
                <input name="medic_id" value={form.medic_id} onChange={handleChange} required style={{ width: '100%', padding: 10, borderRadius: 8, border: '1px solid #e6edf3' }} />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#475569', marginBottom: 6 }}>Paciente ID</label>
                <input name="patient_id" value={form.patient_id} onChange={handleChange} required style={{ width: '100%', padding: 10, borderRadius: 8, border: '1px solid #e6edf3' }} />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#475569', marginBottom: 6 }}>Hora</label>
                <input name="hour" type="number" min="0" max="23" value={form.hour} onChange={handleChange} required style={{ width: '100%', padding: 10, borderRadius: 8, border: '1px solid #e6edf3' }} />
              </div>

              <div>
                <label style={{ display: 'block', fontSize: 12, color: '#475569', marginBottom: 6 }}>Día</label>
                <input name="day" type="number" min="1" max="31" value={form.day} onChange={handleChange} required style={{ width: '100%', padding: 10, borderRadius: 8, border: '1px solid #e6edf3' }} />
              </div>

              <div style={{ gridColumn: '1 / -1' }}>
                <label style={{ display: 'block', fontSize: 12, color: '#475569', marginBottom: 6 }}>Mes</label>
                <input name="month" type="number" min="1" max="12" value={form.month} onChange={handleChange} required style={{ padding: 10, borderRadius: 8, border: '1px solid #e6edf3', width: 140 }} />
              </div>
            </div>

            <div style={{ marginTop: 16, display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button type="submit" style={{ padding: '8px 14px', background: '#059669', color: '#fff', borderRadius: 8, border: 'none', cursor: 'pointer' }}>Crear cita</button>
              <button type="button" onClick={() => setShowForm(false)} style={{ padding: '8px 14px', background: '#f3f4f6', color: '#111827', borderRadius: 8, border: '1px solid #e5e7eb' }}>Cancelar</button>
            </div>
          </form>
        )}

        <div className="bg-white rounded-lg shadow p-4">
          <FullCalendar
            ref={calendarRef}
            plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin]}
            initialView="dayGridMonth"
            headerToolbar={{ left: 'prev,next today', center: 'title', right: 'dayGridMonth,timeGridWeek,timeGridDay' }}
            events={events}
            eventClick={(info) => {
              const apptId = info.event.id || info.event.extendedProps?.id
              if (apptId) fetchAppointmentInfo(apptId)
            }}
            dateClick={handleDateClick}
            height="auto"
          />
          <div style={{ marginTop: 12, display: 'flex', justifyContent: 'flex-start' }}>
            <button
              onClick={toggleForm}
              style={{
                background: '#059669',
                color: '#fff',
                padding: '8px 14px',
                borderRadius: 8,
                boxShadow: '0 4px 8px rgba(0,0,0,0.12)',
                border: 'none',
                cursor: 'pointer'
              }}
            >
              {showForm ? 'Cerrar' : 'Añadir cita'}
            </button>
          </div>
        </div>
        {showDetails && selectedAppt && (
          <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }} onClick={() => setShowDetails(false)}>
            <div style={{ background: '#fff', padding: 20, borderRadius: 8, minWidth: 320, maxWidth: '90%' }} onClick={(e) => e.stopPropagation()}>
              <h2 style={{ marginTop: 0 }}>Detalle de la cita #{selectedAppt.id}</h2>
              <div style={{ display: 'grid', gap: 8 }}>
                <div><strong>Médico ID:</strong> {selectedAppt.medic_id ?? '—'}</div>
                <div><strong>Paciente ID:</strong> {selectedAppt.patient_id}</div>
                <div><strong>Fecha (día/mes/año):</strong> {`${selectedAppt.day}/${selectedAppt.month}/2026`}</div>
                <div><strong>Hora:</strong> {selectedAppt.hour}:00</div>
                <div><strong>Tipo de cita:</strong> {TYPE_LABELS[selectedAppt.appointment_type] ?? selectedAppt.appointment_type}</div>
                <div><strong>Creada en:</strong> {selectedAppt.created_at ?? '—'}</div>
                {selectedAppt.appointment_type === 2 && predictionsMap[String(selectedAppt.id)] && (
                  <div>
                    <strong>Predicción de asistencia:</strong>{' '}
                    {(() => {
                      const p = predictionsMap[String(selectedAppt.id)]
                      const probAttend =
                        p?.prob_attend != null
                          ? Number(p.prob_attend)
                          : p?.prob_no_show != null
                            ? 1 - Number(p.prob_no_show)
                            : p?.probability != null
                              ? 1 - Number(p.probability)
                              : null

                      return probAttend != null && Number.isFinite(probAttend)
                        ? `${(probAttend * 100).toFixed(1)}%`
                        : '—'
                    })()}
                  </div>
                )}
              </div>
              <div style={{ marginTop: 12, display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                {/* Disable change button unless appointment is 'En espera' AND its date/time is already past */}
                {(() => {
                  const canChangeState = selectedAppt && selectedAppt.appointment_type === 2 && isAppointmentInPast(selectedAppt)
                  return (
                    <button
                      onClick={() => { if (canChangeState) setShowSelectType(v => !v) }}
                      disabled={!canChangeState}
                      style={{
                        padding: '6px 10px',
                        borderRadius: 6,
                        background: canChangeState ? '#2563eb' : '#94a3b8',
                        color: '#fff',
                        border: 'none',
                        cursor: canChangeState ? 'pointer' : 'not-allowed',
                        opacity: canChangeState ? 1 : 0.6
                      }}
                    >
                      Cambiar estado de cita
                    </button>
                  )
                })()}
                {showSelectType && selectedAppt && selectedAppt.appointment_type === 2 && isAppointmentInPast(selectedAppt) && (
                  <select
                    onChange={(e) => {
                      const newType = Number(e.target.value)
                      handleChangeAppointmentType(selectedAppt.id, newType)
                      setShowSelectType(false)
                    }}
                    defaultValue={String(selectedAppt.appointment_type ?? '')}
                    style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid #d1d5db' }}
                  >
                    <option value={0}>Asistida</option>
                    <option value={1}>No show</option>
                    <option value={2}>En espera</option>
                  </select>
                )}
                <button onClick={() => handleDelete(selectedAppt.id)} style={{ padding: '6px 10px', borderRadius: 6, background: '#dc2626', color: '#fff', border: 'none' }}>Eliminar cita</button>
                <button onClick={() => setShowDetails(false)} style={{ padding: '6px 10px', borderRadius: 6 }}>Cerrar</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
