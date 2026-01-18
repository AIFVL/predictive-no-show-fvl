import { useState, useEffect, useMemo, useRef } from 'react'
import FullCalendar from '@fullcalendar/react'
import dayGridPlugin from '@fullcalendar/daygrid'
import timeGridPlugin from '@fullcalendar/timegrid'
import interactionPlugin from '@fullcalendar/interaction'
// CSS loaded from CDN in index.html

function App() {
  const [appointments, setAppointments] = useState([])
  const [predictionsMap, setPredictionsMap] = useState({})
  const TYPE_LABELS = {
    0: 'Asistida',
    1: 'No show',
    2: 'En espera'
  }
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({ medic_id: '', patient_id: '', hour: 9, day: 1, month: 1, search: '' })

  const fetchAppointments = async () => {
    try {
      const res = await fetch('http://localhost:8000/appointments')
      const data = await res.json()
      const appts = Array.isArray(data) ? data : []
      setAppointments(appts)
      // after loading appointments, fetch predictions for waiting ones
      fetchPredictionsWaiting()
    } catch (err) {
      console.error('Failed to fetch appointments', err)
    }
  }

  const fetchPredictionsWaiting = async (medicId) => {
    try {
      const url = medicId ? `http://localhost:8000/predictions/waiting?medic_id=${encodeURIComponent(medicId)}` : `http://localhost:8000/predictions/waiting`
      const res = await fetch(url)
      if (!res.ok) {
        console.warn('Failed to fetch predictions for waiting appointments')
        return
      }
      const data = await res.json()
      const map = {}
      for (const item of (data.per_appointment || [])) {
        if (item && item.appointment_id != null) map[String(item.appointment_id)] = item
      }
      setPredictionsMap(map)
    } catch (err) {
      console.error('Failed to fetch waiting predictions', err)
    }
  }

  useEffect(() => { fetchAppointments() }, [])

  const toggleForm = () => setShowForm(v => !v)

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: name === 'hour' || name === 'day' || name === 'month' ? Number(value) : value }))
  }



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
    } catch (err) {
      console.error('Search failed', err)
      alert('Error searching appointments: ' + err.message)
    }
  }

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

  const events = useMemo(() => {
    const year = 2026
    return appointments.map(appt => {
      const start = new Date(year, (appt.month || 1) - 1, appt.day || 1, appt.hour || 0)
      // color the event according to appointment type
      const colorMap = {
        0: '#10b981', // Asistida -> green
        1: '#dc2626', // No show -> red
        2: '#2563eb'  // En espera -> blue/default
      }
      const color = colorMap[appt.appointment_type] || '#2563eb'

      return {
        id: String(appt.id),
        title: `Paciente ${appt.patient_id}` + (appt.appointment_type !== undefined ? ` — ${TYPE_LABELS[appt.appointment_type] ?? appt.appointment_type}` : ''),
        start: start.toISOString(),
        allDay: false,
        extendedProps: appt,
        color,
      }
    })
  }, [appointments])

  const [selectedAppt, setSelectedAppt] = useState(null)
  const [showDetails, setShowDetails] = useState(false)
  const [showSelectType, setShowSelectType] = useState(false)
  const calendarRef = useRef(null)

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

  const handleDateClick = (info) => {
    // When clicking a day in month/year view, switch to the single-day time grid
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
          <form onSubmit={handleSubmit} className="mb-6 p-4 bg-white border rounded-lg shadow-sm">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="flex flex-col">
                <span className="text-sm text-slate-600">Médico ID</span>
                <input name="medic_id" value={form.medic_id} onChange={handleChange} required className="mt-1 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-200" />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-slate-600">Paciente ID</span>
                <input name="patient_id" value={form.patient_id} onChange={handleChange} required className="mt-1 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-200" />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-slate-600">Hora</span>
                <input name="hour" type="number" min="0" max="23" value={form.hour} onChange={handleChange} required className="mt-1 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-200" />
              </label>
              <label className="flex flex-col">
                <span className="text-sm text-slate-600">Día</span>
                <input name="day" type="number" min="1" max="31" value={form.day} onChange={handleChange} required className="mt-1 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-200" />
              </label>
              <label className="flex flex-col md:col-span-2">
                <span className="text-sm text-slate-600">Mes</span>
                <input name="month" type="number" min="1" max="12" value={form.month} onChange={handleChange} required className="mt-1 p-2 border rounded w-32 focus:outline-none focus:ring-2 focus:ring-blue-200" />
              </label>
            </div>
            <div className="mt-4 flex gap-2">
              <button type="submit" className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded shadow">Crear cita</button>
              <button type="button" onClick={() => setShowForm(false)} className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded">Cancelar</button>
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
                    {predictionsMap[String(selectedAppt.id)].prob_attend != null
                      ? `${(predictionsMap[String(selectedAppt.id)].prob_attend * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                )}
              </div>
              <div style={{ marginTop: 12, display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                {/* Disable change button when appointment is already decided (not 'En espera') */}
                <button
                  onClick={() => { if (selectedAppt.appointment_type === 2) setShowSelectType(v => !v) }}
                  disabled={selectedAppt.appointment_type !== 2}
                  style={{
                    padding: '6px 10px',
                    borderRadius: 6,
                    background: selectedAppt.appointment_type === 2 ? '#2563eb' : '#94a3b8',
                    color: '#fff',
                    border: 'none',
                    cursor: selectedAppt.appointment_type === 2 ? 'pointer' : 'not-allowed',
                    opacity: selectedAppt.appointment_type === 2 ? 1 : 0.6
                  }}
                >
                  Cambiar estado de cita
                </button>
                {showSelectType && selectedAppt.appointment_type === 2 && (
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
