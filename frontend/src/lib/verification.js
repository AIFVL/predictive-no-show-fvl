export const MANUAL_VERIFICATION_RULES = {
  non_attendance: {
    title: 'Checklist automático de inasistencia',
    minScore: 4,
    items: [
      { id: 'NA1', label: 'Historial de inasistencias alto', condition: 'Previous Non-Attendance >= 2', rationale: 'Fuerte predictor de comportamiento futuro', weight: 2 },
      { id: 'NA2', label: 'Alta tasa de no-show', condition: 'Prev_NoShow_Rate > 0.5', rationale: 'Indica tendencia dominante a faltar', weight: 2 },
      { id: 'NA3', label: 'Baja experiencia con citas', condition: 'Prev_Total <= 2', rationale: 'Paciente sin hábito en el sistema', weight: 1 },
      { id: 'NA4', label: 'Última cita fue no-show', condition: 'Last_Attendance = No-Show', rationale: 'Predictor reciente muy fuerte', weight: 2 },
      { id: 'NA5', label: 'Intervalo largo de asignación', condition: 'Creation-Assignment > 7 días', rationale: 'Mayor probabilidad de olvido', weight: 1 },
      { id: 'NA6', label: 'Cita con poca anticipación', condition: 'Creation-Assignment <= 2 días', rationale: 'Conflictos de agenda o falta de preparación', weight: 1 },
      { id: 'NA7', label: 'Paciente joven + bajo compromiso', condition: 'Age < 30 AND Prev_Attendance <= 1', rationale: 'Menor adherencia al control médico', weight: 1 },
      { id: 'NA8', label: 'Baja carga clínica + bajo compromiso', condition: 'Diseases <= 1 AND Prev_Attendance <= 1', rationale: 'Menor percepción de necesidad médica', weight: 1 },
    ],
  },
  attendance: {
    title: 'Checklist automático de asistencia',
    minScore: 4,
    items: [
      { id: 'A1', label: 'Historial alto de asistencia', condition: 'Previous Attendance >= 3', rationale: 'Comportamiento consistente positivo', weight: 2 },
      { id: 'A2', label: 'Baja tasa de no-show', condition: 'Prev_NoShow_Rate <= 0.2', rationale: 'Indica alta adherencia', weight: 2 },
      { id: 'A3', label: 'Última cita fue asistida', condition: 'Last_Attendance = Show', rationale: 'Fuerte predictor reciente', weight: 2 },
      { id: 'A4', label: 'Alta experiencia con citas', condition: 'Prev_Total >= 4', rationale: 'Paciente acostumbrado al sistema', weight: 1 },
      { id: 'A5', label: 'Intervalo moderado', condition: '3 <= Creation-Assignment <= 7 días', rationale: 'Balance entre preparación y olvido', weight: 1 },
      { id: 'A6', label: 'Paciente con mayor carga clínica', condition: 'Diseases >= 2', rationale: 'Mayor percepción de necesidad médica', weight: 1 },
      { id: 'A7', label: 'Mayor consumo de medicamentos', condition: 'Medications >= 2', rationale: 'Mayor compromiso terapéutico', weight: 1 },
      { id: 'A8', label: 'Baja carga clínica + bajo compromiso', condition: 'Diseases <= 1 AND Prev_Attendance <= 1', rationale: 'Mayor adherencia al seguimiento médico', weight: 1 },
    ],
  },
}

export const buildManualStateFromPrediction = (prediction) => {
  const state = {}
  const verificationRules = prediction?.verification?.rules || {}

  for (const groupKey of Object.keys(MANUAL_VERIFICATION_RULES)) {
    state[groupKey] = {}
    const autoChecks = verificationRules[groupKey]?.checks || {}
    for (const item of MANUAL_VERIFICATION_RULES[groupKey].items) {
      state[groupKey][item.id] = autoChecks[item.id]?.triggered === true
    }
  }

  return state
}

export const scoreManualGroup = (groupKey, groupState) => {
  const config = MANUAL_VERIFICATION_RULES[groupKey]
  return config.items.reduce((total, item) => (
    groupState?.[item.id] ? total + item.weight : total
  ), 0)
}

export const getManualGroupMaxScore = (groupKey) => (
  MANUAL_VERIFICATION_RULES[groupKey].items.reduce((total, item) => total + item.weight, 0)
)
