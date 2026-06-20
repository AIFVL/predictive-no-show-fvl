import { RISK_COPY } from '@fvl/shared'
import { getFinalPredictionLabel } from '@fvl/shared'
import {
  getManualGroupMaxScore,
  MANUAL_VERIFICATION_RULES,
} from './verification.js'

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value)))

export const formatPercent = (value) => {
  if (value == null || Number.isNaN(Number(value))) return 'Sin dato'
  return `${(Number(value) * 100).toFixed(1)}%`
}

export const formatShapValue = (value) => {
  if (value == null || Number.isNaN(Number(value))) return 'Sin dato'
  return Number(value).toFixed(4)
}

export const formatFactorValue = (value) => {
  if (value == null || value === '') return 'Sin dato'
  if (typeof value === 'number' && Number.isFinite(value)) return Number(value).toFixed(2)
  return String(value)
}

export const getProbAttendFromPrediction = (prediction) => {
  if (!prediction) return null
  if (prediction.model_analysis?.probability_attend != null) return clamp01(prediction.model_analysis.probability_attend)
  if (prediction.prob_attend != null) return clamp01(prediction.prob_attend)
  if (prediction.prob_no_show != null) return clamp01(1 - Number(prediction.prob_no_show))
  if (prediction.probability != null) return clamp01(1 - Number(prediction.probability))
  return null
}

export const getProbNoShowFromPrediction = (prediction) => {
  if (!prediction) return null
  if (prediction.model_analysis?.probability_no_show != null) return clamp01(prediction.model_analysis.probability_no_show)
  if (prediction.prob_no_show != null) return clamp01(prediction.prob_no_show)
  if (prediction.probability != null) return clamp01(prediction.probability)
  if (prediction.prob_attend != null) return clamp01(1 - Number(prediction.prob_attend))
  return null
}

export const getPredictionProbabilitySummary = (prediction) => {
  const finalLabel = getFinalPredictionLabel(prediction)
  const probAttend = getProbAttendFromPrediction(prediction)
  const probNoShow = getProbNoShowFromPrediction(prediction)

  if (!prediction || (probAttend == null && probNoShow == null)) {
    return { label: 'Sin probabilidad', probability: null, probAttend, probNoShow }
  }

  if (finalLabel === 1) {
    return { label: 'No asistirá', probability: probNoShow, probAttend, probNoShow }
  }

  return { label: 'Asistirá', probability: probAttend, probAttend, probNoShow }
}

export const getAdjustedProbabilitySummary = (prediction, manualAttendanceScore, manualNonAttendanceScore) => {
  const base = getPredictionProbabilitySummary(prediction)
  const hasManualChecks = manualAttendanceScore > 0 || manualNonAttendanceScore > 0

  if (!prediction || !hasManualChecks) {
    return { ...base, adjusted: false, modelProbAttend: base.probAttend, modelProbNoShow: base.probNoShow }
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

export const getRiskLevel = (prediction) => {
  const finalLabel = getFinalPredictionLabel(prediction)
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

export const getRiskSummary = (prediction) => {
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
    return { ...base, message: 'Hay señales mixtas; revisar antecedentes antes de la consulta.' }
  }

  if (level === 'low') {
    return { ...base, message: 'La combinación histórica sugiere asistencia esperada.' }
  }

  return { ...base, message: 'Aún no hay información analítica suficiente para clasificarla.' }
}

export const getPredictionHeadline = (prediction) => {
  if (!prediction) {
    return { short: 'Sin predicción', long: 'Esta cita programada aún no tiene predicción disponible.' }
  }

  const probAttend = getProbAttendFromPrediction(prediction)
  const finalLabel = getFinalPredictionLabel(prediction)

  if (probAttend == null) {
    return { short: 'Sin probabilidad', long: 'Hay registro de predicción, pero no se recibió una probabilidad interpretable.' }
  }

  if (finalLabel === 1) {
    return { short: 'No asistirá', long: 'Predicción final para esta cita programada: no asistirá.' }
  }

  return { short: 'Asistirá', long: 'Predicción final para esta cita programada: asistirá.' }
}

export const getToneClass = (tone) => {
  if (tone === 'critical') return 'tone-pill tone-pill-critical'
  if (tone === 'warning') return 'tone-pill tone-pill-warning'
  if (tone === 'positive') return 'tone-pill tone-pill-positive'
  return 'tone-pill tone-pill-neutral'
}

export const getManualVerificationSummary = (manualAttendanceScore, manualNonAttendanceScore) => {
  const attendanceMin = MANUAL_VERIFICATION_RULES.attendance.minScore
  const nonAttendanceMin = MANUAL_VERIFICATION_RULES.non_attendance.minScore

  if (manualAttendanceScore >= attendanceMin && manualNonAttendanceScore < nonAttendanceMin) {
    return 'La doble verificación automática favorece asistencia.'
  }
  if (manualNonAttendanceScore >= nonAttendanceMin && manualAttendanceScore < attendanceMin) {
    return 'La doble verificación automática favorece inasistencia.'
  }
  if (manualAttendanceScore >= attendanceMin && manualNonAttendanceScore >= nonAttendanceMin) {
    return 'La doble verificación automática tiene señales contradictorias.'
  }
  return 'La doble verificación automática aún no confirma asistencia ni inasistencia.'
}
