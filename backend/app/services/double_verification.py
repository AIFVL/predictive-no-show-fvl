"""Rule-based double verification for appointment attendance predictions.

This module computes weighted checklists for two perspectives:
- non_attendance: signals that support a no-show prediction
- attendance: signals that support an attendance prediction

The rule configuration is persisted by the training pipeline inside
`backend/models/metrics.json` under the `double_verification` key.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except Exception:
        return None


def _norm_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value).strip().lower()
    except Exception:
        return None


def _derive_prev_total_and_rate(features: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    prev_att = _to_float(features.get("Number of Previous Attendance"))
    prev_non = _to_float(features.get("Number of Previous Non-Attendance"))

    if prev_att is None or prev_non is None:
        return None, None

    prev_total = prev_att + prev_non
    if prev_total <= 0:
        return prev_total, 0.0

    return prev_total, (prev_non / prev_total)


def _last_attendance_label(features: Dict[str, Any]) -> Optional[str]:
    """Try to interpret a last-attendance-like field.

    Accepted string-ish inputs: "no-show", "noshow", "show", "attended", "asistida".
    If the dataset uses a numeric encoding, this function returns None (unknown).
    """

    for key in (
        "Last Attendance",
        "Last_Attendance",
        "LastAttendance",
        "last_attendance",
    ):
        if key in features:
            raw = features.get(key)
            break
    else:
        raw = None

    # Fallback: some datasets store last attendance as a numeric appointment type.
    # Convention (as used in this project):
    # - 0 => no-show
    # - 1 => show
    if raw is None:
        for key in (
            "Appointment Type",
            "Appointment_Type",
            "AppointmentType",
            "appointment_type",
            # tolerate common typos
            "Opointment Type",
            "Opointment_Type",
            "OpointmentType",
        ):
            if key in features:
                raw = features.get(key)
                break

    if raw is None:
        return None

    n = _to_float(raw)
    if n is not None:
        if int(n) == 0:
            return "no-show"
        if int(n) == 1:
            return "show"

    s = _norm_str(raw)
    if not s:
        return None

    # common variants
    if s in {"no-show", "no show", "noshow", "no_show", "no\u2011show"}:
        return "no-show"
    if s in {"show", "attended", "attendance", "asistida", "asistio", "asistió"}:
        return "show"

    return None


def _eval_rule(rule_id: str, features: Dict[str, Any]) -> Optional[bool]:
    """Evaluate a known rule.

    Returns:
    - True/False when the rule can be evaluated
    - None when required inputs are missing / not interpretable
    """

    age = _to_float(features.get("Age"))
    diseases = _to_float(features.get("Number of Diseases"))
    medications = _to_float(features.get("Number of Medications"))
    creation_interval = _to_float(features.get("Creation to Assignment Interval"))
    prev_att = _to_float(features.get("Number of Previous Attendance"))
    prev_non = _to_float(features.get("Number of Previous Non-Attendance"))
    prev_total, prev_noshow_rate = _derive_prev_total_and_rate(features)
    last_att = _last_attendance_label(features)

    if rule_id == "NA1":
        return None if prev_non is None else (prev_non >= 2)

    if rule_id == "NA2":
        return None if prev_noshow_rate is None else (prev_noshow_rate > 0.5)

    if rule_id == "NA3":
        return None if prev_total is None else (prev_total <= 2)

    if rule_id == "NA4":
        return None if last_att is None else (last_att == "no-show")

    if rule_id == "NA5":
        return None if creation_interval is None else (creation_interval > 7)

    if rule_id == "NA6":
        return None if creation_interval is None else (creation_interval <= 2)

    if rule_id == "NA7":
        if age is None or prev_att is None:
            return None
        return (age < 30) and (prev_att <= 1)

    if rule_id == "NA8":
        if diseases is None or prev_att is None:
            return None
        return (diseases <= 1) and (prev_att <= 1)

    if rule_id == "A1":
        return None if prev_att is None else (prev_att >= 3)

    if rule_id == "A2":
        return None if prev_noshow_rate is None else (prev_noshow_rate <= 0.2)

    if rule_id == "A3":
        return None if last_att is None else (last_att == "show")

    if rule_id == "A4":
        return None if prev_total is None else (prev_total >= 4)

    if rule_id == "A5":
        if creation_interval is None:
            return None
        return 3 <= creation_interval <= 7

    if rule_id == "A6":
        return None if diseases is None else (diseases >= 2)

    if rule_id == "A7":
        return None if medications is None else (medications >= 2)

    if rule_id == "A8":
        # This rule is currently disabled by default in the persisted config.
        if diseases is None or prev_att is None:
            return None
        return (diseases <= 1) and (prev_att <= 1)

    # Unknown rule id
    return None


def score_rules(features: Dict[str, Any], ruleset: Dict[str, Any]) -> Dict[str, Any]:
    """Compute score + checklist for a ruleset configuration."""

    rules = list(ruleset.get("rules", []))
    checks: Dict[str, Any] = {}

    score = 0
    max_score = 0
    triggered = []
    skipped = []

    for r in rules:
        rule_id = str(r.get("id"))
        enabled = bool(r.get("enabled", True))
        weight = int(r.get("weight", 1))

        if not enabled:
            checks[rule_id] = {
                "enabled": False,
                "triggered": None,
                "weight": weight,
                "name": r.get("name"),
                "condition": r.get("condition"),
            }
            continue

        max_score += weight
        res = _eval_rule(rule_id, features)
        if res is None:
            skipped.append(rule_id)
            checks[rule_id] = {
                "enabled": True,
                "triggered": None,
                "weight": weight,
                "name": r.get("name"),
                "condition": r.get("condition"),
            }
            continue

        if bool(res):
            score += weight
            triggered.append(rule_id)

        checks[rule_id] = {
            "enabled": True,
            "triggered": bool(res),
            "weight": weight,
            "name": r.get("name"),
            "condition": r.get("condition"),
        }

    return {
        "score": int(score),
        "max_score": int(max_score),
        "triggered": triggered,
        "skipped": skipped,
        "checks": checks,
        "min_score_confirm": int(ruleset.get("min_score_confirm", 0)),
    }


def double_verify(
    *,
    features: Dict[str, Any],
    probability_no_show: Optional[float],
    model_threshold: Optional[float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return rule scores and a conservative final label.

    Final label policy (conservative):
    - If model predicts no-show (prob >= threshold), only keep label=1 when
      non_attendance rules confirm.
    - Otherwise return label=0.

    This helps reduce false positives for flagged no-shows.
    """

    non_attendance_cfg = dict(config.get("non_attendance", {}))
    attendance_cfg = dict(config.get("attendance", {}))

    non_attendance = score_rules(features, non_attendance_cfg)
    attendance = score_rules(features, attendance_cfg)

    model_label: Optional[int] = None
    final_label: Optional[int] = None

    if probability_no_show is not None and isinstance(model_threshold, (int, float)):
        model_label = int(float(probability_no_show) >= float(model_threshold))

    status = "no_model_threshold"
    confirmed = None
    contradiction = None

    if model_label is not None:
        if model_label == 1:
            confirmed = non_attendance["score"] >= non_attendance["min_score_confirm"]
            contradiction = attendance["score"] >= attendance["min_score_confirm"]
            if contradiction and not confirmed:
                status = "contradictory_high_attendance_signals"
            else:
                status = "confirmed_no_show" if confirmed else "unconfirmed_no_show"
            final_label = 1 if confirmed else 0
        else:
            confirmed = attendance["score"] >= attendance["min_score_confirm"]
            contradiction = non_attendance["score"] >= non_attendance["min_score_confirm"]
            if contradiction and not confirmed:
                status = "contradictory_high_no_show_signals"
            else:
                status = "confirmed_show" if confirmed else "unconfirmed_show"
            final_label = 0

    return {
        "version": config.get("version", 1),
        "model": {
            "probability_no_show": probability_no_show,
            "threshold": model_threshold,
            "model_label": model_label,
        },
        "rules": {
            "non_attendance": non_attendance,
            "attendance": attendance,
        },
        "decision": {
            "final_label": final_label,
            "status": status,
            "confirmed": confirmed,
            "contradiction": contradiction,
        },
    }
