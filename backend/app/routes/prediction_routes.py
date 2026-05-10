from fastapi import APIRouter, HTTPException
from ..schemas import PatientInput, PredictionResponse
from ..services import prediction as prediction_service
from ..services import db_service
from ..db import SessionLocal
from pathlib import Path
import hashlib
import pandas as pd


prediction_routes = APIRouter(tags=["prediction"])


def _build_features_from_row(
    row: pd.Series,
    feature_columns: list[str],
    *,
    appointment_hour: int,
    appointment_day: int,
    appointment_month: int,
) -> dict:
    features = {c: row[c] for c in row.index}
    for c in feature_columns:
        features.setdefault(c, row[c] if c in row.index else None)
    # Always prioritize the schedule currently stored in the appointment.
    features["Hour"] = appointment_hour
    features["Day"] = appointment_day
    features["Month"] = appointment_month
    return features


def _fallback_row_for_appointment(df: pd.DataFrame, patient_id: str, medic_id: str | None) -> pd.Series:
    """Pick a deterministic row when patient identifiers are unavailable in the dataset.

    The processed CSV in this repository does not contain patient_id/medic_id columns,
    so demo appointments cannot be matched directly. This fallback keeps the UI usable
    by selecting a stable reference profile from the processed dataset.
    """

    seed_text = f"{patient_id}|{medic_id or ''}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    index = int(digest[:12], 16) % len(df)
    return df.iloc[index]


def _load_processed_df() -> pd.DataFrame:
    repo_root = Path(__file__).resolve().parents[3]
    processed_csv = repo_root / "data" / "processed" / "df_limpio.csv"
    if not processed_csv.exists():
        raise HTTPException(status_code=500, detail=f"Processed dataset not found at {processed_csv}")
    return pd.read_csv(processed_csv)


def _norm_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _first_col_by_aliases(df: pd.DataFrame, aliases: set[str]) -> str | None:
    for c in df.columns:
        if _norm_col(c) in aliases:
            return c
    return None


def _predict_for_appointment(appointment, df: pd.DataFrame, feature_columns: list[str]) -> dict:
    pid = str(appointment.patient_id).strip()
    mid = str(appointment.medic_id).strip() if getattr(appointment, "medic_id", None) is not None else None

    patient_col = _first_col_by_aliases(
        df,
        {
            "patient id",
            "patientid",
            "patient_id",
            "id patient",
            "idpatient",
            "id_patient",
        },
    )
    medic_col = _first_col_by_aliases(
        df,
        {
            "medic id",
            "medicid",
            "medic_id",
            "id medic",
            "idmedic",
            "id_medic",
            "doctor id",
            "doctorid",
            "doctor_id",
        },
    )

    matched_row = None
    feature_source = "matched_dataset_row"
    if patient_col is not None:
        found = df[df[patient_col].astype(str).str.strip() == pid]
        if mid is not None and medic_col is not None and not found.empty:
            found2 = found[found[medic_col].astype(str).str.strip() == mid]
            if not found2.empty:
                found = found2
        if not found.empty:
            matched_row = found.iloc[0]

    if matched_row is None:
        matched_row = _fallback_row_for_appointment(df, pid, mid)
        feature_source = "fallback_reference_profile"

    features = _build_features_from_row(
        matched_row,
        feature_columns,
        appointment_hour=int(appointment.hour),
        appointment_day=int(appointment.day),
        appointment_month=int(appointment.month),
    )

    pred = prediction_service.predict_from_dict(features)
    prob_no_show = pred.get("probability")
    prob_attend = None
    if prob_no_show is not None:
        prob_no_show = float(prob_no_show)
        prob_attend = 1.0 - prob_no_show

    model_label = pred.get("label")
    final_label = pred.get("final_label")
    if final_label is None:
        final_label = model_label

    verification = pred.get("verification")
    verification_status = None
    non_attendance_score = None
    attendance_score = None
    if isinstance(verification, dict):
        verification_status = (verification.get("decision", {}) or {}).get("status")
        rules = verification.get("rules", {}) or {}
        non_attendance_score = (rules.get("non_attendance", {}) or {}).get("score")
        attendance_score = (rules.get("attendance", {}) or {}).get("score")

    return {
        "appointment_id": int(appointment.id),
        "patient_id": pid,
        "model_label": int(model_label) if model_label is not None else None,
        "final_label": int(final_label) if final_label is not None else None,
        "predicted_label": int(final_label) if final_label is not None else None,
        "prob_attend": prob_attend,
        "prob_no_show": prob_no_show,
        "verification_status": verification_status,
        "non_attendance_score": non_attendance_score,
        "attendance_score": attendance_score,
        "verification": verification,
        "model_analysis": pred.get("model_analysis"),
        "shap_analysis": pred.get("shap_analysis"),
        "model_version": pred.get("model_version"),
        "feature_source": feature_source,
    }


@prediction_routes.get("/model-info")
def get_model_info():
    # Load artifacts so returned metrics reflect the current model.
    prediction_service._ensure_loaded()
    return prediction_service.info()


@prediction_routes.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    try:
        res = prediction_service.predict_from_dict(payload.features)
        return PredictionResponse(**res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@prediction_routes.get("/predictions/appointment/{appointment_id}")
def prediction_for_appointment(appointment_id: int):
    db = SessionLocal()
    try:
        prediction_service._ensure_loaded()
        appointment = db_service.get_appointment_info(db, appointment_id)
        if not appointment:
            raise HTTPException(status_code=404, detail="Appointment not found")

        df = _load_processed_df()
        feature_columns = prediction_service.METRICS.get("feature_columns", [])
        return _predict_for_appointment(appointment, df, feature_columns)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
    finally:
        db.close()


@prediction_routes.get("/predictions/waiting")
def predictions_for_waiting(medic_id: str | None = None):
    """Predict no-show probability for appointments currently in 'En espera' (type 2).

    Optional `medic_id` query parameter filters to a single medic.
    Returns aggregated percentages and per-appointment probabilities.
    """
    db = SessionLocal()
    try:
        # Ensure model + metrics are loaded before using feature_columns.
        prediction_service._ensure_loaded()
        appts = db_service.list_appointments(db)
        # filter waiting
        waiting = [a for a in appts if int(a.appointment_type) == 2]
        if medic_id:
            waiting = [a for a in waiting if str(a.medic_id) == str(medic_id)]

        df = _load_processed_df()

        results = []
        probs_no_show = []
        preds_no_show_model = 0
        preds_no_show_final = 0

        feature_columns = prediction_service.METRICS.get("feature_columns", [])

        for a in waiting:
            try:
                pred = _predict_for_appointment(a, df, feature_columns)
            except Exception as e:
                results.append({"appointment_id": int(a.id), "patient_id": str(a.patient_id).strip(), "error": f"prediction failed: {e}"})
                continue

            prob_no_show = pred.get("prob_no_show")
            if prob_no_show is not None:
                probs_no_show.append(float(prob_no_show))

            model_label = pred.get("model_label")
            final_label = pred.get("final_label")
            if model_label == 1:
                preds_no_show_model += 1
            if final_label == 1:
                preds_no_show_final += 1
            results.append(pred)

        total = len(waiting)
        analyzed = len([r for r in results if 'error' not in r])
        mean_prob_no_show = float(sum(probs_no_show) / len(probs_no_show)) if probs_no_show else None
        # mean probability of attending is complementary
        mean_prob_attend = float(sum((1.0 - p) for p in probs_no_show) / len(probs_no_show)) if probs_no_show else None

        percent_model_predicted_no_show = (preds_no_show_model / analyzed * 100.0) if analyzed else None
        percent_predicted_no_show = (preds_no_show_final / analyzed * 100.0) if analyzed else None

        preds_attend = (analyzed - preds_no_show_final) if analyzed else 0
        percent_predicted_attend = (preds_attend / analyzed * 100.0) if analyzed else None

        return {
            "total_waiting": total,
            "analyzed": analyzed,
            "mean_prob_no_show": mean_prob_no_show,
            "mean_prob_attend": mean_prob_attend,
            "percent_model_predicted_no_show": percent_model_predicted_no_show,
            "percent_predicted_no_show": percent_predicted_no_show,
            "percent_predicted_attend": percent_predicted_attend,
            "per_appointment": results,
        }
    finally:
        db.close()
