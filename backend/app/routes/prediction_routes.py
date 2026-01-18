from fastapi import APIRouter, HTTPException
from ..schemas import PatientInput, PredictionResponse
from ..services.prediction import predict_from_dict, info as model_info, METRICS
from ..services import db_service
from ..db import SessionLocal
from pathlib import Path
import pandas as pd


prediction_routes = APIRouter(tags=["prediction"])


@prediction_routes.get("/model-info")
def get_model_info():
    return model_info()


@prediction_routes.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    try:
        res = predict_from_dict(payload.features)
        return PredictionResponse(**res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@prediction_routes.get("/predictions/waiting")
def predictions_for_waiting(medic_id: str | None = None):
    """Predict no-show probability for appointments currently in 'En espera' (type 2).

    Optional `medic_id` query parameter filters to a single medic.
    Returns aggregated percentages and per-appointment probabilities.
    """
    db = SessionLocal()
    try:
        appts = db_service.list_appointments(db)
        # filter waiting
        waiting = [a for a in appts if int(a.appointment_type) == 2]
        if medic_id:
            waiting = [a for a in waiting if str(a.medic_id) == str(medic_id)]

        # load processed cleaned dataset (written by train pipeline)
        repo_root = Path(__file__).resolve().parents[3]
        processed_csv = repo_root / "data" / "processed" / "df_limpio.csv"
        if not processed_csv.exists():
            raise HTTPException(status_code=500, detail=f"Processed dataset not found at {processed_csv}")

        df = pd.read_csv(processed_csv)

        results = []
        probs_no_show = []
        preds_no_show = 0

        feature_columns = METRICS.get("feature_columns", [])

        for a in waiting:
            pid = str(a.patient_id)
            # look for matching patient row in processed df using Id Patient (case-insensitive)
            match_cols = [c for c in df.columns if c.lower().replace('_', ' ').strip() in ("id patient", "id_patient", "idpatient", "id patient")]
            matched_row = None
            if match_cols:
                col = match_cols[0]
                found = df[df[col].astype(str).str.strip() == pid]
                if not found.empty:
                    matched_row = found.iloc[0]
            # fallback: try matching against 'Id Patient' exact
            if matched_row is None and 'Id Patient' in df.columns:
                found = df[df['Id Patient'].astype(str).str.strip() == pid]
                if not found.empty:
                    matched_row = found.iloc[0]

            if matched_row is None:
                results.append({
                    "appointment_id": int(a.id),
                    "patient_id": pid,
                    "error": "patient data not found"
                })
                continue

            # build features dict expected by the model
            features = {c: (matched_row[c] if c in matched_row.index else None) for c in feature_columns}

            try:
                pred = predict_from_dict(features)
            except Exception as e:
                results.append({"appointment_id": int(a.id), "patient_id": pid, "error": f"prediction failed: {e}"})
                continue

            # predict_from_dict returns 'probability' as probability of class 1.
            # In the dataset mapping 0 = attended, 1 = no-show, so the returned
            # probability corresponds to no-show.
            prob_no_show = pred.get('probability')
            prob_attend = None
            if prob_no_show is not None:
                prob_no_show = float(prob_no_show)
                prob_attend = 1.0 - prob_no_show
                probs_no_show.append(prob_no_show)

            label = pred.get('label')
            # label == 1 indicates predicted no-show
            if label == 1:
                preds_no_show += 1

            results.append({
                "appointment_id": int(a.id),
                "patient_id": pid,
                "predicted_label": int(label) if label is not None else None,
                "prob_attend": prob_attend,
                "prob_no_show": prob_no_show,
            })

        total = len(waiting)
        analyzed = len([r for r in results if 'error' not in r])
        mean_prob_no_show = float(sum(probs_no_show) / len(probs_no_show)) if probs_no_show else None
        # mean probability of attending is complementary
        mean_prob_attend = float(sum((1.0 - p) for p in probs_no_show) / len(probs_no_show)) if probs_no_show else None

        percent_predicted_no_show = (preds_no_show / analyzed * 100.0) if analyzed else None
        preds_attend = (analyzed - preds_no_show) if analyzed else 0
        percent_predicted_attend = (preds_attend / analyzed * 100.0) if analyzed else None

        return {
            "total_waiting": total,
            "analyzed": analyzed,
            "mean_prob_no_show": mean_prob_no_show,
            "mean_prob_attend": mean_prob_attend,
            "percent_predicted_no_show": percent_predicted_no_show,
            "percent_predicted_attend": percent_predicted_attend,
            "per_appointment": results,
        }
    finally:
        db.close()
