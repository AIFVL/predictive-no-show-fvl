from fastapi import APIRouter, HTTPException
from ..schemas import PatientInput, PredictionResponse
from ..services.prediction import predict_from_dict, info as model_info

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
