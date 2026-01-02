from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PatientInput, PredictionResponse
from .services.prediction import predict_from_dict, info as model_info


app = FastAPI(title="Predictive No-Show API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    try:
        res = predict_from_dict(payload.features)
        return PredictionResponse(**res)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
 


