from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import appointment_routes, prediction_routes
from .db import init_db


app = FastAPI(title="Predictive No-Show API")

app.include_router(appointment_routes)
app.include_router(prediction_routes)

origins = [
    "http://localhost:5173"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()
    print("Database initialized")

@app.on_event("shutdown")
def on_shutdown():
    print("Shutting down application")


@app.get("/health")
def health():
    return {"status": "ok"}


