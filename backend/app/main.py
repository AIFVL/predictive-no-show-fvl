from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import appointment_routes, prediction_routes
from .db import init_db, SessionLocal
from .services.seed_service import seed_appointments_2026


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
    # Seed demo data: ensure at least 20 appointments per month in 2026.
    db = SessionLocal()
    try:
        stats = seed_appointments_2026(db, min_per_month=20)
        print("Database initialized")
        print(f"Seeded appointments (2026): created={stats.get('created')} by_month={stats.get('by_month')}")
    finally:
        db.close()

@app.on_event("shutdown")
def on_shutdown():
    print("Shutting down application")


@app.get("/health")
def health():
    return {"status": "ok"}


