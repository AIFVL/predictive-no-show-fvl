from fastapi import APIRouter, HTTPException
from ..schemas import AppointmentCreate, AppointmentOut
from ..db import Appointment, SessionLocal
from ..services import db_service

appointment_routes = APIRouter(prefix="/appointments", tags=["appointments"])


@appointment_routes.post("/", response_model=AppointmentOut)
def create_appointment(payload: AppointmentCreate):
    if not (0 <= payload.hour <= 23):
        raise HTTPException(status_code=400, detail="hour must be between 0 and 23")
    if not (1 <= payload.day <= 31):
        raise HTTPException(status_code=400, detail="day must be between 1 and 31")
    if not (1 <= payload.month <= 12):
        raise HTTPException(status_code=400, detail="month must be between 1 and 12")

    db = SessionLocal()
    try:
        appt = db_service.create_appointment(
            db, payload.patient_id, payload.hour, payload.day, payload.month, payload.apointment_type
        )
        return appt
    except Exception as e:
        print("Error creating appointment:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@appointment_routes.get("/", response_model=list[AppointmentOut])
def get_appointments():
    db = SessionLocal()
    try:
        appts = db_service.list_appointments(db)
        return appts
    finally:
        db.close()


@appointment_routes.put("/{appointment_id}", response_model=AppointmentOut)
def update_appointment(appointment_id: int, payload: AppointmentCreate):
    db = SessionLocal()
    try:
        appt = db_service.update_appointment(
            db, appointment_id, payload.patient_id, payload.hour, payload.day, payload.month, payload.appointment_type
        )
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return appt
    except Exception as e:
        print("Error updating appointment:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@appointment_routes.patch("/{appointment_id}/type", response_model=AppointmentOut)
def update_appointment_type(appointment_id: int, appointment_type: int):
    db = SessionLocal()
    try:
        appt = db_service.update_appointment_type(
            db, appointment_id, appointment_type
        )
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
    except Exception as e:
        print("Error updating appointment type:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@appointment_routes.delete("/{appointment_id}", response_model=dict)
def delete_appointment(appointment_id: int):
    db = SessionLocal()
    try:
        appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        db.delete(appt)
        db.commit()
        return {"detail": "Appointment deleted successfully"}
    except Exception as e:
        print("Error deleting appointment:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        