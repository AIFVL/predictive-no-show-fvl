from fastapi import APIRouter, HTTPException
from ..schemas import AppointmentCreate, AppointmentOut, AppointmentUpdate
from ..db import Appointment, SessionLocal
from ..services import db_service
from ..services import training_service

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
            db, payload.medic_id, payload.patient_id, payload.hour, payload.day, payload.month, 2
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

@appointment_routes.get("/info/{appointment_id}", response_model=AppointmentOut)
def get_appointment_info(appointment_id: int):
    db = SessionLocal()
    try:
        appt = db_service.get_appointment_info(db, appointment_id)
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return appt
    finally:
        db.close()

@appointment_routes.get("/{medic_id}", response_model=list[AppointmentOut])
def get_appointments_by_medic(medic_id: str):
    db = SessionLocal()
    try:
        appts = db_service.list_appointments_by_medic(db, medic_id)
        return appts
    finally:
        db.close()



@appointment_routes.put("/{appointment_id}", response_model=AppointmentOut)
def update_appointment(appointment_id: int, payload: AppointmentUpdate):
    if not (0 <= payload.hour <= 23):
        raise HTTPException(status_code=400, detail="hour must be between 0 and 23")
    if not (1 <= payload.day <= 31):
        raise HTTPException(status_code=400, detail="day must be between 1 and 31")
    if not (1 <= payload.month <= 12):
        raise HTTPException(status_code=400, detail="month must be between 1 and 12")
    if payload.appointment_type not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="appointment_type must be 0, 1 or 2")

    db = SessionLocal()
    try:
        appt = db_service.update_appointment(
            db, appointment_id, payload.medic_id, payload.patient_id, payload.hour, payload.day, payload.month, payload.appointment_type
        )
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return appt
    except Exception as e:
        print("Error updating appointment:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@appointment_routes.patch("/type/{appointment_id}", response_model=AppointmentOut)
def update_appointment_type(appointment_id: int, appointment_type: int):
    if appointment_type not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="appointment_type must be 0, 1 or 2")
    db = SessionLocal()
    try:
        appt = db_service.update_appointment_type(
            db, appointment_id, appointment_type
        )
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        
        # Si la cita cambió a "Asistida" (0) o "No asistió" (1),
        # actualizar datos de entrenamiento y reentrenar el modelo
        if appointment_type in (0, 1):
            training_result = training_service.handle_appointment_status_change(
                appointment=appt,
                new_status=appointment_type,
            )
            # Log del resultado pero no interrumpir la respuesta
            if training_result["success"]:
                print(f"Training update: {training_result['message']}")
            else:
                print(f"Training warning: {training_result['message']}")
        
        return appt
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
        