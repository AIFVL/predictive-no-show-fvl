from typing import List

from sqlalchemy.orm import Session

from ..db import Appointment


def create_appointment(db: Session, patient_id: str, hour: int, day: int, month: int, appointment_type: int = 2) -> Appointment:
    appt = Appointment(
        patient_id=str(patient_id),
        hour=int(hour),
        day=int(day),
        month=int(month),
        appointment_type=int(appointment_type),
    )
    db.add(appt)
    db.commit()
    db.refresh(appt)
    return appt


def list_appointments(db: Session) -> List[Appointment]:
    return db.query(Appointment).order_by(Appointment.created_at.desc()).all()


def update_appointment(db: Session, appointment_id: int, patient_id: str, hour: int, day: int, month: int, appointment_type: int) -> Appointment:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appt:
        appt.patient_id = str(patient_id)
        appt.hour = int(hour)
        appt.day = int(day)
        appt.month = int(month)
        appt.appointment_type = int(appointment_type)
        db.commit()
        db.refresh(appt)
    return appt

def update_appointment_type(db: Session, appointment_id: int, appointment_type: int) -> Appointment:
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if appt:
        appt.appointment_type = int(appointment_type)
        db.commit()
        db.refresh(appt)
    return appt